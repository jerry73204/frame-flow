use crate::{
    common::*,
    config, message as msg,
    model::{
        self, DetectionEmbeddingInit, Discriminator, Generator, NLayerDiscriminatorInit,
        ResnetGeneratorInit, UnetGeneratorInit, WGanGpInit,
    },
    FILE_STRFTIME,
};
use yolo_dl::model::YoloModel;

const WEIGHT_CLAMP: f64 = 0.01;

pub fn training_worker(
    config: ArcRef<config::Config>,
    checkpoint_dir: impl AsRef<Path>,
    mut train_rx: mpsc::Receiver<msg::TrainingMessage>,
    log_tx: mpsc::Sender<msg::LogMessage>,
) -> Result<()> {
    let device = config.train.device;
    let gan_loss = config.loss.image_recon;
    let latent_dim = config.train.latent_dim.get();
    let image_dim = config.dataset.image_dim();
    let batch_size = config.train.batch_size.get();
    let label_flip_prob = config.train.label_flip_prob.raw();
    let critic_steps = config.train.critic_steps.get();
    let generate_steps = config.train.generate_steps.get();
    let warm_up_steps = config.train.warm_up_steps;
    let critic_noise_prob = config.train.critic_noise_prob.raw();
    ensure!((0.0..=1.0).contains(&label_flip_prob));
    ensure!((0.0..=1.0).contains(&critic_noise_prob));

    // variables
    let mut train_step = 0;
    let mut lr_scheduler = train::utils::LrScheduler::new(&config.train.lr_schedule, 0)?;
    let mut rate_counter = train::utils::RateCounter::with_second_intertal();
    let mut lr = lr_scheduler.next();
    let mut rng = rand::thread_rng();
    let gp = WGanGpInit::default().build()?;

    // load detector model
    let (mut detector_vs, mut detector_model) = {
        let config::DetectionModel {
            ref model_file,
            ref weights_file,
            ..
        } = config.model.detector;

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();
        let model = YoloModel::load_newslab_v1_json(root, model_file)?;

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        (vs, model)
    };

    // load generator model
    let (mut generator_vs, embedding_model, generator_model, mut generator_opt) = {
        let config::GeneratorModel {
            ref weights_file, ..
        } = config.model.generator;
        let embedding_dim = 128;

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let embedding = DetectionEmbeddingInit {
            in_c: [255, 255, 255],
            out_c: embedding_dim,
            inner_c: 128,
            norm_kind: config.model.generator.norm,
        }
        .build(&root / "embedding");

        let gen_model: Generator = match config.model.generator.kind {
            config::GeneratorModelKind::Resnet => ResnetGeneratorInit {
                norm_kind: config.model.generator.norm,
                ..Default::default()
            }
            .build(
                &root / "generator",
                embedding_dim + latent_dim,
                image_dim,
                128,
                4,
                1,
            )
            .into(),
            config::GeneratorModelKind::UNet => UnetGeneratorInit::<5> {
                norm_kind: config.model.generator.norm,
                ..Default::default()
            }
            .build(
                &root / "generator",
                embedding_dim + latent_dim,
                image_dim,
                128,
            )
            .into(),
        };

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, lr)?;

        (vs, embedding, gen_model, opt)
    };

    // load discriminator model
    let (mut discriminator_vs, discriminator_model, mut discriminator_opt) = {
        let config::DiscriminatorModel {
            ref weights_file, ..
        } = config.model.discriminator;
        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let disc_model: Discriminator = NLayerDiscriminatorInit::<8> {
            norm_kind: config.model.discriminator.norm,
            ..Default::default()
        }
        .build(
            &root / "discriminator",
            image_dim,
            [16, 32, 64, 64, 64, 64, 64, 64],
        )
        .into();

        // let disc_model: Discriminator =
        //     PixelDiscriminatorInit::<7>::default().build(&root / "discriminator", image_dim, 64);

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, lr)?;

        (vs, disc_model, opt)
    };

    let (_loss_vs, detector_loss_fn) = {
        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let det_loss_fn = config
            .loss
            .detector
            .yolo_loss_init()
            .build(&root / "detector_loss")?;
        (vs, det_loss_fn)
    };

    // warm up
    info!("run warm-up for {} steps", warm_up_steps);
    tch::no_grad(|| -> Result<_> {
        detector_vs.freeze();
        generator_vs.freeze();
        discriminator_vs.freeze();

        for round in 0..warm_up_steps {
            let msg = match train_rx.blocking_recv() {
                Some(msg) => msg,
                None => break,
            };
            let msg::TrainingMessage {
                batch_index: _,
                image_batch_seq,
                ..
            } = msg.to_device(device);

            image_batch_seq.iter().try_for_each(|image| -> Result<_> {
                debug_assert!(image.kind() == Kind::Float);
                debug_assert!(f64::from(image.min()) >= 0.0 && f64::from(image.max()) <= 1.0);

                let det = detector_model.forward_t(image, false)?;
                let image = image * 2.0 - 1.0;

                // clamp running_var in norms
                discriminator_vs.variables().iter().for_each(|(name, var)| {
                    if name.ends_with(".running_var") {
                        let _ = var.shallow_clone().clamp_(1e-3, 1e3);
                    }
                });

                let image_recon = {
                    let embedding = embedding_model.forward_t(&det, true)?;
                    let noise = {
                        let (b, _c, h, w) = embedding.size4()?;
                        Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                            .expand(&[b, latent_dim as i64, h, w], false)
                    };
                    let input = Tensor::cat(&[embedding, noise], 1);
                    generator_model.forward_t(&input, true)
                };
                debug_assert_eq!(image.size(), image_recon.size());
                let _real_score = discriminator_model.forward_t(&image, true);
                let _fake_score = discriminator_model.forward_t(&image_recon.copy().detach(), true);

                Ok(())
            })?;
        }

        Ok(())
    })?;

    info!("start training");
    while let Some(msg) = train_rx.blocking_recv() {
        let msg::TrainingMessage {
            batch_index: _,
            image_batch_seq,
            boxes_batch_seq,
            ..
        } = msg.to_device(device);
        let seq_len = image_batch_seq.len();

        detector_vs.freeze();

        let log_seq: Vec<_> = izip!(&image_batch_seq, &boxes_batch_seq)
            .enumerate()
            .map(|(_seq_index, (image, boxes))| -> Result<_> {
                debug_assert!(image.kind() == Kind::Float);
                debug_assert!(f64::from(image.min()) >= 0.0 && f64::from(image.max()) <= 1.0);

                let det = detector_model.forward_t(image, false)?;
                let (det_loss, _) =
                    detector_loss_fn.forward(&det.shallow_clone().try_into()?, boxes);
                let image = image * 2.0 - 1.0;

                // train discriminator
                let (discriminator_loss, disc_grads) = {
                    let mut step = 0;
                    generator_vs.freeze();
                    discriminator_vs.unfreeze();

                    loop {
                        // clamp running_var in norms
                        tch::no_grad(|| {
                            discriminator_vs.variables().iter().for_each(|(name, var)| {
                                if name.ends_with(".running_var") {
                                    let _ = var.shallow_clone().clamp_(1e-3, 1e3);
                                }
                            });
                        });

                        let image_recon = {
                            let embedding = embedding_model.forward_t(&det, true)?;
                            let noise = {
                                let (b, _c, h, w) = embedding.size4()?;
                                Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                                    .expand(&[b, latent_dim as i64, h, w], false)
                            };
                            let input = Tensor::cat(&[embedding, noise], 1);
                            generator_model.forward_t(&input, true)
                        };
                        debug_assert_eq!(image.size(), image_recon.size());
                        debug_assert!(!image_recon.has_nan());

                        let (real_image, fake_image) = {
                            let real_image = image.shallow_clone();
                            let fake_image = image_recon.copy().detach();

                            let label_flip = rng.gen_bool(label_flip_prob);
                            let (real_image, fake_image) = if label_flip {
                                (fake_image, real_image)
                            } else {
                                (real_image, fake_image)
                            };

                            let critic_noise = rng.gen_bool(critic_noise_prob);
                            let (real_image, fake_image) = if critic_noise {
                                let real_noise =
                                    (real_image.randn_like() * 1e-5).set_requires_grad(false);
                                let fake_noise =
                                    (fake_image.randn_like() * 1e-5).set_requires_grad(false);

                                (real_image + real_noise, fake_image + fake_noise)
                            } else {
                                (real_image, fake_image)
                            };

                            (real_image, fake_image)
                        };

                        let real_score = discriminator_model.forward_t(&real_image, true);
                        let fake_score = discriminator_model.forward_t(&fake_image, true);

                        let discriminator_loss = match gan_loss {
                            config::GanLoss::DcGan => {
                                let ones =
                                    (Tensor::rand(&[batch_size as i64], (Kind::Float, device))
                                        * 0.1
                                        + 0.9)
                                        .set_requires_grad(false);
                                let zeros =
                                    (Tensor::rand(&[batch_size as i64], (Kind::Float, device))
                                        * 0.1)
                                        .set_requires_grad(false);

                                real_score.binary_cross_entropy_with_logits::<Tensor>(
                                    &ones,
                                    None,
                                    None,
                                    Reduction::Mean,
                                ) + fake_score.binary_cross_entropy_with_logits::<Tensor>(
                                    &zeros,
                                    None,
                                    None,
                                    Reduction::Mean,
                                )
                            }
                            config::GanLoss::RaSGan => {
                                let ones =
                                    (Tensor::rand(&[batch_size as i64], (Kind::Float, device))
                                        * 0.1
                                        + 0.9)
                                        .set_requires_grad(false);
                                let zeros =
                                    (Tensor::rand(&[batch_size as i64], (Kind::Float, device))
                                        * 0.1)
                                        .set_requires_grad(false);

                                bce_loss(&real_score - &fake_score.mean(Kind::Float), ones)
                                    + bce_loss(&fake_score - &real_score.mean(Kind::Float), zeros)
                            }
                            config::GanLoss::RaLsGan => {
                                mse_loss(&real_score, fake_score.mean(Kind::Float) + 1.0)
                                    + mse_loss(fake_score, real_score.mean(Kind::Float) - 1.0)
                            }
                            config::GanLoss::WGan => (&fake_score - &real_score).mean(Kind::Float),
                            config::GanLoss::WGanGp => {
                                let discriminator_loss =
                                    (&fake_score - &real_score).mean(Kind::Float);

                                let gp_loss = gp.forward(
                                    &image,
                                    &image_recon,
                                    |xs, train| discriminator_model.forward_t(xs, train),
                                    true,
                                )?;
                                discriminator_loss + gp_loss
                            }
                        };
                        debug_assert!(!discriminator_loss.has_nan());

                        if gan_loss == config::GanLoss::WGan {
                            discriminator_opt.clip_grad_norm(WEIGHT_CLAMP);
                        }

                        discriminator_opt.backward_step(&discriminator_loss);

                        step += 1;
                        if step == critic_steps {
                            let disc_grads: Vec<_> = tch::no_grad(|| {
                                let vars = discriminator_vs.variables();
                                let mut vars: Vec<_> = vars
                                    .iter()
                                    .filter(|(name, var)| {
                                        (name.ends_with(".weight") || name.ends_with(".bias"))
                                            && var.requires_grad()
                                    })
                                    .collect();
                                vars.sort_by_cached_key(|(name, _var)| name.to_owned());
                                vars.into_iter()
                                    .map(|(name, var)| {
                                        let grad = f64::from(var.grad().abs().max());
                                        (name.to_owned(), grad)
                                    })
                                    .collect()
                            });

                            break (discriminator_loss, disc_grads);
                        }
                    }
                };

                // train generator
                let (image_recon, det_recon_loss, generator_loss, gen_grads) = {
                    let mut step = 0;
                    generator_vs.unfreeze();
                    discriminator_vs.freeze();

                    loop {
                        // clamp running_var in norms
                        tch::no_grad(|| {
                            generator_vs.variables().iter().for_each(|(name, var)| {
                                if name.ends_with(".running_var") {
                                    let _ = var.shallow_clone().clamp_(1e-3, 1e3);
                                }
                            });
                        });

                        let image_recon = {
                            let embedding = embedding_model.forward_t(&det, true)?;
                            let noise = {
                                let (b, _c, h, w) = embedding.size4()?;
                                Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                                    .expand(&[b, latent_dim as i64, h, w], false)
                            };
                            let input = Tensor::cat(&[embedding, noise], 1);
                            generator_model.forward_t(&input, true)
                        };
                        debug_assert_eq!(image.size(), image_recon.size());
                        let det_recon =
                            detector_model.forward_t(&(&image_recon / 2.0 + 0.5), false)?;
                        let det_recon_loss = model::dense_detectino_similarity(&det, &det_recon)?;
                        // let (det_recon_loss, _) =
                        //     detector_loss_fn.forward(&det_recon.shallow_clone().try_into()?, boxes);
                        let real_score = discriminator_model.forward_t(&image, true);
                        let fake_score = discriminator_model.forward_t(&image_recon, true);

                        let label_flip = rng.gen_bool(label_flip_prob);
                        let (real_score, fake_score) = if label_flip {
                            (fake_score, real_score)
                        } else {
                            (real_score, fake_score)
                        };

                        let generator_loss = match gan_loss {
                            config::GanLoss::DcGan => {
                                let ones =
                                    (Tensor::rand(&[batch_size as i64], (Kind::Float, device))
                                        * 0.1
                                        + 0.9)
                                        .set_requires_grad(false);

                                fake_score.binary_cross_entropy_with_logits::<Tensor>(
                                    &ones,
                                    None,
                                    None,
                                    Reduction::Mean,
                                )
                            }
                            config::GanLoss::RaSGan => {
                                let ones =
                                    (Tensor::rand(&[batch_size as i64], (Kind::Float, device))
                                        * 0.1
                                        + 0.9)
                                        .set_requires_grad(false);
                                let zeros =
                                    (Tensor::rand(&[batch_size as i64], (Kind::Float, device))
                                        * 0.1)
                                        .set_requires_grad(false);

                                bce_loss(&real_score - &fake_score.mean(Kind::Float), zeros)
                                    + bce_loss(&fake_score - &real_score.mean(Kind::Float), ones)
                            }
                            config::GanLoss::RaLsGan => {
                                mse_loss(&real_score, fake_score.mean(Kind::Float) - 1.0)
                                    + mse_loss(fake_score, real_score.mean(Kind::Float) + 1.0)
                            }
                            config::GanLoss::WGan | config::GanLoss::WGanGp => {
                                (-&fake_score).mean(Kind::Float)
                            }
                        };
                        debug_assert!(!generator_loss.has_nan());

                        // let loss = &det_recon_loss + &generator_loss;
                        let loss = &generator_loss;
                        generator_opt.backward_step(loss);

                        step += 1;

                        if step == generate_steps {
                            let gen_grads: Vec<_> = tch::no_grad(|| {
                                let vars = generator_vs.variables();
                                let mut vars: Vec<_> = vars
                                    .iter()
                                    .filter(|(name, var)| {
                                        (name.ends_with(".weight") || name.ends_with(".bias"))
                                            && var.requires_grad()
                                    })
                                    .collect();
                                vars.sort_by_cached_key(|(name, _var)| name.to_owned());
                                vars.into_iter()
                                    .map(|(name, var)| {
                                        let grad = f64::from(var.grad().abs().max());
                                        (name.to_owned(), grad)
                                    })
                                    .collect()
                            });

                            break (image_recon, det_recon_loss, generator_loss, gen_grads);
                        }
                    }
                };

                let loss_log = msg::LossLog {
                    det_loss: det_loss.total_loss.into(),
                    det_recon_loss: det_recon_loss.into(),
                    discriminator_loss: discriminator_loss.into(),
                    generator_loss: generator_loss.into(),
                    disc_grads,
                    gen_grads,
                };
                let image_log = msg::ImageLog {
                    true_image: image.to_device(Device::Cpu),
                    fake_image: image_recon.to_device(Device::Cpu),
                };

                Ok((loss_log, image_log))
            })
            .try_collect()?;

        let (loss_log_seq, image_log_seq) = log_seq.into_iter().unzip_n_vec();

        // send to logger
        {
            let msg = msg::LogMessage::Loss {
                step: train_step,
                learning_rate: lr,
                sequence: loss_log_seq,
            };

            let result = log_tx.blocking_send(msg);
            if result.is_err() {
                break;
            }
        }

        if let Some(save_image_steps) = config.logging.save_image_steps {
            if train_step % save_image_steps.get() == 0 {
                let msg = msg::LogMessage::Image {
                    step: train_step,
                    sequence: image_log_seq,
                };

                let result = log_tx.blocking_send(msg);
                if result.is_err() {
                    break;
                }
            }
        }

        if let Some(save_checkpoint_steps) = config.logging.save_checkpoint_steps {
            if train_step % save_checkpoint_steps.get() == 0 {
                save_checkpoint_files(
                    &detector_vs,
                    &discriminator_vs,
                    &generator_vs,
                    &checkpoint_dir,
                    train_step,
                )?;
            }
        }

        // print message
        rate_counter.add(seq_len as f64);

        if let Some(batch_rate) = rate_counter.rate() {
            let record_rate = batch_rate * batch_size as f64;
            info!(
                "step: {}\tlr: {:.5}\t{:.2} batch/s\t{:.2} sample/s",
                train_step,
                lr_scheduler.lr(),
                batch_rate,
                record_rate
            );
        } else {
            info!("step: {}\tlr: {:.5}", train_step, lr_scheduler.lr());
        }

        // update state
        train_step += 1;
        lr = lr_scheduler.next();
        discriminator_opt.set_lr(lr);
        generator_opt.set_lr(lr);
    }

    Ok(())
}

fn bce_loss(pred: impl Borrow<Tensor>, target: impl Borrow<Tensor>) -> Tensor {
    pred.borrow().binary_cross_entropy_with_logits::<Tensor>(
        target.borrow(),
        None,
        None,
        Reduction::Mean,
    )
}

fn mse_loss(pred: impl Borrow<Tensor>, target: impl Borrow<Tensor>) -> Tensor {
    pred.borrow().mse_loss(target.borrow(), Reduction::Mean)
}

/// Save parameters to a checkpoint file.
fn save_checkpoint_files(
    det_vs: &nn::VarStore,
    dis_vs: &nn::VarStore,
    gen_vs: &nn::VarStore,
    checkpoint_dir: impl AsRef<Path>,
    training_step: usize,
) -> Result<()> {
    let checkpoint_dir = checkpoint_dir.as_ref();

    // save detector
    {
        let filename = format!(
            "detector_{}_{:06}.ckpt",
            Local::now().format(FILE_STRFTIME),
            training_step,
        );
        let path = checkpoint_dir.join(filename);
        det_vs.save(&path)?;
    }

    // save discriminator
    {
        let filename = format!(
            "discriminator_{}_{:06}.ckpt",
            Local::now().format(FILE_STRFTIME),
            training_step,
        );
        let path = checkpoint_dir.join(filename);
        dis_vs.save(&path)?;
    }

    // save generator
    {
        let filename = format!(
            "generator_{}_{:06}.ckpt",
            Local::now().format(FILE_STRFTIME),
            training_step,
        );
        let path = checkpoint_dir.join(filename);
        gen_vs.save(&path)?;
    }
    Ok(())
}
