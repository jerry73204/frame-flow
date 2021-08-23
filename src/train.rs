use crate::{
    common::*,
    config, message as msg,
    model::{
        CustomGeneratorInit, DetectionEmbeddingInit, Discriminator, Generator,
        NLayerDiscriminatorInit, ResnetGeneratorInit, TransformerInit, UnetGeneratorInit,
        WGanGpInit,
    },
    FILE_STRFTIME,
};
use yolo_dl::model::YoloModel;

const WEIGHT_CLAMP: f64 = 0.01;

pub fn training_worker(
    config: ArcRef<config::Config>,
    num_classes: usize,
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
    let train_detector_steps = config.train.train_detector_steps;
    let train_discriminator_steps = config.train.train_discriminator_steps;
    let train_generator_steps = config.train.train_generator_steps;
    let train_consistency_steps = config.train.train_consistency_steps;
    let train_transformer_steps = config.train.train_transformer_steps;
    let train_transformer_discriminator_steps = config.train.train_transformer_discriminator_steps;
    let warm_up_steps = config.train.warm_up_steps;
    let critic_noise_prob = config.train.critic_noise_prob.raw();
    let train_detector = train_detector_steps > 0;
    let train_discriminator = train_discriminator_steps > 0;
    let train_generator = train_generator_steps > 0;
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
    let (mut detector_vs, mut detector_model, mut detector_opt) = {
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

        let opt = nn::Adam {
            beta1: 0.937,
            beta2: 0.999,
            wd: 5e-4,
        }
        .build(&vs, lr)?;

        (vs, model, opt)
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
            inner_c: 128,
            norm_kind: config.model.generator.norm,
            ..Default::default()
        }
        .build(
            &root / "embedding",
            &config.model.detection_embedding.channels,
            embedding_dim,
            &config.model.detection_embedding.num_blocks,
        )?;

        let gen_model: Generator = match config.model.generator.kind {
            config::GeneratorModelKind::Resnet => ResnetGeneratorInit {
                norm_kind: config.model.generator.norm,
                num_scale_blocks: 4,
                num_blocks: 1,
                ..Default::default()
            }
            .build(
                &root / "generator",
                embedding_dim + latent_dim,
                image_dim,
                128,
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
            config::GeneratorModelKind::Custom => CustomGeneratorInit {
                norm_kind: config.model.generator.norm,
                num_scale_blocks: 3,
                num_blocks: 3,
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
            num_blocks,
            ref weights_file,
            ..
        } = config.model.discriminator;
        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let channels: Vec<_> = chain!(array::IntoIter::new([16, 32]), iter::repeat(64))
            .take(num_blocks)
            .collect();
        let disc_model: Discriminator = NLayerDiscriminatorInit {
            norm_kind: config.model.discriminator.norm,
            num_blocks,
            ..Default::default()
        }
        .build(&root / "discriminator", image_dim, &channels)?
        .into();

        // let disc_model: Discriminator =
        //     PixelDiscriminatorInit::<7>::default().build(&root / "discriminator", image_dim, 64);

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, lr)?;

        (vs, disc_model, opt)
    };

    // load transformer model
    let (mut transformer_vs, transformer_model, mut transformer_opt) = {
        let config::TransformerModel {
            ref weights_file,
            norm,
            num_input_detections,
            num_resnet_blocks,
            num_scaling_blocks,
            num_down_sample,
        } = config.model.transformer;
        ensure!(num_input_detections == config.train.peek_len && config.train.pred_len.get() == 1);

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let model = TransformerInit {
            norm_kind: norm,
            num_resnet_blocks,
            num_scaling_blocks,
            num_down_sample,
            ..Default::default()
        }
        .build(&root / "transformer", num_input_detections, num_classes, 1)?;

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, lr)?;

        (vs, model, opt)
    };

    // load transformer discriminator model
    let (
        mut transformer_discriminator_vs,
        transformer_discriminator_model,
        mut transformer_discriminator_opt,
    ) = {
        let config::TransformerDiscriminatorModel {
            num_blocks,
            num_detections,
            norm,
            ref weights_file,
        } = config.model.transformer_discriminator;
        ensure!(num_detections == config.train.peek_len + config.train.pred_len.get());

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let channels: Vec<_> = chain!(array::IntoIter::new([16, 32]), iter::repeat(64))
            .take(num_blocks)
            .collect();
        let model: Discriminator = NLayerDiscriminatorInit {
            norm_kind: norm,
            num_blocks,
            ..Default::default()
        }
        .build(
            &root / "transformer_discriminator",
            image_dim * num_detections,
            &channels,
        )?
        .into();

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, lr)?;

        (vs, model, opt)
    };

    // initialize detector loss function
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

    let get_weights_and_grads = |vs: &nn::VarStore| {
        tch::no_grad(|| {
            let mut orig_weights: Vec<_> = vs.variables().into_iter().collect();
            orig_weights.sort_by_cached_key(|(name, _)| name.to_owned());

            let weights: Vec<_> = orig_weights
                .iter()
                .map(|(name, var)| {
                    let max = f64::from(var.abs().max());
                    (name.to_owned(), max)
                })
                .collect();

            let grads: Vec<_> = orig_weights
                .iter()
                .filter(|(_name, var)| var.requires_grad())
                .map(|(name, var)| {
                    let grad = f64::from(var.grad().abs().max());
                    (name.to_owned(), grad)
                })
                .collect();
            (weights, grads)
        })
    };

    let clamp_running_var = |vs: &mut nn::VarStore| {
        vs.variables().iter().for_each(|(name, var)| {
            if name.ends_with(".running_var") {
                let _ = var.shallow_clone().clamp_(1e-3, 1e3);
            }
        });
    };

    // warm up
    info!("run warm-up for {} steps", warm_up_steps);
    tch::no_grad(|| -> Result<_> {
        detector_vs.freeze();
        generator_vs.freeze();
        discriminator_vs.freeze();

        for _round in 0..warm_up_steps {
            let msg = match train_rx.blocking_recv() {
                Some(msg) => msg,
                None => break,
            };
            let msg::TrainingMessage {
                batch_index: _,
                image_batch_seq,
                ..
            } = msg.to_device(device);

            image_batch_seq
                .iter()
                .try_for_each(|orig_image| -> Result<_> {
                    debug_assert!(orig_image.kind() == Kind::Float);
                    debug_assert!(
                        f64::from(orig_image.min()) >= 0.0 && f64::from(orig_image.max()) <= 1.0
                    );

                    // clamp running_var in norms
                    if train_detector {
                        clamp_running_var(&mut detector_vs);
                    }
                    if train_generator {
                        clamp_running_var(&mut generator_vs);
                    }
                    if train_discriminator {
                        clamp_running_var(&mut discriminator_vs);
                    }

                    // run detector
                    let det = detector_model.forward_t(orig_image, train_detector)?;
                    let real_image = orig_image * 2.0 - 1.0;

                    // run generator
                    let fake_image = {
                        let embedding = embedding_model.forward_t(&det, train_generator)?;
                        let noise = {
                            let (b, _c, h, w) = embedding.size4()?;
                            Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                                .expand(&[b, latent_dim as i64, h, w], false)
                        };
                        let input = Tensor::cat(&[embedding, noise], 1);
                        generator_model.forward_t(&input, train_generator)
                    };
                    debug_assert_eq!(real_image.size(), fake_image.size());

                    // run discriminator
                    let _real_score =
                        discriminator_model.forward_t(&real_image, train_discriminator);
                    let _fake_score =
                        discriminator_model.forward_t(&fake_image, train_discriminator);

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

        // train detector, GAN and consistency
        transformer_vs.freeze();
        transformer_discriminator_vs.freeze();

        let (loss_log_seq, image_log_seq) = izip!(&image_batch_seq, &boxes_batch_seq)
            .enumerate()
            .map(|(_seq_index, (orig_image, boxes))| -> Result<_> {
                debug_assert!(orig_image.kind() == Kind::Float);
                debug_assert!(
                    f64::from(orig_image.min()) >= 0.0 && f64::from(orig_image.max()) <= 1.0
                );
                let real_image = orig_image * 2.0 - 1.0;

                // train detector
                let real_det_loss = {
                    generator_vs.freeze();
                    discriminator_vs.freeze();
                    detector_vs.unfreeze();

                    (0..train_detector_steps).try_fold(None, |_, _| -> Result<_> {
                        // clamp running_var in norms
                        clamp_running_var(&mut detector_vs);

                        // run detector
                        let real_det = detector_model.forward_t(orig_image, true)?;
                        let (real_det_loss, _) =
                            detector_loss_fn.forward(&real_det.shallow_clone().try_into()?, boxes);

                        // optimize detector
                        detector_opt.backward_step(&real_det_loss.total_loss);

                        Ok(Some(f64::from(real_det_loss.total_loss)))
                    })?
                };

                // save detector gradients
                let (detector_weights, detector_grads) = if real_det_loss.is_some() {
                    let (weights, grads) = get_weights_and_grads(&detector_vs);
                    (Some(weights), Some(grads))
                } else {
                    (None, None)
                };

                // train discriminator
                let (discriminator_loss, fake_image_disc) = {
                    generator_vs.freeze();
                    discriminator_vs.unfreeze();
                    detector_vs.freeze();

                    (0..train_discriminator_steps).try_fold((None, None), |_, _| -> Result<_> {
                        // clamp running_var in norms
                        clamp_running_var(&mut discriminator_vs);
                        if train_generator {
                            clamp_running_var(&mut generator_vs);
                        }
                        if train_detector {
                            clamp_running_var(&mut detector_vs);
                        }

                        // run detector
                        let real_det = detector_model.forward_t(orig_image, train_detector)?;

                        // generate fake image
                        let fake_image = {
                            let embedding =
                                embedding_model.forward_t(&real_det, train_generator)?;
                            let noise = {
                                let (b, _c, h, w) = embedding.size4()?;
                                Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                                    .expand(&[b, latent_dim as i64, h, w], false)
                            };
                            let input = Tensor::cat(&[embedding, noise], 1);
                            generator_model.forward_t(&input, train_generator)
                        };
                        debug_assert_eq!(real_image.size(), fake_image.size());
                        debug_assert!(!fake_image.has_nan());
                        let orig_fake_image = fake_image.shallow_clone();

                        // augmentation
                        let (real_image, fake_image) = {
                            let real_image = real_image.shallow_clone();
                            let fake_image = fake_image.copy().detach();

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

                        // run discriminator
                        let real_score = discriminator_model.forward_t(&real_image, true);
                        let fake_score = discriminator_model.forward_t(&fake_image, true);

                        // compute discriminator loss
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
                                    &real_image,
                                    &fake_image,
                                    |xs, train| discriminator_model.forward_t(xs, train),
                                    true,
                                )?;
                                discriminator_loss + gp_loss
                            }
                        };
                        debug_assert!(!discriminator_loss.has_nan());

                        // clip weights for classical Wasserstain GAN
                        if gan_loss == config::GanLoss::WGan {
                            discriminator_opt.clip_grad_norm(WEIGHT_CLAMP);
                        }

                        // optimize discriminator
                        discriminator_opt.backward_step(&discriminator_loss);

                        Ok((Some(f64::from(discriminator_loss)), Some(orig_fake_image)))
                    })?
                };

                // save discriminator gradients
                let (discriminator_weights, discriminator_grads) = if discriminator_loss.is_some() {
                    let (weights, grads) = get_weights_and_grads(&discriminator_vs);
                    (Some(weights), Some(grads))
                } else {
                    (None, None)
                };

                // train generator
                let (generator_loss, fake_image_gen) = {
                    generator_vs.unfreeze();
                    discriminator_vs.freeze();
                    detector_vs.freeze();

                    (0..train_generator_steps).try_fold((None, None), |_, _| -> Result<_> {
                        // clamp running_var in norms
                        clamp_running_var(&mut generator_vs);
                        if train_discriminator {
                            clamp_running_var(&mut discriminator_vs);
                        }
                        if train_detector {
                            clamp_running_var(&mut detector_vs);
                        }

                        // run detector on real image
                        let real_det = detector_model.forward_t(orig_image, train_detector)?;

                        // generate fake image
                        let fake_image = {
                            let embedding = embedding_model.forward_t(&real_det, true)?;
                            let noise = {
                                let (b, _c, h, w) = embedding.size4()?;
                                Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                                    .expand(&[b, latent_dim as i64, h, w], false)
                            };
                            let input = Tensor::cat(&[embedding, noise], 1);
                            generator_model.forward_t(&input, true)
                        };
                        debug_assert_eq!(real_image.size(), fake_image.size());

                        // augmentation

                        let (real_image, fake_image) = {
                            let real_image = real_image.shallow_clone();

                            let label_flip = rng.gen_bool(label_flip_prob);
                            if label_flip {
                                (fake_image, real_image)
                            } else {
                                (real_image, fake_image)
                            }
                        };

                        // run discriminator
                        let real_score =
                            discriminator_model.forward_t(&real_image, train_discriminator);
                        let fake_score =
                            discriminator_model.forward_t(&fake_image, train_discriminator);

                        // compute generator loss
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

                        // optimize generator
                        generator_opt.backward_step(&generator_loss);

                        Ok((Some(f64::from(generator_loss)), Some(fake_image)))
                    })?
                };

                // save generator gradients
                let (generator_weights, generator_grads) = if generator_loss.is_some() {
                    let (weights, grads) = get_weights_and_grads(&generator_vs);
                    (Some(weights), Some(grads))
                } else {
                    (None, None)
                };

                // train consistency
                let (fake_det_loss, fake_image_cons) = {
                    generator_vs.unfreeze();
                    discriminator_vs.freeze();
                    detector_vs.unfreeze();

                    (0..train_consistency_steps).try_fold((None, None), |_, _| -> Result<_> {
                        // clamp running var in norms
                        clamp_running_var(&mut generator_vs);
                        clamp_running_var(&mut detector_vs);

                        // run detector image real image
                        let real_det = detector_model.forward_t(orig_image, true)?;

                        // generate fake image
                        let fake_image = {
                            let embedding = embedding_model.forward_t(&real_det, true)?;
                            let noise = {
                                let (b, _c, h, w) = embedding.size4()?;
                                Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                                    .expand(&[b, latent_dim as i64, h, w], false)
                            };
                            let input = Tensor::cat(&[embedding, noise], 1);
                            generator_model.forward_t(&input, true)
                        };

                        // run detector on fake image and compute loss
                        let fake_det =
                            detector_model.forward_t(&(&fake_image / 2.0 + 0.5), true)?;
                        let (fake_det_loss, _) =
                            detector_loss_fn.forward(&fake_det.shallow_clone().try_into()?, boxes);

                        // optimize consistency
                        detector_opt.zero_grad();
                        generator_opt.zero_grad();
                        fake_det_loss.total_loss.backward();
                        detector_opt.step();
                        generator_opt.step();

                        Ok((Some(f64::from(fake_det_loss.total_loss)), Some(fake_image)))
                    })?
                };

                let fake_image = fake_image_cons.or(fake_image_gen).or(fake_image_disc);

                // create logs
                let loss_log = msg::LossLog {
                    real_det_loss,
                    fake_det_loss,
                    discriminator_loss,
                    generator_loss,
                    detector_grads,
                    discriminator_grads,
                    generator_grads,
                    detector_weights,
                    discriminator_weights,
                    generator_weights,
                };
                let image_log = msg::ImageLog {
                    true_image: real_image.to_device(Device::Cpu),
                    fake_image: fake_image.to_device(Device::Cpu),
                };

                Ok((loss_log, image_log))
            })
            .try_collect::<_, Vec<_>, _>()?
            .into_iter()
            .unzip_n_vec();

        // train transformer
        detector_vs.freeze();
        generator_vs.freeze();
        discriminator_vs.freeze();

        transformer_vs.unfreeze();
        transformer_discriminator_vs.freeze();

        for _ in 0..train_transformer_steps {
            ensure!(config.train.peek_len >= 1);
            ensure!(config.train.pred_len.get() == 1);

            let input_det_seq: Vec<_> = image_batch_seq[0..config.train.peek_len]
                .iter()
                .map(|image| detector_model.forward_t(image, false))
                .try_collect()?;

            let real_bboxes = &boxes_batch_seq[config.train.peek_len];
            let fake_det = transformer_model.forward_t(&input_det_seq, true)?;

            let (fake_det_loss, _) =
                detector_loss_fn.forward(&fake_det.shallow_clone().try_into()?, real_bboxes);

            // optimize
            transformer_opt.backward_step(&fake_det_loss.total_loss);
        }

        // train discriminator
        transformer_vs.freeze();
        transformer_discriminator_vs.unfreeze();

        for _ in 0..train_transformer_discriminator_steps {
            ensure!(config.train.peek_len >= 1);
            ensure!(config.train.pred_len.get() == 1);

            let input_det_seq: Vec<_> = image_batch_seq[0..config.train.peek_len]
                .iter()
                .map(|image| detector_model.forward_t(image, false))
                .try_collect()?;

            let real_image_seq = &image_batch_seq;

            let fake_det = transformer_model.forward_t(&input_det_seq, true)?;
            let fake_image = {
                let embedding = embedding_model.forward_t(&fake_det, false)?;
                let noise = {
                    let (b, _c, h, w) = embedding.size4()?;
                    Tensor::randn(&[b, latent_dim as i64, 1, 1], (Kind::Float, device))
                        .expand(&[b, latent_dim as i64, h, w], false)
                };
                let input = Tensor::cat(&[embedding, noise], 1);
                generator_model.forward_t(&input, false)
            };
            let fake_image_seq: Vec<_> = chain!(
                &image_batch_seq[0..config.train.peek_len],
                iter::once(&fake_image)
            )
            .collect();
            debug_assert!(real_image_seq.len() == fake_image_seq.len());

            let real_input = Tensor::cat(real_image_seq, 1);
            let fake_input = Tensor::cat(&fake_image_seq, 1);

            let real_score = transformer_discriminator_model.forward_t(&real_input, true);
            let fake_score = transformer_discriminator_model.forward_t(&fake_input, true);

            // compute discriminator loss
            let loss = match gan_loss {
                config::GanLoss::DcGan => {
                    let ones = (Tensor::rand(&[batch_size as i64], (Kind::Float, device)) * 0.1
                        + 0.9)
                        .set_requires_grad(false);
                    let zeros = (Tensor::rand(&[batch_size as i64], (Kind::Float, device)) * 0.1)
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
                    let ones = (Tensor::rand(&[batch_size as i64], (Kind::Float, device)) * 0.1
                        + 0.9)
                        .set_requires_grad(false);
                    let zeros = (Tensor::rand(&[batch_size as i64], (Kind::Float, device)) * 0.1)
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
                    let discriminator_loss = (&fake_score - &real_score).mean(Kind::Float);

                    let gp_loss = gp.forward(
                        &real_input,
                        &fake_input,
                        |xs, train| transformer_discriminator_model.forward_t(xs, train),
                        true,
                    )?;
                    discriminator_loss + gp_loss
                }
            };
            debug_assert!(!loss.has_nan());

            // clip weights for classical Wasserstain GAN
            if gan_loss == config::GanLoss::WGan {
                transformer_discriminator_opt.clip_grad_norm(WEIGHT_CLAMP);
            }

            // optimize
            transformer_discriminator_opt.backward_step(&loss);
        }

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
                let config::Logging {
                    save_detector_checkpoint,
                    save_discriminator_checkpoint,
                    save_generator_checkpoint,
                    save_transformer_checkpoint,
                    save_transformer_discriminator_checkpoint,
                    ..
                } = config.logging;

                save_checkpoint_files(
                    save_detector_checkpoint.then(|| &detector_vs),
                    save_discriminator_checkpoint.then(|| &discriminator_vs),
                    save_generator_checkpoint.then(|| &generator_vs),
                    save_transformer_checkpoint.then(|| &transformer_vs),
                    save_transformer_discriminator_checkpoint
                        .then(|| &transformer_discriminator_vs),
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
    det_vs: Option<&nn::VarStore>,
    dis_vs: Option<&nn::VarStore>,
    gen_vs: Option<&nn::VarStore>,
    trans_vs: Option<&nn::VarStore>,
    trans_dis_vs: Option<&nn::VarStore>,
    checkpoint_dir: impl AsRef<Path>,
    training_step: usize,
) -> Result<()> {
    let timestamp = Local::now().format(FILE_STRFTIME);
    let checkpoint_dir = checkpoint_dir.as_ref();

    // save detector
    if let Some(det_vs) = det_vs {
        let filename = format!("detector_{}_{:06}.ckpt", timestamp, training_step,);
        let path = checkpoint_dir.join(filename);
        det_vs.save(&path)?;
    }

    // save discriminator
    if let Some(dis_vs) = dis_vs {
        let filename = format!("discriminator_{}_{:06}.ckpt", timestamp, training_step,);
        let path = checkpoint_dir.join(filename);
        dis_vs.save(&path)?;
    }

    // save generator
    if let Some(gen_vs) = gen_vs {
        let filename = format!("generator_{}_{:06}.ckpt", timestamp, training_step,);
        let path = checkpoint_dir.join(filename);
        gen_vs.save(&path)?;
    }

    // save transformer
    if let Some(trans_vs) = trans_vs {
        let filename = format!("transformer_{}_{:06}.ckpt", timestamp, training_step,);
        let path = checkpoint_dir.join(filename);
        trans_vs.save(&path)?;
    }

    // save transformer discriminator
    if let Some(trans_dis_vs) = trans_dis_vs {
        let filename = format!(
            "transformer_discriminator_{}_{:06}.ckpt",
            timestamp, training_step,
        );
        let path = checkpoint_dir.join(filename);
        trans_dis_vs.save(&path)?;
    }

    Ok(())
}
