use crate::{
    common::*,
    config, message as msg,
    model::{
        self, DetectionEmbeddingInit, Discriminator, Generator, NLayerDiscriminatorInit, NormKind,
        ResnetGeneratorInit, UnetGeneratorInit, WGanGpInit,
    },
    FILE_STRFTIME,
};
use yolo_dl::model::YoloModel;

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

    // learning rate scheduler
    let mut train_step = 0;
    let mut lr_scheduler = train::utils::LrScheduler::new(&config.train.lr_schedule, 0)?;
    let mut rate_counter = train::utils::RateCounter::with_second_intertal();
    let mut lr = lr_scheduler.next();

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
        let embedding_dim = 32;

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let embedding = DetectionEmbeddingInit {
            in_c: [255, 255, 255],
            out_c: embedding_dim,
            inner_c: 32,
            norm_kind: NormKind::BatchNorm,
        }
        .build(&root / "embedding");

        let gen_model: Generator = match config.model.generator.kind {
            config::GeneratorModelKind::Resnet => ResnetGeneratorInit::<1>::default()
                .build(
                    &root / "generator",
                    embedding_dim + latent_dim,
                    image_dim,
                    64,
                )
                .into(),
            config::GeneratorModelKind::UNet => UnetGeneratorInit::<5>::default()
                .build(
                    &root / "generator",
                    embedding_dim + latent_dim,
                    image_dim,
                    64,
                )
                .into(),
        };

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.).build(&vs, lr)?;

        (vs, embedding, gen_model, opt)
    };

    // load discriminator model
    let (mut discriminator_vs, discriminator_model, mut discriminator_opt) = {
        let config::DiscriminatorModel {
            ref weights_file, ..
        } = config.model.discriminator;
        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let disc_model: Discriminator = NLayerDiscriminatorInit::<7>::default()
            .build(&root / "discriminator", image_dim, 64)
            .into();

        // let disc_model: Discriminator =
        //     PixelDiscriminatorInit::<7>::default().build(&root / "discriminator", image_dim, 64);

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.).build(&vs, lr)?;

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

    const CRITIC_STEPS: usize = 5;
    const GENERATE_STEPS: usize = 1;

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
                let det = detector_model.forward_t(image, false)?;
                let (det_loss, _) =
                    detector_loss_fn.forward(&det.shallow_clone().try_into()?, boxes);
                let image = image * 2.0 - 1.0;

                // train discriminator
                let discriminator_loss = {
                    let mut step = 0;
                    generator_vs.freeze();
                    discriminator_vs.unfreeze();

                    loop {
                        if gan_loss == config::GanLoss::WGan {
                            const WEIGHT_CLAMP: f64 = 0.01;

                            discriminator_vs.freeze();
                            discriminator_vs
                                .trainable_variables()
                                .iter_mut()
                                .for_each(|var| {
                                    let _ = var.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP);
                                });
                            discriminator_vs.unfreeze();
                        }

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
                        let real_score = discriminator_model.forward_t(&image, true);
                        let fake_score =
                            discriminator_model.forward_t(&image_recon.detach().copy(), true);

                        let discriminator_loss = match gan_loss {
                            config::GanLoss::DcGan => {
                                let ones =
                                    Tensor::ones(&[batch_size as i64], (Kind::Float, device));
                                let zeros =
                                    Tensor::zeros(&[batch_size as i64], (Kind::Float, device));

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
                            config::GanLoss::RelativisticGan => {
                                mse_loss(&real_score, &(fake_score.mean(Kind::Float) + 1.0))
                                    + mse_loss(&fake_score, &(real_score.mean(Kind::Float) - 1.0))
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

                        discriminator_opt.backward_step(&discriminator_loss);

                        step += 1;
                        if step == CRITIC_STEPS {
                            break discriminator_loss;
                        }
                    }
                };

                // train generator
                let (image_recon, det_recon_loss, generator_loss) = {
                    let mut step = 0;
                    generator_vs.unfreeze();
                    discriminator_vs.freeze();

                    loop {
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
                        let fake_score = discriminator_model.forward_t(&image_recon, true);
                        let generator_loss = match gan_loss {
                            config::GanLoss::DcGan => {
                                let ones =
                                    Tensor::ones(&[batch_size as i64], (Kind::Float, device));
                                fake_score.binary_cross_entropy_with_logits::<Tensor>(
                                    &ones,
                                    None,
                                    None,
                                    Reduction::Mean,
                                )
                            }
                            config::GanLoss::RelativisticGan => {
                                let real_score = discriminator_model.forward_t(&image, true);

                                mse_loss(&real_score, &(fake_score.mean(Kind::Float) + 1.0))
                                    + mse_loss(&fake_score, &(real_score.mean(Kind::Float) - 1.0))
                            }
                            config::GanLoss::WGan | config::GanLoss::WGanGp => {
                                (-&fake_score).mean(Kind::Float)
                            }
                        };

                        // let loss = &det_recon_loss + &generator_loss;
                        let loss = &generator_loss;

                        generator_opt.backward_step(&loss);

                        step += 1;

                        if step == GENERATE_STEPS {
                            break (image_recon, det_recon_loss, generator_loss);
                        }
                    }
                };

                let loss_log = msg::LossLog {
                    det_loss: det_loss.total_loss.into(),
                    det_recon_loss: det_recon_loss.into(),
                    discriminator_loss: discriminator_loss.into(),
                    generator_loss: generator_loss.into(),
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

    // // warm-up
    // for _ in 0..warm_up_steps {
    //     let input = Tensor::randn(
    //         &[
    //             batch_size,
    //             image_dim + 1 + latent_dim as i64,
    //             image_size,
    //             image_size,
    //         ],
    //         (Kind::Float, device),
    //     ) * 0.5
    //         + 0.5;
    //     let _ = generator.forward_t(&input, NONE_TENSORS, true)?;
    // }

    // for (train_step, _) in iter::repeat(()).enumerate() {
    //     let record = train_rx.blocking_recv().unwrap()?;
    //     let TrainingRecord {
    //         sequence,
    //         noise,
    //         batch_index,
    //         record_index: _,
    //     } = record;

    //     let contexts = sequence[0..peek_len].iter().try_fold(
    //         None,
    //         |in_contexts, image| -> Result<_> {
    //             let indicator = Tensor::ones(
    //                 &[batch_size as i64, 1, image_size as i64, image_size as i64],
    //                 (Kind::Float, device),
    //             );
    //             let noise = noise
    //                 .view([batch_size as i64, latent_dim as i64, 1, 1])
    //                 .expand(
    //                     &[
    //                         batch_size as i64,
    //                         latent_dim as i64,
    //                         image_size as i64,
    //                         image_size as i64,
    //                     ],
    //                     false,
    //                 );

    //             let input = Tensor::cat(&[image, &noise, &indicator], 1);
    //             let GeneratorOutput {
    //                 contexts: out_contexts,
    //                 ..
    //             } = generator.forward_t(&input, in_contexts, true)?;

    //             debug_assert!(
    //                 out_contexts.iter().all(|context| !context.has_nan()),
    //                 "NaN detected"
    //             );

    //             Ok(Some(out_contexts))
    //         },
    //     )?;

    //     let (outputs, _contexts) = (0..pred_len).try_fold(
    //         (vec![], contexts),
    //         |(mut outputs, in_contexts), _index| -> Result<_> {
    //             let input = Tensor::zeros(
    //                 &[
    //                     batch_size as i64,
    //                     (image_dim + latent_dim + 1) as i64,
    //                     image_size as i64,
    //                     image_size as i64,
    //                 ],
    //                 (Kind::Float, device),
    //             );

    //             let GeneratorOutput {
    //                 output,
    //                 contexts: out_contexts,
    //             } = generator.forward_t(&input, in_contexts, true)?;

    //             outputs.push(output);
    //             Ok((outputs, Some(out_contexts)))
    //         },
    //     )?;

    //     // let true_sequence = &sequence;
    //     // let fake_sequence: Vec<_> =
    //     //     sequence[0..peek_len].iter().chain(outputs.iter()).collect();

    //     let true_sequence = &sequence[peek_len..seq_len];
    //     let fake_sequence: Vec<_> = outputs.iter().collect();

    //     let true_sample = Tensor::stack(true_sequence, 2);
    //     let fake_sample = Tensor::stack(&fake_sequence, 2);

    //     // optimize discriminator
    //     let dis_loss = {
    //         discriminator_vs.unfreeze();
    //         generator_vs.freeze();

    //         let true_score = discriminator.forward_t(&true_sample, true)?;
    //         let fake_score = discriminator.forward_t(&fake_sample.detach(), true)?;
    //         let loss = mse_loss(&true_score, &(&fake_score + 1.0))
    //             + mse_loss(&fake_score, &(true_score - 1.0));
    //         discriminator_opt.backward_step(&loss);
    //         f64::from(&loss)
    //     };

    //     // optimize generator
    //     let gen_loss = {
    //         discriminator_vs.freeze();
    //         generator_vs.unfreeze();

    //         let true_score = discriminator.forward_t(&true_sample, true)?;
    //         let fake_score = discriminator.forward_t(&fake_sample, true)?;
    //         let loss = mse_loss(&true_score, &(&fake_score - 1.0))
    //             + mse_loss(&fake_score, &(true_score + 1.0));
    //         // generator_opt.backward_step(&loss);
    //         generator_opt.zero_grad();
    //         loss.backward();
    //         // dbg!(generator.grad());
    //         generator_opt.step();
    //         f64::from(&loss)
    //     };

    //     discriminator.clamp_bn_var();
    //     generator.clamp_bn_var();

    //     // let loss = {
    //     //     let diff = true_sample - fake_sample;
    //     //     (&diff * &diff).mean(Kind::Float)
    //     // };

    //     // generator_opt.backward_step(&loss);

    //     // let gen_loss = f64::from(&loss);
    //     // let dis_loss = f64::from(&loss);

    //     info!(
    //         "batch_index = {}\tdis_loss = {:.5}\tgen_loss = {:.5}",
    //         batch_index, dis_loss, gen_loss
    //     );

    //     // save checkpoint
    //     {
    //         let save_checkpoint = save_checkpoint_steps
    //             .map(|steps| train_step % steps == 0)
    //             .unwrap_or(false);

    //         if save_checkpoint {
    //             save_checkpoint_files(
    //                 &discriminator_vs,
    //                 &generator_vs,
    //                 &checkpoint_dir,
    //                 train_step,
    //                 dis_loss,
    //                 gen_loss,
    //             )?;
    //         }
    //     }

    //     // save results
    //     {
    //         let save_image = save_image_steps
    //             .map(|steps| train_step % steps == 0)
    //             .unwrap_or(false);

    //         let true_image = save_image.then(|| {
    //             let seq: Vec<_> = true_sequence
    //                 .iter()
    //                 .map(|image| image.shallow_clone())
    //                 .collect();
    //             seq
    //         });
    //         let fake_image = save_image.then(|| {
    //             let seq: Vec<_> = fake_sequence
    //                 .iter()
    //                 .map(|&image| image.shallow_clone())
    //                 .collect();
    //             seq
    //         });

    //         let msg = LogMessage {
    //             step: train_step,
    //             dis_loss,
    //             gen_loss,
    //             learning_rate,
    //             true_image,
    //             fake_image,
    //         };

    //         log_tx.blocking_send(msg).unwrap();
    //     };
    // }

    Ok(())
}

fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
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
