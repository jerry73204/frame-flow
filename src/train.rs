use crate::{
    common::*,
    config, message as msg,
    model::{
        self, DetectionEmbeddingInit, GanLoss, NLayerDiscriminatorInit, NormKind,
        ResnetGeneratorInit,
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
    let latent_dim = config.train.latent_dim.get();
    let image_dim = config.dataset.image_dim();
    let batch_size = config.train.batch_size.get() as i64;

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

        let gen_model = ResnetGeneratorInit::<9>::default().build(
            &root / "generator",
            embedding_dim + latent_dim,
            image_dim,
            64,
        );

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.).build(&vs, config.train.learning_rate.raw())?;

        (vs, embedding, gen_model, opt)
    };

    // load discriminator model
    let (mut discriminator_vs, discriminator_model, mut discriminator_opt) = {
        let config::DiscriminatorModel {
            ref weights_file, ..
        } = config.model.discriminator;
        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let disc_model =
            NLayerDiscriminatorInit::<7>::default().build(&root / "discriminator", image_dim, 64);

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.).build(&vs, config.train.learning_rate.raw())?;

        (vs, disc_model, opt)
    };

    // let (loss_vs, detector_loss_fn, gan_loss_fn, det_recon_loss_fn) = {
    //     let vs = nn::VarStore::new(device);
    //     let root = vs.root();
    //     let det_loss_fn = config
    //         .loss
    //         .detector
    //         .yolo_loss_init()
    //         .build(&root / "detector_loss")?;
    //     let image_recon_loss_fn = GanLoss::new(
    //         &root / "image_reconstruction_loss",
    //         &config.loss.image_recon,
    //         Reduction::Mean,
    //     );
    //     let det_recon_loss_fn = GanLoss::new(
    //         &root / "detection_reconstruction_loss",
    //         &config.loss.det_recon,
    //         Reduction::Mean,
    //     );
    //     (vs, det_loss_fn, image_recon_loss_fn, det_recon_loss_fn)
    // };

    let mut train_step = 0;

    while let Some(msg) = train_rx.blocking_recv() {
        let msg::TrainingMessage {
            batch_index: _,
            image_batch_seq,
            noise_seq,
            boxes_batch_seq,
        } = msg;

        detector_vs.freeze();

        let log_seq: Vec<_> = izip!(&image_batch_seq, &noise_seq, &boxes_batch_seq)
            .enumerate()
            .map(|(_seq_index, (image, noise, _boxes))| -> Result<_> {
                let det = detector_model.forward_t(image, false)?;

                // train discriminator
                let discriminator_loss = {
                    generator_vs.freeze();
                    discriminator_vs.unfreeze();

                    let image_recon = {
                        let embedding = embedding_model.forward_t(&det, true)?;
                        let noise = {
                            let (b, c) = noise.size2()?;
                            let (_b, _c, h, w) = embedding.size4()?;
                            noise.view([b, c, 1, 1]).expand(&[b, c, h, w], false)
                        };
                        let input = Tensor::cat(&[embedding, noise], 1);
                        generator_model.forward_t(&input, true)
                    };
                    debug_assert_eq!(image.size(), image_recon.size());

                    let image_score = discriminator_model.forward_t(&image, true);

                    let image_recon_score = discriminator_model.forward_t(&image_recon, true);

                    let discriminator_loss =
                        mse_loss(&image_score, &(image_recon_score.mean(Kind::Float) + 1.0))
                            + mse_loss(&image_recon_score, &(image_score.mean(Kind::Float) - 1.0));
                    debug_assert_eq!(discriminator_loss.numel(), 1);

                    discriminator_opt.backward_step(&discriminator_loss);

                    discriminator_loss
                };

                // train generator
                let (image_recon, det_recon_loss, generator_loss) = {
                    generator_vs.unfreeze();
                    discriminator_vs.freeze();

                    let image_recon = {
                        let embedding = embedding_model.forward_t(&det, true)?;
                        let noise = {
                            let (b, c) = noise.size2()?;
                            let (_b, _c, h, w) = embedding.size4()?;
                            noise.view([b, c, 1, 1]).expand(&[b, c, h, w], false)
                        };
                        let input = Tensor::cat(&[embedding, noise], 1);
                        generator_model.forward_t(&input, true)
                    };
                    debug_assert_eq!(image.size(), image_recon.size());

                    let det_recon = detector_model.forward_t(&image_recon, true)?;

                    // let det_loss =
                    //     detector_loss_fn.forward(&det.shallow_clone().try_into()?, boxes);
                    let det_recon_loss = model::dense_detectino_similarity(&det, &det_recon)?;
                    debug_assert_eq!(det_recon_loss.numel(), 1);

                    let image_score = discriminator_model.forward_t(&image, true);

                    let image_recon_score = discriminator_model.forward_t(&image_recon, true);

                    let generator_loss =
                        mse_loss(&image_score, &(image_recon_score.mean(Kind::Float) + 1.0))
                            + mse_loss(&image_recon_score, &(image_score.mean(Kind::Float) - 1.0));
                    debug_assert_eq!(generator_loss.numel(), 1);

                    generator_opt.backward_step(&(&det_recon_loss + &generator_loss));

                    (image_recon, det_recon_loss, generator_loss)
                };

                let loss_log = msg::LossLog {
                    det_recon_loss: det_recon_loss.into(),
                    discriminator_loss: discriminator_loss.into(),
                    generator_loss: generator_loss.into(),
                };
                let image_log = msg::ImageLog {
                    true_image: image.shallow_clone(),
                    fake_image: image_recon,
                };

                Ok((loss_log, image_log))
            })
            .try_collect()?;

        let (loss_log_seq, image_log_seq) = log_seq.into_iter().unzip_n_vec();

        {
            let msg = msg::LogMessage::Loss {
                step: train_step,
                learning_rate: config.train.learning_rate.raw(),
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

        train_step += 1;
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
