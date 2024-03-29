use crate::{
    common::*,
    config, message as msg,
    model::{
        self, CustomGeneratorInit, DetectionEmbeddingInit, DetectionSimilarity, DetectorWrapper,
        Discriminator, Generator, GeneratorWrapper, ImageSequenceDiscriminatorWrapper,
        MotionBasedTransformerInit, NLayerDiscriminatorInit, ResnetGeneratorInit, Transformer,
        UnetGeneratorInit, WGanGp, WGanGpInit,
    },
    rate_meter::RateMeter,
    utils::DenseDetectionTensorListExt,
    FILE_STRFTIME,
};
use tch_goodies::{lr_schedule::LrScheduler, DenseDetectionTensorList, Ratio};
use yolo_dl::{loss::YoloLoss, model::YoloModel};

const WEIGHT_CLAMP: f64 = 0.01;

#[derive(Debug)]
struct TrainWorker {
    detector_vs: nn::VarStore,
    generator_vs: nn::VarStore,
    discriminator_vs: nn::VarStore,
    transformer_vs: nn::VarStore,
    image_seq_discriminator_vs: nn::VarStore,

    detector_opt: nn::Optimizer,
    generator_opt: nn::Optimizer,
    discriminator_opt: nn::Optimizer,
    transformer_opt: nn::Optimizer,
    image_seq_discriminator_opt: nn::Optimizer,

    detector_model: DetectorWrapper,
    generator_model: GeneratorWrapper,
    discriminator_model: Discriminator,
    transformer_model: Transformer,
    image_seq_discriminator_model: ImageSequenceDiscriminatorWrapper,

    save_detector_checkpoint: bool,
    save_discriminator_checkpoint: bool,
    save_generator_checkpoint: bool,
    save_transformer_checkpoint: bool,
    save_image_seq_discriminator_checkpoint: bool,

    checkpoint_dir: PathBuf,
    detector_loss_fn: YoloLoss,
    label_flip_prob: f64,
    gan_loss_kind: config::GanLoss,
    gp: WGanGp,
    gp_transformer: WGanGp,
}

impl TrainWorker {
    pub fn freeze_all_vs(&mut self) {
        let Self {
            detector_vs,
            generator_vs,
            discriminator_vs,
            transformer_vs,
            image_seq_discriminator_vs,
            ..
        } = self;
        detector_vs.freeze();
        generator_vs.freeze();
        discriminator_vs.freeze();
        transformer_vs.freeze();
        image_seq_discriminator_vs.freeze();
    }

    pub fn train_detector(
        &mut self,
        steps: usize,
        image: &Tensor,
        labels: &[Vec<Ratio<RectLabel>>],
        with_artifacts: bool,
        dry_run: bool,
    ) -> Result<(Option<f64>, Option<msg::WeightsAndGrads>)> {
        dbg!();
        self.freeze_all_vs();
        let Self {
            detector_vs,
            detector_model,
            detector_opt,
            detector_loss_fn,
            ..
        } = self;

        detector_vs.unfreeze();

        // train detector
        let loss = (0..steps).try_fold(None, |_, _| -> Result<_> {
            // clamp running_var in norms
            clamp_running_var(detector_vs);

            // run detector
            let real_det = detector_model.forward_t(image, true)?;
            let (real_det_loss, _) =
                detector_loss_fn.forward(&real_det.shallow_clone().try_into()?, labels);

            // optimize detector
            if !dry_run {
                detector_opt.backward_step(&real_det_loss.total_loss);
            }

            Ok(Some(f64::from(real_det_loss.total_loss)))
        })?;

        let weights_and_grads = (!dry_run && with_artifacts && loss.is_some())
            .then(|| get_weights_and_grads(detector_vs));

        Ok((loss, weights_and_grads))
    }

    pub fn train_generator(
        &mut self,
        steps: usize,
        real_image: &Tensor,
        input_det: &DenseDetectionTensorList,
        train_discriminator: bool,
        with_artifacts: bool,
        dry_run: bool,
    ) -> Result<(Option<f64>, Option<Tensor>, Option<msg::WeightsAndGrads>)> {
        self.freeze_all_vs();

        let Self {
            label_flip_prob,
            gan_loss_kind,
            ref mut generator_vs,
            ref generator_model,
            ref discriminator_model,
            ref mut generator_opt,
            ..
        } = *self;
        let mut rng = rand::thread_rng();

        generator_vs.unfreeze();

        let (loss, fake_image) = (0..steps).try_fold((None, None), |_, _| -> Result<_> {
            // clamp running_var in norms
            clamp_running_var(generator_vs);

            // generate fake image
            let fake_image = generator_model.forward_t(input_det, None, true)?;

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
            let real_score = discriminator_model.forward_t(&real_image, train_discriminator);
            let fake_score = discriminator_model.forward_t(&fake_image, train_discriminator);

            // compute loss
            let generator_loss =
                crate::model::generator_gan_loss(gan_loss_kind, &real_score, &fake_score)?;

            // optimize generator
            if !dry_run {
                generator_opt.backward_step(&generator_loss);
            }

            Ok((Some(f64::from(generator_loss)), Some(fake_image)))
        })?;

        let weights_and_grads = (!dry_run && with_artifacts && loss.is_some())
            .then(|| get_weights_and_grads(generator_vs));

        Ok((loss, fake_image, weights_and_grads))
    }

    pub fn train_discriminator(
        &mut self,
        steps: usize,
        real_image: &Tensor,
        input_det: &DenseDetectionTensorList,
        train_generator: bool,
        with_artifacts: bool,
        dry_run: bool,
    ) -> Result<(Option<f64>, Option<Tensor>, Option<msg::WeightsAndGrads>)> {
        self.freeze_all_vs();

        let Self {
            label_flip_prob,
            gan_loss_kind,
            ref mut generator_vs,
            ref mut discriminator_vs,
            ref generator_model,
            ref discriminator_model,
            ref mut discriminator_opt,
            ref gp,
            ..
        } = *self;
        let mut rng = rand::thread_rng();

        discriminator_vs.unfreeze();

        let (loss, fake_image) = (0..steps).try_fold((None, None), |_, _| -> Result<_> {
            // clamp running_var in norms
            clamp_running_var(generator_vs);

            // generate fake image
            let fake_image = generator_model
                .forward_t(input_det, None, train_generator)?
                .detach()
                .copy();

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
            let real_score = discriminator_model.forward_t(&real_image, true);
            let fake_score = discriminator_model.forward_t(&fake_image, true);

            // compute loss
            let loss = crate::model::discriminator_gan_loss(
                gan_loss_kind,
                &real_score,
                &fake_score,
                &real_image,
                &fake_image,
                gp,
                |xs, train| discriminator_model.forward_t(xs, train),
            )?;

            // optimize generator
            if !dry_run {
                discriminator_opt.backward_step(&loss);
            }

            // clip gradient
            if !dry_run && gan_loss_kind == config::GanLoss::WGan {
                discriminator_opt.clip_grad_norm(WEIGHT_CLAMP);
            }

            Ok((Some(f64::from(loss)), Some(fake_image)))
        })?;

        let weights_and_grads = (!dry_run && with_artifacts && loss.is_some())
            .then(|| get_weights_and_grads(discriminator_vs));

        Ok((loss, fake_image, weights_and_grads))
    }

    pub fn train_retraction_identity(
        &mut self,
        steps: usize,
        gt_det: &DenseDetectionTensorList,
        gt_labels: &[Vec<Ratio<RectLabel>>],
        dry_run: bool,
    ) -> Result<(Option<f64>, Option<DetectionSimilarity>)> {
        self.freeze_all_vs();
        let Self {
            // detector_vs,
            generator_vs,
            detector_model,
            generator_model,
            generator_opt,
            // detector_opt,
            detector_loss_fn,
            ..
        } = self;

        // detector_vs.unfreeze();
        generator_vs.unfreeze();

        (0..steps).try_fold((None, None), |_, _| -> Result<_> {
            // clamp running_var in norms
            clamp_running_var(generator_vs);

            let recon_image = generator_model.forward_t(gt_det, None, true)?;
            let recon_det = detector_model.forward_t(&recon_image, true)?;

            let (loss, _) =
                detector_loss_fn.forward(&recon_det.shallow_clone().try_into()?, gt_labels);
            let similarity = crate::model::dense_detection_list_similarity(gt_det, &recon_det)?;

            // optimize generator
            if !dry_run {
                generator_opt.backward_step(&loss.total_loss);
                // detector_opt.zero_grad();
                // generator_opt.zero_grad();
                // loss.total_loss.backward();
                // detector_opt.step();
                // generator_opt.step();
            }

            Ok((Some(f64::from(loss.total_loss)), Some(similarity)))
        })
    }

    pub fn train_triangular_identity(
        &mut self,
        steps: usize,
        gt_image: &Tensor,
        gt_labels: &[Vec<Ratio<RectLabel>>],
        dry_run: bool,
    ) -> Result<(Option<f64>, Option<DetectionSimilarity>)> {
        self.freeze_all_vs();
        let Self {
            // detector_vs,
            generator_vs,
            detector_model,
            generator_model,
            generator_opt,
            // detector_opt,
            detector_loss_fn,
            ..
        } = self;

        // detector_vs.unfreeze();
        generator_vs.unfreeze();

        (0..steps).try_fold((None, None), |_, _| -> Result<_> {
            // clamp running_var in norms
            clamp_running_var(generator_vs);

            let orig_det = detector_model.forward_t(gt_image, true)?;
            let fake_image = generator_model.forward_t(&orig_det, None, true)?;
            let recon_det = detector_model.forward_t(&fake_image, true)?;

            let (loss, _) =
                detector_loss_fn.forward(&recon_det.shallow_clone().try_into()?, gt_labels);

            let similarity = crate::model::dense_detection_list_similarity(&orig_det, &recon_det)?;

            // optimize generator
            if !dry_run {
                generator_opt.backward_step(&loss.total_loss);
                // detector_opt.zero_grad();
                // generator_opt.zero_grad();
                // loss.total_loss.backward();
                // detector_opt.step();
                // generator_opt.step();
            }

            Ok((Some(f64::from(loss.total_loss)), Some(similarity)))
        })
    }

    pub fn train_forward_consistency(
        &mut self,
        steps: usize,
        gt_image_seq: &[Tensor],
        gt_labels_seq: &[Vec<Vec<Ratio<RectLabel>>>],
        // _gt_det_seq: &[DenseDetectionTensorList],
        with_artifacts: bool,
        dry_run: bool,
    ) -> Result<(
        Option<f64>,
        Option<Vec<DetectionSimilarity>>,
        Option<msg::WeightsAndGrads>,
    )> {
        self.freeze_all_vs();
        let Self {
            transformer_vs,
            transformer_model,
            detector_model,
            transformer_opt,
            detector_loss_fn,
            ..
        } = self;
        transformer_vs.unfreeze();

        let seq_len = gt_image_seq.len();
        let input_len = transformer_model.input_len();
        ensure!(input_len < seq_len);

        let (loss, similarity_vec) = (0..steps).try_fold((None, None), |_, _| -> Result<_> {
            clamp_running_var(transformer_vs);
            let real_det_seq: Vec<_> = gt_image_seq
                .iter()
                .map(|image| detector_model.forward_t(image, true))
                .try_collect()?;
            let mut fake_det_seq: Vec<_> = real_det_seq[0..input_len]
                .iter()
                .map(|det| det.shallow_clone())
                .collect();

            let (total_consistency_loss, similarity_vec): (AddVal<_>, Vec<_>) =
                (0..=(seq_len - input_len - 1))
                    .map(|index| -> Result<_> {
                        let input_det_window = &real_det_seq[index..(index + input_len)];
                        let last_gt_labels = &gt_labels_seq[index + input_len];
                        let last_real_det = &real_det_seq[index + input_len];
                        // let last_gt_det = &gt_det_seq[index + input_len];
                        // dbg!(last_real_det.tensors[0].obj_prob().max());
                        let (last_fake_det, artifacts) =
                            transformer_model.forward_t(input_det_window, true, true)?;

                        ensure!(last_real_det.tensors.len() == last_fake_det.tensors.len());

                        // let (real_det_loss, _) = detector_loss_fn
                        //     .forward(&last_real_det.shallow_clone().try_into()?, last_gt_labels);
                        let (fake_det_loss, _) = detector_loss_fn
                            .forward(&last_fake_det.shallow_clone().try_into()?, last_gt_labels);
                        // let consitency_loss =
                        //     (real_det_loss.total_loss + fake_det_loss.total_loss) / 2.0;
                        let consistency_loss = fake_det_loss.total_loss
                            + artifacts
                                .unwrap()
                                .motion_norm_pixel
                                .unwrap()
                                .mean(Kind::Float)
                                .pow_tensor_scalar(2);

                        let similarity = crate::model::dense_detection_list_similarity(
                            last_real_det,
                            &last_fake_det,
                        )?;

                        // let recon_loss = artifacts.unwrap().autoencoder_recon_loss;
                        fake_det_seq.push(last_fake_det);

                        Ok((consistency_loss, similarity))
                    })
                    .try_collect::<_, Vec<_>, _>()?
                    .into_iter()
                    .unzip();

            let total_loss = total_consistency_loss.unwrap();

            // optimize
            if !dry_run {
                transformer_opt.backward_step(&total_loss);
            }

            Ok((Some(f64::from(total_loss)), Some(similarity_vec)))
        })?;

        let weights_and_grads = (!dry_run && with_artifacts && loss.is_some())
            .then(|| get_weights_and_grads(transformer_vs));

        Ok((loss, similarity_vec, weights_and_grads))
    }

    pub fn train_backward_consistency_gen(
        &mut self,
        steps: usize,
        gt_image_seq: &[Tensor],
        gt_det_seq: &[DenseDetectionTensorList],
        with_artifacts: bool,
        dry_run: bool,
    ) -> Result<(Option<f64>, Option<msg::WeightsAndGrads>)> {
        self.freeze_all_vs();
        let Self {
            gan_loss_kind,
            ref mut transformer_vs,
            ref transformer_model,
            ref generator_model,
            ref image_seq_discriminator_model,
            ref mut transformer_opt,
            ..
        } = *self;

        transformer_vs.unfreeze();

        let input_len = transformer_model.input_len();
        let noise = Tensor::randn(&[generator_model.latent_dim], FLOAT_CPU);

        let loss = (0..steps).try_fold(None, |_, _| -> Result<_> {
            clamp_running_var(transformer_vs);

            let total_consistency_loss: AddVal<_> = izip!(
                gt_image_seq.windows(input_len + 1),
                gt_det_seq.windows(input_len)
            )
            .map(|(gt_image_window, gt_det_window)| -> Result<_> {
                let (last_fake_det, _artifacts) =
                    transformer_model.forward_t(gt_det_window, true, false)?;
                let last_fake_image = generator_model.forward_t(&last_fake_det, Some(&noise), true);
                let fake_image_window: Vec<_> = {
                    let prior_image_window = gt_det_window
                        .iter()
                        .map(|gt_det| generator_model.forward_t(gt_det, Some(&noise), true));
                    chain!(prior_image_window, iter::once(last_fake_image)).try_collect()?
                };

                let real_input = &gt_image_window;
                let fake_input = &fake_image_window;

                let real_score = image_seq_discriminator_model.forward_t(real_input, true)?;
                let fake_score = image_seq_discriminator_model.forward_t(fake_input, true)?;

                // compute loss
                let consistency_loss =
                    crate::model::generator_gan_loss(gan_loss_kind, &real_score, &fake_score)?;
                // let recon_loss = artifacts.unwrap().autoencoder_recon_loss;

                Ok(consistency_loss)
            })
            .try_collect()?;

            // optimize
            // let total_loss = total_consistency_loss.unwrap() + total_recon_loss.unwrap();
            let total_loss = total_consistency_loss.unwrap();
            if !dry_run {
                transformer_opt.backward_step(&total_loss);
            }

            Ok(Some(f64::from(total_loss)))
        })?;

        let weights_and_grads = (!dry_run && with_artifacts && loss.is_some())
            .then(|| get_weights_and_grads(transformer_vs));

        Ok((loss, weights_and_grads))
    }

    pub fn train_backward_consistency_disc(
        &mut self,
        steps: usize,
        gt_image_seq: &[Tensor],
        gt_det_seq: &[DenseDetectionTensorList],
        with_artifacts: bool,
        dry_run: bool,
    ) -> Result<(Option<f64>, Option<msg::WeightsAndGrads>)> {
        self.freeze_all_vs();
        let Self {
            gan_loss_kind,
            ref mut image_seq_discriminator_vs,
            ref transformer_model,
            ref generator_model,
            ref image_seq_discriminator_model,
            ref mut image_seq_discriminator_opt,
            ref gp_transformer,
            ..
        } = *self;

        image_seq_discriminator_vs.unfreeze();

        let input_len = transformer_model.input_len();
        let noise = Tensor::randn(&[generator_model.latent_dim], FLOAT_CPU);

        let loss = (0..steps).try_fold(None, |_, _| -> Result<_> {
            clamp_running_var(image_seq_discriminator_vs);

            let total_consistency_loss: AddVal<_> = izip!(
                gt_image_seq.windows(input_len + 1),
                gt_det_seq.windows(input_len)
            )
            .map(|(gt_image_window, gt_det_window)| -> Result<_> {
                let (last_fake_det, _artifacts) =
                    transformer_model.forward_t(gt_det_window, true, false)?;

                let last_fake_image = generator_model.forward_t(&last_fake_det, Some(&noise), true);
                let fake_image_window: Vec<_> = {
                    let prior_image_seq = gt_det_window
                        .iter()
                        .map(|gt_det| generator_model.forward_t(gt_det, Some(&noise), true));
                    chain!(prior_image_seq, iter::once(last_fake_image)).try_collect()?
                };

                let real_input_seq = gt_image_window;
                let fake_input_seq = &fake_image_window;

                let real_score = image_seq_discriminator_model.forward_t(real_input_seq, true)?;
                let fake_score = image_seq_discriminator_model.forward_t(fake_input_seq, true)?;

                let real_input_tensor = Tensor::cat(real_input_seq, 1);
                let fake_input_tensor = Tensor::cat(fake_input_seq, 1);

                // compute loss
                let consistency_loss = crate::model::discriminator_gan_loss(
                    gan_loss_kind,
                    &real_score,
                    &fake_score,
                    &real_input_tensor,
                    &fake_input_tensor,
                    gp_transformer,
                    |xs, train| image_seq_discriminator_model.model.forward_t(xs, train),
                )?;
                // let recon_loss = artifacts.unwrap().autoencoder_recon_loss;

                // clip gradient
                if !dry_run && gan_loss_kind == config::GanLoss::WGan {
                    image_seq_discriminator_opt.clip_grad_norm(WEIGHT_CLAMP);
                }

                Ok(consistency_loss)
            })
            .try_collect()?;

            // optimize
            // let total_loss = total_consistency_loss.unwrap() + total_recon_loss.unwrap();
            let total_loss = total_consistency_loss.unwrap();
            if !dry_run {
                image_seq_discriminator_opt.backward_step(&total_loss);
            }

            Ok(Some(f64::from(total_loss)))
        })?;

        let weights_and_grads = (!dry_run && with_artifacts && loss.is_some())
            .then(|| get_weights_and_grads(image_seq_discriminator_vs));

        Ok((loss, weights_and_grads))
    }

    /// Save parameters to a checkpoint file.
    fn save_checkpoint_files(&self, step: usize) -> Result<()> {
        let TrainWorker {
            ref checkpoint_dir,

            save_detector_checkpoint,
            save_discriminator_checkpoint,
            save_generator_checkpoint,
            save_transformer_checkpoint,
            save_image_seq_discriminator_checkpoint,

            ref detector_vs,
            ref generator_vs,
            ref discriminator_vs,
            ref transformer_vs,
            ref image_seq_discriminator_vs,
            ..
        } = *self;

        let timestamp = Local::now().format(FILE_STRFTIME);

        // save detector
        if save_detector_checkpoint {
            let filename = format!("detector_{}_{:06}.ckpt", timestamp, step);
            let path = checkpoint_dir.join(filename);
            detector_vs.save(&path)?;
        }

        // save discriminator
        if save_discriminator_checkpoint {
            let filename = format!("discriminator_{}_{:06}.ckpt", timestamp, step);
            let path = checkpoint_dir.join(filename);
            discriminator_vs.save(&path)?;
        }

        // save generator
        if save_generator_checkpoint {
            let filename = format!("generator_{}_{:06}.ckpt", timestamp, step);
            let path = checkpoint_dir.join(filename);
            generator_vs.save(&path)?;
        }

        // save transformer
        if save_transformer_checkpoint {
            let filename = format!("transformer_{}_{:06}.ckpt", timestamp, step);
            let path = checkpoint_dir.join(filename);
            transformer_vs.save(&path)?;
        }

        // save transformer discriminator
        if save_image_seq_discriminator_checkpoint {
            let filename = format!("image_seq_discriminator_{}_{:06}.ckpt", timestamp, step);
            let path = checkpoint_dir.join(filename);
            image_seq_discriminator_vs.save(&path)?;
        }

        Ok(())
    }

    pub fn set_lr(&mut self, lr: f64) {
        let Self {
            detector_opt,
            generator_opt,
            discriminator_opt,
            transformer_opt,
            image_seq_discriminator_opt,
            ..
        } = self;

        detector_opt.set_lr(lr);
        generator_opt.set_lr(lr);
        discriminator_opt.set_lr(lr);
        transformer_opt.set_lr(lr);
        image_seq_discriminator_opt.set_lr(lr);
    }
}

pub fn training_worker(
    config: ArcRef<config::Config>,
    num_classes: usize,
    checkpoint_dir: impl AsRef<Path>,
    mut train_rx: flume::Receiver<msg::TrainingMessage>,
    log_tx: flume::Sender<msg::LogMessage>,
) -> Result<()> {
    let image_h = config.train.image_size.get();
    let image_w = config.train.image_size.get();
    let device = config.train.device;
    let gan_loss_kind = config.loss.image_recon;
    let latent_dim = config.train.latent_dim.get();
    let image_dim = config.dataset.image_dim();
    let batch_size = config.train.batch_size.get();
    let label_flip_prob = config.train.label_flip_prob.raw();
    let critic_noise_prob = config.train.critic_noise_prob.raw();
    let train_detector = config.train.train_detector_steps > 0;
    let train_discriminator = config.train.train_discriminator_steps > 0;
    let train_generator = config.train.train_generator_steps > 0;
    let train_transformer = config.train.train_forward_consistency_steps > 0
        || config.train.train_backward_consistency_gen_steps > 0;
    let train_image_seq_discriminator = config.train.train_backward_consistency_disc_steps > 0;
    let dry_run_training = config.train.dry_run_training;
    let config::Logging {
        save_detector_checkpoint,
        save_discriminator_checkpoint,
        save_generator_checkpoint,
        save_transformer_checkpoint,
        save_image_seq_discriminator_checkpoint,
        ..
    } = config.logging;
    ensure!((0.0..=1.0).contains(&label_flip_prob));
    ensure!((0.0..=1.0).contains(&critic_noise_prob));

    // variables
    // let mut train_step = 0;
    let mut lr_scheduler = LrScheduler::new(&config.train.lr_schedule, 0)?;
    let mut rate_counter = RateMeter::with_second_intertal();
    let init_lr = lr_scheduler.next();

    let train_step_iter = 0..;
    let lr_iter = chain!(
        iter::once(init_lr),
        iter::from_fn(move || Some(lr_scheduler.next()))
    );
    let save_checkpoint_iter: Box<dyn Iterator<Item = bool>> =
        match config.logging.save_checkpoint_steps {
            Some(steps) => {
                let steps = steps.get();
                Box::new(chain!(iter::repeat(false).take(steps - 1), iter::once(true)).cycle())
            }
            None => Box::new(iter::repeat(false)),
        };
    let save_image_iter: Box<dyn Iterator<Item = bool>> = match config.logging.save_image_steps {
        Some(steps) => {
            let steps = steps.get();
            Box::new(chain!(iter::once(true), iter::repeat(false).take(steps - 1)).cycle())
        }
        None => Box::new(iter::repeat(false)),
    };
    let mut train_param_iter = izip!(
        train_step_iter,
        lr_iter,
        save_checkpoint_iter,
        save_image_iter
    );

    let gp = WGanGpInit::default().build()?;
    let gp_transformer = WGanGpInit::default().build()?;

    // load detector model
    let (detector_vs, detector_model, detector_opt) = {
        let config::DetectionModel {
            ref model_file,
            ref weights_file,
            ..
        } = config.model.detector;

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();
        let model = YoloModel::load_newslab_v1_json(root, model_file)?;
        let model = DetectorWrapper { model };

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::Adam {
            beta1: 0.937,
            beta2: 0.999,
            wd: 5e-4,
        }
        .build(&vs, init_lr)?;

        (vs, model, opt)
    };

    // load generator model
    let (generator_vs, generator_model, generator_opt) = {
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

        let model = GeneratorWrapper {
            latent_dim: latent_dim as i64,
            embedding_model: embedding,
            generator_model: gen_model,
        };

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, init_lr)?;

        (vs, model, opt)
    };

    // load discriminator model
    let (discriminator_vs, discriminator_model, discriminator_opt) = {
        let config::DiscriminatorModel {
            num_blocks,
            ref weights_file,
            ..
        } = config.model.discriminator;
        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let channels: Vec<_> = chain!([16, 32], iter::repeat(64))
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

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, init_lr)?;

        (vs, disc_model, opt)
    };

    // load transformer model
    let (transformer_vs, transformer_model, transformer_opt) = {
        let config::TransformerModel {
            ref weights_file,
            norm,
            num_input_detections,
            ..
        } = config.model.transformer;
        // ensure!(num_input_detections == config.train.peek_len && config.train.pred_len.get() == 1);

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let model: Transformer = MotionBasedTransformerInit {
            norm_kind: norm,
            ..Default::default()
        }
        .build(&root / "transformer", num_input_detections, num_classes, 64)?
        .into();

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, init_lr)?;

        (vs, model, opt)
    };

    // load transformer discriminator model
    let (image_seq_discriminator_vs, image_seq_discriminator_model, image_seq_discriminator_opt) = {
        let config::ImageSequenceDiscriminatorModel {
            num_blocks,
            num_detections,
            norm,
            ref weights_file,
        } = config.model.image_seq_discriminator;
        // ensure!(num_detections == config.train.peek_len + config.train.pred_len.get());

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();

        let channels: Vec<_> = chain!([16, 32], iter::repeat(64))
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
        let model = ImageSequenceDiscriminatorWrapper {
            input_len: num_detections,
            model,
        };

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        let opt = nn::adam(0.5, 0.999, 0.0).build(&vs, init_lr)?;

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

    let mut worker = TrainWorker {
        detector_vs,
        discriminator_vs,
        generator_vs,
        transformer_vs,
        image_seq_discriminator_vs,

        detector_model,
        discriminator_model,
        generator_model,
        transformer_model,
        image_seq_discriminator_model,

        detector_opt,
        discriminator_opt,
        generator_opt,
        transformer_opt,
        image_seq_discriminator_opt,

        save_detector_checkpoint,
        save_discriminator_checkpoint,
        save_generator_checkpoint,
        save_transformer_checkpoint,
        save_image_seq_discriminator_checkpoint,

        checkpoint_dir: checkpoint_dir.as_ref().to_owned(),
        detector_loss_fn,
        label_flip_prob,
        gan_loss_kind,
        gp,
        gp_transformer,
    };

    // warm up
    tch::no_grad(|| -> Result<_> {
        worker.freeze_all_vs();
        let warm_up_steps = config.train.warm_up_steps;
        let mut rate_counter = RateMeter::with_second_intertal();
        info!("run warm-up for {} steps", warm_up_steps);

        for warm_up_step in 0..warm_up_steps {
            let msg = match train_rx.recv() {
                Ok(msg) => msg,
                Err(_) => break,
            };
            let msg::TrainingMessage {
                image_batch_seq: gt_image_seq,
                boxes_batch_seq: gt_labels_seq,
                ..
            } = msg.to_device(device);
            let seq_len = gt_image_seq.len();

            let gt_det_seq: Vec<_> = gt_labels_seq
                .iter()
                .map(|batch| -> Result<_> {
                    let det_vec: Vec<_> = batch
                        .iter()
                        .map(|labels| {
                            DenseDetectionTensorList::from_labels(
                                labels,
                                &worker.detector_model.model.anchors(),
                                &[image_h],
                                &[image_w],
                                num_classes,
                            )
                        })
                        .try_collect()?;
                    let det_batch = DenseDetectionTensorList::cat_batch(&det_vec)
                        .unwrap()
                        .to_device(device);
                    Ok(det_batch)
                })
                .try_collect()?;

            gt_image_seq.iter().try_for_each(|gt_image| -> Result<_> {
                debug_assert!(gt_image.kind() == Kind::Float);
                debug_assert!(f64::from(gt_image.min()) >= 0.0 && f64::from(gt_image.max()) <= 1.0);

                // detector
                if train_detector {
                    clamp_running_var(&mut worker.detector_vs);
                    let _ = worker.detector_model.forward_t(gt_image, true)?;
                }

                // generator
                if train_generator {
                    clamp_running_var(&mut worker.generator_vs);
                    let det = worker.detector_model.forward_t(gt_image, true)?;
                    let fake_image = worker.generator_model.forward_t(&det, None, true)?;
                    debug_assert_eq!(gt_image.size(), fake_image.size());
                }

                // discriminator
                if train_discriminator {
                    clamp_running_var(&mut worker.discriminator_vs);
                    let det = worker.detector_model.forward_t(gt_image, true)?;
                    let fake_image = worker.generator_model.forward_t(&det, None, true)?;
                    let _ = worker.discriminator_model.forward_t(gt_image, true);
                    let _ = worker.discriminator_model.forward_t(&fake_image, true);
                }

                Ok(())
            })?;

            let input_len = worker.transformer_model.input_len();

            if train_transformer {
                let fake_det_seq: Vec<_> = gt_image_seq
                    .iter()
                    .map(|image| worker.detector_model.forward_t(image, true))
                    .try_collect()?;

                izip!(
                    fake_det_seq.windows(input_len),
                    gt_det_seq.windows(input_len)
                )
                .try_for_each(|(fake_det_window, gt_det_window)| -> Result<_> {
                    let _ = worker
                        .transformer_model
                        .forward_t(gt_det_window, true, false)?;
                    let _ = worker
                        .transformer_model
                        .forward_t(fake_det_window, true, false)?;
                    Ok(())
                })?;
            }

            if train_image_seq_discriminator {
                let noise = Tensor::randn(&[worker.generator_model.latent_dim], FLOAT_CPU);

                let fake_image_seq: Vec<_> = gt_det_seq
                    .iter()
                    .map(|det| worker.generator_model.forward_t(det, Some(&noise), true))
                    .try_collect()?;

                izip!(
                    fake_image_seq.windows(input_len + 1),
                    gt_image_seq.windows(input_len + 1)
                )
                .try_for_each(
                    |(fake_image_window, gt_image_window)| -> Result<_> {
                        let _ = worker
                            .image_seq_discriminator_model
                            .forward_t(gt_image_window, true)?;
                        let _ = worker
                            .image_seq_discriminator_model
                            .forward_t(fake_image_window, true)?;
                        Ok(())
                    },
                )?;
            }

            rate_counter.add(seq_len as f64);

            if let Some(batch_rate) = rate_counter.rate() {
                let record_rate = batch_rate * batch_size as f64;
                info!(
                    "warm-up step: {}\t{:.2} batch/s\t{:.2} sample/s",
                    warm_up_step, batch_rate, record_rate
                );
            } else {
                info!("warm-up step: {}", warm_up_step);
            }
        }

        Ok(())
    })?;

    info!("start training");

    while let Ok(msg) = train_rx.recv() {
        let (train_step, lr, save_checkpoint, save_image) = train_param_iter.next().unwrap();
        worker.set_lr(lr);

        let msg::TrainingMessage {
            batch_index: _,
            image_batch_seq: gt_image_seq,
            boxes_batch_seq: gt_labels_seq,
            ..
        } = msg.to_device(device);
        let seq_len = gt_image_seq.len();

        let gt_det_seq: Vec<_> = gt_labels_seq
            .iter()
            .map(|batch| -> Result<_> {
                let det_vec: Vec<_> = batch
                    .iter()
                    .map(|labels| -> Result<_> {
                        let det = DenseDetectionTensorList::from_labels(
                            labels,
                            &worker.detector_model.model.anchors(),
                            &[image_h],
                            &[image_w],
                            num_classes,
                        )?;
                        Ok(det)
                    })
                    .try_collect()?;
                let det_batch = DenseDetectionTensorList::cat_batch(&det_vec)
                    .unwrap()
                    .to_device(device);
                Ok(det_batch)
            })
            .try_collect()?;

        let (
            detector_loss,
            detector_weights,
            discriminator_loss,
            discriminator_weights,
            generator_loss,
            generator_weights,
            retraction_identity_loss,
            retraction_identity_similarity,
            triangular_identity_loss,
            triangular_identity_similarity,
            _generator_generated_image_seq,
        ): (
            Last<_>,
            Last<_>,
            Last<_>,
            Last<_>,
            Last<_>,
            Last<_>,
            Last<_>,
            Last<_>,
            Last<_>,
            Last<_>,
            Vec<_>,
        ) = izip!(&gt_image_seq, &gt_labels_seq, &gt_det_seq)
            .map(|(gt_image, gt_labels, gt_det)| -> Result<_> {
                debug_assert!(gt_image.kind() == Kind::Float);
                debug_assert!(
                    f64::from(gt_image.min()) >= -1.0 && f64::from(gt_image.max()) <= 1.0
                );

                // train detector
                dbg!();
                let (detector_loss, detector_weights) = worker.train_detector(
                    config.train.train_detector_steps,
                    gt_image,
                    gt_labels,
                    true,
                    dry_run_training,
                )?;
                dbg!();

                // train discriminator
                let (discriminator_loss, generated_image_1, discriminator_weights) = worker
                    .train_discriminator(
                        config.train.train_discriminator_steps,
                        gt_image,
                        gt_det,
                        train_generator,
                        true,
                        dry_run_training,
                    )?;

                // train generator
                let (generator_loss, generated_image_2, generator_weights) = worker
                    .train_generator(
                        config.train.train_generator_steps,
                        gt_image,
                        gt_det,
                        train_discriminator,
                        true,
                        dry_run_training,
                    )?;

                let generated_image = generated_image_1.and(generated_image_2);

                // train retraction identity
                let (retraction_identity_loss, retraction_identity_similarity) = worker
                    .train_retraction_identity(
                        config.train.train_retraction_identity_steps,
                        gt_det,
                        gt_labels,
                        dry_run_training,
                    )?;

                let (triangular_identity_loss, triangular_identity_similarity) = worker
                    .train_triangular_identity(
                        config.train.train_triangular_identity_steps,
                        gt_image,
                        gt_labels,
                        dry_run_training,
                    )?;

                // create logs
                Ok((
                    detector_loss,
                    detector_weights,
                    discriminator_loss,
                    discriminator_weights,
                    generator_loss,
                    generator_weights,
                    retraction_identity_loss,
                    retraction_identity_similarity,
                    triangular_identity_loss,
                    triangular_identity_similarity,
                    generated_image,
                ))
            })
            .try_collect::<_, Vec<_>, _>()?
            .into_iter()
            .unzip_n();

        // train forward time consistency
        let (forward_consistency_loss, forward_consistency_similarity_seq, transformer_weights_1) =
            worker.train_forward_consistency(
                config.train.train_forward_consistency_steps,
                &gt_image_seq,
                &gt_labels_seq,
                // &gt_det_seq,
                true,
                dry_run_training,
            )?;

        // train backward time consistency (adversarial step)
        let (backward_consistency_disc_loss, image_seq_discriminator_weights) = worker
            .train_backward_consistency_disc(
                config.train.train_backward_consistency_disc_steps,
                &gt_image_seq,
                &gt_det_seq,
                true,
                dry_run_training,
            )?;

        // train backward time consistency (generator step)
        let (backward_consistency_gen_loss, transformer_weights_2) = worker
            .train_backward_consistency_gen(
                config.train.train_backward_consistency_gen_steps,
                &gt_image_seq,
                &gt_det_seq,
                true,
                dry_run_training,
            )?;

        let transformer_weights = transformer_weights_1.or(transformer_weights_2);

        let tuple = (train_transformer && save_image)
            .then(|| {
                tch::no_grad(|| -> Result<_> {
                    let TrainWorker {
                        detector_model,
                        generator_model,
                        transformer_model,
                        ..
                    } = &mut worker;

                    let input_len = transformer_model.input_len();
                    let noise = Tensor::randn(&[generator_model.latent_dim], FLOAT_CPU);

                    let detector_det_seq: Vec<_> = gt_image_seq
                        .iter()
                        .map(|gt_image| detector_model.forward_t(gt_image, true))
                        .try_collect()?;
                    let generator_image_seq: Vec<_> = detector_det_seq
                        .iter()
                        .map(|det| generator_model.forward_t(det, Some(&noise), true))
                        .try_collect()?;

                    let mut input_det_buffer = detector_det_seq.shallow_clone();

                    let (transformer_det_seq, transformer_artifacts_seq, transformer_image_seq) =
                        (0..=(seq_len - input_len - 1))
                            .map(|index| {
                                let input_det_window =
                                    &input_det_buffer[index..(index + input_len)];
                                let (pred_det, artifacts) = transformer_model
                                    .forward_t(input_det_window, true, true)
                                    .unwrap();
                                let recon_image = generator_model
                                    .forward_t(&pred_det, Some(&noise), true)
                                    .unwrap();

                                input_det_buffer.push(pred_det.shallow_clone());

                                (pred_det, artifacts.unwrap(), recon_image)
                            })
                            .unzip_n_vec();

                    Ok((
                        detector_det_seq.to_device(Device::Cpu),
                        generator_image_seq.to_device(Device::Cpu),
                        transformer_det_seq.to_device(Device::Cpu),
                        transformer_artifacts_seq.to_device(Device::Cpu),
                        transformer_image_seq.to_device(Device::Cpu),
                    ))
                })
            })
            .transpose()?;

        let (
            detector_det_seq,
            generator_image_seq,
            transformer_det_seq,
            transformer_artifacts_seq,
            transformer_image_seq,
        ) = match tuple {
            Some((
                detector_det_seq,
                generator_image_seq,
                transformer_det_seq,
                transformer_artifacts_seq,
                transformer_image_seq,
            )) => (
                Some(detector_det_seq),
                Some(generator_image_seq),
                Some(transformer_det_seq),
                Some(transformer_artifacts_seq),
                Some(transformer_image_seq),
            ),
            None => (None, None, None, None, None),
        };

        // send to logger
        {
            let (motion_potential_pixel_seq, motion_field_pixel_seq, attention_image_seq) =
                match transformer_artifacts_seq {
                    Some(seq) => {
                        let (
                            motion_potential_pixel_seq,
                            motion_field_pixel_seq,
                            attention_image_seq,
                        ) = seq
                            .into_iter()
                            .map(|artifacts| {
                                let msg::TransformerArtifacts {
                                    motion_potential_pixel,
                                    motion_field_pixel,
                                    attention_image,
                                    ..
                                } = artifacts;
                                (motion_potential_pixel, motion_field_pixel, attention_image)
                            })
                            .unzip_n_vec();
                        let motion_potential_pixel_seq: Option<Vec<_>> =
                            motion_potential_pixel_seq.into_iter().collect();
                        let motion_field_pixel_seq: Option<Vec<_>> =
                            motion_field_pixel_seq.into_iter().collect();
                        let attention_image_seq: Option<Vec<_>> =
                            attention_image_seq.into_iter().collect();
                        (
                            motion_potential_pixel_seq,
                            motion_field_pixel_seq,
                            attention_image_seq,
                        )
                    }
                    None => (None, None, None),
                };

            let msg = msg::LogMessage::Loss(msg::Loss {
                step: train_step,
                learning_rate: lr,

                detector_loss: detector_loss.into_inner().unwrap(),
                discriminator_loss: discriminator_loss.into_inner().unwrap(),
                generator_loss: generator_loss.into_inner().unwrap(),
                retraction_identity_loss: retraction_identity_loss.into_inner().unwrap(),
                retraction_identity_similarity: retraction_identity_similarity
                    .into_inner()
                    .unwrap(),
                triangular_identity_loss: triangular_identity_loss.into_inner().unwrap(),
                triangular_identity_similarity: triangular_identity_similarity
                    .into_inner()
                    .unwrap(),
                forward_consistency_loss,
                forward_consistency_similarity_seq,
                backward_consistency_gen_loss,
                backward_consistency_disc_loss,

                detector_weights: detector_weights.into_inner().unwrap(),
                generator_weights: generator_weights.into_inner().unwrap(),
                discriminator_weights: discriminator_weights.into_inner().unwrap(),
                transformer_weights,
                image_seq_discriminator_weights,

                gt_det_seq: save_image.then(|| gt_det_seq),
                ground_truth_image_seq: save_image.then(|| gt_image_seq),
                detector_det_seq,
                generator_image_seq,
                transformer_det_seq,
                transformer_image_seq,

                motion_potential_pixel_seq,
                motion_field_pixel_seq,
                attention_image_seq,
            });

            let result = log_tx.send(msg);
            if result.is_err() {
                break;
            }
        }

        if save_checkpoint {
            worker.save_checkpoint_files(train_step)?;
        }

        // print message
        rate_counter.add(seq_len as f64);

        if let Some(batch_rate) = rate_counter.rate() {
            let record_rate = batch_rate * batch_size as f64;
            info!(
                "step: {}\tlr: {:.5}\t{:.2} batch/s\t{:.2} sample/s",
                train_step, lr, batch_rate, record_rate
            );
        } else {
            info!("step: {}\tlr: {:.5}", train_step, lr);
        }

        // update state
        // train_step += 1;
        // lr = lr_scheduler.next();
        // worker.set_lr(lr);
    }

    Ok(())
}

fn clamp_running_var(vs: &mut nn::VarStore) {
    vs.variables().iter().for_each(|(name, var)| {
        if name.ends_with(".running_var") {
            let _ = var.shallow_clone().clamp_(1e-3, 1e3);
        }
    });
}

fn get_weights_and_grads(vs: &nn::VarStore) -> msg::WeightsAndGrads {
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
        msg::WeightsAndGrads { weights, grads }
    })
}
