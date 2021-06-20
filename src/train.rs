use crate::{common::*, config, message as msg, FILE_STRFTIME};
use yolo_dl::model::YoloModel;

pub fn training_worker(
    config: ArcRef<config::Config>,
    mut train_rx: mpsc::Receiver<msg::TrainingMessage>,
    log_tx: mpsc::Sender<msg::LogMessage>,
) -> Result<()> {
    let device = config.train.device;

    // load detector model
    let (detector_vs, detector_model) = {
        let config::DetectionModel {
            ref model_file,
            ref weights_file,
            ..
        } = config.model.detector;

        let mut vs = nn::VarStore::new(device);
        let root = vs.root();
        let model = YoloModel::open_newslab_v1(root, model_file)?;

        if let Some(weights_file) = weights_file {
            vs.load_partial(weights_file)?;
        }

        (vs, model)
    };

    for msg in train_rx.blocking_recv() {
        let msg::TrainingMessage {
            sequence,
            noise,
            batch_index,
            record_index,
        } = msg;
        dbg!();
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
    dis_vs: &nn::VarStore,
    gen_vs: &nn::VarStore,
    checkpoint_dir: &Path,
    training_step: usize,
    dis_loss: f64,
    gen_loss: f64,
) -> Result<()> {
    // save discriminator
    {
        let filename = format!(
            "discriminator_{}_{:06}_{:08.5}.ckpt",
            Local::now().format(FILE_STRFTIME),
            training_step,
            dis_loss
        );
        let path = checkpoint_dir.join(filename);
        dis_vs.save(&path)?;
    }

    // save generator
    {
        let filename = format!(
            "generator_{}_{:06}_{:08.5}.ckpt",
            Local::now().format(FILE_STRFTIME),
            training_step,
            gen_loss
        );
        let path = checkpoint_dir.join(filename);
        gen_vs.save(&path)?;
    }
    Ok(())
}
