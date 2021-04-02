use anyhow::Result;
pub use chrono::{DateTime, Local};
use frame_flow::{
    common::*,
    config,
    model::{DiscriminatorInit, GeneratorInit, GeneratorOutput},
    training_stream::{TrainingRecord, TrainingStreamInit},
};
use std::{env, path::PathBuf};
use structopt::StructOpt;
use tracing_subscriber::{filter::LevelFilter, prelude::*, EnvFilter};

const FILE_STRFTIME: &str = "%Y-%m-%d-%H-%M-%S.%3f%z";

#[derive(Debug, Clone, StructOpt)]
/// Implementation for 'frame-flow' model.
pub struct Args {
    #[structopt(long, default_value = "config.json5")]
    pub config: PathBuf,
}

#[derive(Debug)]
pub struct LogMessage {
    pub step: usize,
    pub dis_loss: f64,
    pub gen_loss: f64,
    pub learning_rate: f64,
    pub true_image: Option<Vec<Tensor>>,
    pub fake_image: Option<Vec<Tensor>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let start_time = Local::now();

    // setup tracing
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true).compact();
    let filter_layer = {
        let filter = EnvFilter::from_default_env();
        let filter = if env::var("RUST_LOG").is_err() {
            filter.add_directive(LevelFilter::INFO.into())
        } else {
            filter
        };
        filter
    };

    let (tracer, _uninstall) = opentelemetry_jaeger::new_pipeline()
        .with_service_name("train")
        .install()?;
    let otel_layer = tracing_opentelemetry::OpenTelemetryLayer::new(tracer);

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    // parse config
    let latent_dim: usize = 8;

    let Args { config } = Args::from_args();
    let config::Config {
        dataset:
            config::Dataset {
                dataset_dir,
                cache_dir,
                image_size,
                image_dim,
            },
        train:
            config::Training {
                batch_size,
                device,
                peek_len,
                pred_len,
                learning_rate,
            },
        logging:
            config::Logging {
                log_dir: log_base_dir,
                save_image_steps,
                save_checkpoint_steps,
            },
    } = config::Config::load(&config)?;
    let save_image_steps = save_image_steps.map(|steps| steps.get());
    let batch_size = batch_size.get();
    let image_size = image_size.get();
    let image_dim = image_dim.get();
    let seq_len = peek_len + pred_len;
    let learning_rate = learning_rate.raw();

    // data logging
    let log_dir = log_base_dir.join(format!("{}", start_time.format(FILE_STRFTIME)));
    let checkpoint_dir = log_dir.join("checkpoints");
    let event_dir = log_dir.join("events");

    tokio::fs::create_dir_all(&checkpoint_dir).await?;
    tokio::fs::create_dir_all(&event_dir).await?;

    // load dataset
    let train_stream = TrainingStreamInit {
        dataset_dir: &dataset_dir,
        cache_dir: &cache_dir,
        file_name_digits: 8,
        image_size,
        image_dim,
        latent_dim,
        batch_size,
        device,
        seq_len,
    }
    .build()
    .await?;

    // initialize model
    let mut generator_vs = nn::VarStore::new(device);
    let mut discriminator_vs = nn::VarStore::new(device);

    let mut generator = GeneratorInit {
        ndims: 2,
        input_channels: image_dim + latent_dim + 1,
        output_channels: 3,
        num_heads: 4,
        strides: [2, 2, 2],
        block_channels: [4, 8, 16],
        context_channels: [4, 8, 16],
        repeats: [3, 3, 3],
    }
    .build(&generator_vs.root() / "generator")?;

    let mut discriminator = DiscriminatorInit {
        ndims: 3,
        ksize: 3,
        input_channels: 3,
        channels: [4, 8, 16],
        strides: [2, 2, 2],
    }
    .build(&discriminator_vs.root() / "discriminator")?;

    let mut generator_opt = nn::adam(0.5, 0.999, 0.).build(&generator_vs, learning_rate)?;
    let mut discriminator_opt = nn::adam(0.5, 0.999, 0.).build(&discriminator_vs, learning_rate)?;

    let (train_tx, mut train_rx) = tokio::sync::mpsc::channel(2);
    let (log_tx, mut log_rx) = tokio::sync::mpsc::channel(1);

    // data stream to channel worker
    let data_fut = tokio::task::spawn(async move {
        train_stream
            .stream()
            .for_each(move |record| {
                let train_tx = train_tx.clone();
                async move {
                    train_tx.send(record).await.unwrap();
                }
            })
            .await;

        Fallible::Ok(())
    })
    .map(|result| Fallible::Ok(result??));

    // training worker
    let train_fut = tokio::task::spawn_blocking(move || -> Result<()> {
        for (train_step, _) in iter::repeat(()).enumerate() {
            let record = train_rx.blocking_recv().unwrap()?;
            let TrainingRecord {
                sequence,
                noise,
                batch_index,
                record_index: _,
            } = record;

            let contexts =
                sequence[0..peek_len]
                    .iter()
                    .try_fold(None, |in_contexts, image| -> Result<_> {
                        let indicator = Tensor::ones(
                            &[batch_size as i64, 1, image_size as i64, image_size as i64],
                            (Kind::Float, device),
                        );
                        let noise = noise
                            .view([batch_size as i64, latent_dim as i64, 1, 1])
                            .expand(
                                &[
                                    batch_size as i64,
                                    latent_dim as i64,
                                    image_size as i64,
                                    image_size as i64,
                                ],
                                false,
                            );

                        let input = Tensor::cat(&[image, &noise, &indicator], 1);
                        let GeneratorOutput {
                            contexts: out_contexts,
                            ..
                        } = generator.forward_t(&input, in_contexts, true)?;

                        debug_assert!(
                            out_contexts.iter().all(|context| !context.has_nan()),
                            "NaN detected"
                        );

                        Ok(Some(out_contexts))
                    })?;

            let (outputs, _contexts) = (0..pred_len).try_fold(
                (vec![], contexts),
                |(mut outputs, in_contexts), _index| -> Result<_> {
                    let input = Tensor::zeros(
                        &[
                            batch_size as i64,
                            (image_dim + latent_dim + 1) as i64,
                            image_size as i64,
                            image_size as i64,
                        ],
                        (Kind::Float, device),
                    );

                    let GeneratorOutput {
                        output,
                        contexts: out_contexts,
                    } = generator.forward_t(&input, in_contexts, true)?;

                    outputs.push(output);
                    Ok((outputs, Some(out_contexts)))
                },
            )?;

            let true_sequence = &sequence;
            let fake_sequence: Vec<_> =
                sequence[0..peek_len].iter().chain(outputs.iter()).collect();

            let true_sample = Tensor::stack(true_sequence, 2);
            let fake_sample = Tensor::stack(&fake_sequence, 2);

            // optimize discriminator
            let dis_loss = {
                discriminator_vs.unfreeze();
                generator_vs.freeze();

                let true_score = discriminator.forward_t(&true_sample, true)?;
                let fake_score = discriminator.forward_t(&fake_sample.detach(), true)?;
                let loss = mse_loss(&true_score, &(&fake_score + 1.0))
                    + mse_loss(&fake_score, &(true_score - 1.0));

                discriminator_opt.backward_step(&loss);

                f64::from(&loss)
            };

            // optimize generator
            let gen_loss = {
                discriminator_vs.freeze();
                generator_vs.unfreeze();

                let true_score = discriminator.forward_t(&true_sample, true)?;
                let fake_score = discriminator.forward_t(&fake_sample, true)?;
                let loss = mse_loss(&true_score, &(&fake_score - 1.0))
                    + mse_loss(&fake_score, &(true_score + 1.0));
                generator_opt.backward_step(&loss);

                f64::from(&loss)
            };

            info!(
                "batch_index = {}\tdis_loss = {:.5}\tgen_loss = {:.5}",
                batch_index, dis_loss, gen_loss
            );

            // save checkpoint
            {
                let save_checkpoint = save_checkpoint_steps
                    .map(|steps| train_step % steps == 0)
                    .unwrap_or(false);

                if save_checkpoint {
                    save_checkpoint_files(
                        &discriminator_vs,
                        &generator_vs,
                        &checkpoint_dir,
                        train_step,
                        dis_loss,
                        gen_loss,
                    )?;
                }
            }

            // save results
            {
                let save_image = save_image_steps
                    .map(|steps| train_step % steps == 0)
                    .unwrap_or(false);

                let true_image = save_image.then(|| true_sequence.shallow_clone());
                let fake_image = save_image.then(|| {
                    let seq: Vec<_> = fake_sequence
                        .iter()
                        .map(|&image| image.shallow_clone())
                        .collect();
                    seq
                });

                let msg = LogMessage {
                    step: train_step,
                    dis_loss,
                    gen_loss,
                    learning_rate,
                    true_image,
                    fake_image,
                };

                log_tx.blocking_send(msg).unwrap();
            };
        }

        Ok(())
    })
    .map(|result| Fallible::Ok(result??));

    let log_fut = tokio::task::spawn(async move {
        let mut event_writer = {
            let event_path_prefix = event_dir
                .join("frame-flow")
                .into_os_string()
                .into_string()
                .unwrap();

            let event_writer = EventWriterInit::default()
                .from_prefix_async(event_path_prefix, None)
                .await?;

            event_writer
        };

        loop {
            let msg = log_rx.recv().await.unwrap();
            let LogMessage {
                step,
                dis_loss,
                gen_loss,
                learning_rate,
                true_image,
                fake_image,
            } = msg;
            let step = step as i64;

            event_writer
                .write_scalar_async("loss/discriminator_loss", step, dis_loss as f32)
                .await?;
            event_writer
                .write_scalar_async("loss/generator_loss", step, gen_loss as f32)
                .await?;
            event_writer
                .write_scalar_async("params/learning_rate", step, learning_rate as f32)
                .await?;

            if let Some(true_image) = true_image {
                for (index, image) in true_image.into_iter().enumerate() {
                    event_writer
                        .write_image_list_async(format!("image/true_image/{}", index), step, image)
                        .await?;
                }
            }

            if let Some(fake_image) = fake_image {
                for (index, image) in fake_image.into_iter().enumerate() {
                    event_writer
                        .write_image_list_async(format!("image/fake_image/{}", index), step, image)
                        .await?;
                }
            }
        }

        Fallible::Ok(())
    })
    .map(|result| Fallible::Ok(result??));

    // run all tasks
    futures::try_join!(data_fut, train_fut)?;

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
