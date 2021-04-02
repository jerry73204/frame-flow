use anyhow::Result;
use frame_flow::{
    common::*,
    config,
    model::{DiscriminatorInit, GeneratorInit, GeneratorOutput},
    training_stream::{TrainingRecord, TrainingStreamInit},
};
use std::{env, path::PathBuf};
use structopt::StructOpt;
use tracing_subscriber::{filter::LevelFilter, prelude::*, EnvFilter};

#[derive(Debug, Clone, StructOpt)]
/// Implementation for 'frame-flow' model.
pub struct Args {
    #[structopt(long, default_value = "config.json5")]
    pub config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
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
    } = config::Config::load(&config)?;
    let batch_size = batch_size.get();
    let image_size = image_size.get();
    let image_dim = image_dim.get();
    let seq_len = peek_len + pred_len;
    let learning_rate = learning_rate.raw();

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

    let mut generator_vs = nn::VarStore::new(device);
    let mut discriminator_vs = nn::VarStore::new(device);

    let mut generator = GeneratorInit {
        ndims: 2,
        input_channels: image_dim + latent_dim + 1,
        output_channels: 3,
        num_heads: 4,
        strides: [2, 2, 2],
        block_channels: [8, 16, 32],
        context_channels: [5, 10, 20],
        repeats: [2, 2, 2],
    }
    .build(&generator_vs.root() / "generator")?;

    let mut discriminator = DiscriminatorInit {
        ndims: 3,
        ksize: 3,
        input_channels: 3,
        channels: [8, 16, 32],
        strides: [2, 2, 2],
    }
    .build(&discriminator_vs.root() / "discriminator")?;

    let mut generator_opt = nn::adam(0.5, 0.999, 0.).build(&generator_vs, learning_rate)?;
    let mut discriminator_opt = nn::adam(0.5, 0.999, 0.).build(&discriminator_vs, learning_rate)?;

    let (train_tx, mut train_rx) = tokio::sync::mpsc::channel(2);

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

    let train_fut = tokio::task::spawn_blocking(move || -> Result<()> {
        loop {
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

            let true_sample = Tensor::stack(&sequence, 2);
            let fake_sample = {
                let seq: Vec<_> = sequence[0..peek_len].iter().chain(outputs.iter()).collect();
                Tensor::stack(&seq, 2)
            };

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
        }
    })
    .map(|result| Fallible::Ok(result??));

    futures::try_join!(data_fut, train_fut)?;

    Ok(())
}

fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
}
