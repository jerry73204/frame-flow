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
            },
    } = config::Config::load(&config)?;

    let seq_len = peek_len + pred_len;

    // load dataset
    let train_stream = TrainingStreamInit {
        dataset_dir: &dataset_dir,
        cache_dir: &cache_dir,
        file_name_digits: 8,
        image_size: image_size.get(),
        image_dim: image_dim.get(),
        latent_dim,
        batch_size: batch_size.get(),
        device,
        seq_len,
    }
    .build()
    .await?;

    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let mut generator = GeneratorInit {
        ndims: 2,
        input_channels: 3 + latent_dim,
        output_channels: 3,
        num_heads: 4,
        strides: [2, 2, 2],
        block_channels: [8, 16, 32],
        context_channels: [5, 10, 20],
        repeats: [2, 2, 2],
    }
    .build(&root / "generator")?;

    let discriminator = DiscriminatorInit {
        ndims: 3,
        ksize: 3,
        input_channels: 3,
        channels: [8, 16, 32],
        strides: [2, 2, 2],
    }
    .build(&root / "discriminator")?;

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
                        let (batch_size, _c, height, width) = image.size4().unwrap();
                        let noise = noise
                            .view([batch_size, latent_dim as i64, 1, 1])
                            .expand(&[batch_size, latent_dim as i64, height, width], false);

                        let input = Tensor::cat(&[image, &noise], 1);

                        let GeneratorOutput {
                            contexts: out_contexts,
                            ..
                        } = generator.forward_t(&input, in_contexts, true)?;

                        Ok(Some(out_contexts))
                    })?;

            info!("batch_index = {}", batch_index);
        }
    })
    .map(|result| Fallible::Ok(result??));

    futures::try_join!(data_fut, train_fut)?;

    Ok(())
}
