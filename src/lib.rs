pub mod common;
pub mod config;
pub mod dataset;
pub mod logging;
pub mod message;
pub mod model;
pub mod train;
pub mod training_stream;

pub(crate) const FILE_STRFTIME: &str = "%Y-%m-%d-%H-%M-%S.%3f%z";

use crate::common::*;

pub async fn start(config: config::Config) -> Result<()> {
    // data logging
    let start_time = Local::now();
    let log_dir = config
        .logging
        .log_dir
        .join(format!("{}", start_time.format(FILE_STRFTIME)));
    let checkpoint_dir = log_dir.join("checkpoints");

    tokio::fs::create_dir_all(&checkpoint_dir).await?;

    // initialize model
    // let mut generator_vs = nn::VarStore::new(device);
    // let mut discriminator_vs = nn::VarStore::new(device);

    // let mut generator = GeneratorInit {
    //     ndims: 2,
    //     input_channels: image_dim + latent_dim + 1,
    //     output_channels: 3,
    //     num_heads: 4,
    //     strides: [1, 2, 2, 2],
    //     block_channels: [16, 16, 16, 16],
    //     context_channels: [16, 16, 16, 16],
    //     repeats: [3, 3, 3, 3],
    // }
    // .build(&generator_vs.root() / "generator")?;

    // let mut discriminator = DiscriminatorInit {
    //     ndims: 3,
    //     ksize: 3,
    //     input_channels: 3,
    //     channels: [4, 8, 16],
    //     strides: [2, 2, 2],
    // }
    // .build(&discriminator_vs.root() / "discriminator")?;

    // let mut generator_opt = nn::adam(0.5, 0.999, 0.).build(&generator_vs, learning_rate)?;
    // let mut discriminator_opt = nn::adam(0.5, 0.999, 0.).build(&discriminator_vs, learning_rate)?;

    let config = ArcRef::new(Arc::new(config));
    let (train_tx, train_rx) = tokio::sync::mpsc::channel(2);
    let (log_tx, log_rx) = tokio::sync::mpsc::channel(1);

    // data stream to channel worker
    let data_fut = {
        let config = config.clone();

        tokio::task::spawn(async move {
            // load dataset
            let mut stream =
                training_stream::training_stream(&config.dataset, &config.train).await?;

            while let Some(msg) = stream.next().await.transpose()? {
                let result = train_tx.send(msg).await;
                if result.is_err() {
                    break;
                }
            }

            Fallible::Ok(())
        })
        .map(|result| Fallible::Ok(result??))
    };

    // training worker
    let train_fut = {
        let config = config.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            train::training_worker(config, train_rx, log_tx)
        })
        .map(|result| Fallible::Ok(result??))
    };

    let log_fut = {
        let log_dir = log_dir.clone();

        tokio::task::spawn(logging::logging_worker(log_dir, log_rx))
            .map(|result| Fallible::Ok(result??))
    };

    // run all tasks
    futures::try_join!(data_fut, train_fut, log_fut)?;

    Ok(())
}
