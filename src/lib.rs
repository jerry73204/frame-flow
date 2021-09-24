pub mod common;
pub mod config;
pub mod dataset;
pub mod logging;
pub mod message;
pub mod model;
// pub mod sampler;
pub mod train;
pub mod training_stream;
pub mod utils;

pub(crate) const FILE_STRFTIME: &str = "%Y-%m-%d-%H-%M-%S.%3f%z";

use crate::common::*;

pub async fn start(config: config::Config) -> Result<()> {
    // prepare logging directories
    let start_time = Local::now();
    let log_dir = config
        .logging
        .log_dir
        .join(format!("{}", start_time.format(FILE_STRFTIME)));
    let checkpoint_dir = log_dir.join("checkpoints");

    tokio::fs::create_dir_all(&checkpoint_dir).await?;

    // save config
    {
        let config_file = log_dir.join("config.json5");
        let text = serde_json::to_string_pretty(&config)?;
        tokio::fs::write(config_file, text).await?;
    }

    let config = ArcRef::new(Arc::new(config));
    let (train_tx, train_rx) = mpsc::channel(16);
    let (log_tx, log_rx) = mpsc::channel(2);

    // load dataset
    let dataset: dataset::Dataset = match config.dataset {
        config::Dataset::Iii(config::IiiDataset {
            ref dataset_dir,
            ref classes_file,
            ref class_whitelist,
            ref blacklist_files,
            min_seq_len,
            ..
        }) => dataset::IiiDataset::load(
            dataset_dir,
            classes_file,
            class_whitelist.clone(),
            min_seq_len,
            blacklist_files.clone().unwrap_or_else(HashSet::new),
        )
        .await?
        .into(),
        config::Dataset::Simple(config::SimpleDataset {
            ref dataset_dir,
            file_name_digits,
            ..
        }) => dataset::SimpleDataset::load(dataset_dir, file_name_digits.get())
            .await?
            .into(),
        config::Dataset::Mnist(config::MnistDataset { ref dataset_dir }) => {
            dataset::MnistDataset::new(dataset_dir)?.into()
        }
    };
    let num_classes = dataset.classes().len();

    // data stream to channel worker
    let data_fut = {
        let config = config.clone();

        tokio::task::spawn(async move {
            let mut stream = training_stream::training_stream(dataset, &config.train).await?;

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
            train::training_worker(config, num_classes, checkpoint_dir, train_rx, log_tx)
        })
        .map(|result| Fallible::Ok(result??))
    };

    let log_fut = {
        let log_dir = log_dir.clone();

        tokio::task::spawn(logging::logging_worker(
            log_dir,
            log_rx,
            config.logging.save_motion_field_image,
        ))
        .map(|result| Fallible::Ok(result??))
    };

    // run all tasks
    futures::try_join!(data_fut, train_fut, log_fut)?;

    Ok(())
}
