use frame_flow::{
    common::*,
    config,
    model::{DiscriminatorInit, GeneratorInit},
    training_stream::{TrainingRecord, TrainingStreamInit},
};

#[derive(Debug, Clone, StructOpt)]
/// Implementation for 'frame-flow' model.
pub struct Args {
    #[structopt(long, default_value = "config.json5")]
    pub config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let Args { config } = Args::from_args();
    let config::Config {
        dataset: config::Dataset { dir, height, width },
        train:
            config::Training {
                device,
                peek_len,
                pred_len,
            },
    } = config::Config::load(&config)?;

    let seq_len = peek_len + pred_len;

    // load dataset
    let train_stream = TrainingStreamInit {
        dir: &dir,
        file_name_digits: 8,
        height: height.get(),
        width: width.get(),
        latent_dim: 64,
        batch_size: 16,
        device,
        seq_len,
    }
    .build()
    .await?;

    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let generator = GeneratorInit {
        ndims: 3,
        input_channels: 3,
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

    train_stream
        .stream()
        .try_for_each(|record| async move {
            let TrainingRecord {
                sequence,
                noise,
                batch_index,
                record_index: _,
            } = record;

            info!("batch_index = {}", batch_index);

            Ok(())
        })
        .await?;

    Ok(())
}
