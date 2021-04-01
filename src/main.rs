use frame_flow::{common::*, config, dataset::DatasetInit};

#[derive(Debug, Clone, StructOpt)]
/// Implementation for 'frame-flow' model.
pub struct Args {
    #[structopt(long, default_value = "config.json5")]
    pub config: PathBuf,
}

fn main() -> Result<()> {
    let Args { config } = Args::from_args();
    let config::Config {
        device,
        dataset: config::Dataset { dir, height, width },
    } = config::Config::load(&config)?;

    // load dataset
    let dataset = DatasetInit {
        dir: &dir,
        file_name_digits: 8,
        height: height.get(),
        width: width.get(),
    }
    .load()?;

    let vs = nn::VarStore::new(device);
    let root = vs.root();
    Ok(())
}
