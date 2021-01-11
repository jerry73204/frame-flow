mod common;
mod config;
mod video;

use crate::{common::*, config::Config};

#[derive(Debug, Clone, StructOpt)]
/// Implementation for 'frame-flow' model.
pub struct Args {
    #[structopt(long, default_value = "config.json5")]
    pub config: PathBuf,
}

fn main() -> Result<()> {
    let Args { config } = Args::from_args();
    let Config { input_url } = Config::load(&config)?;

    video::play(&input_url)?;

    Ok(())
}
