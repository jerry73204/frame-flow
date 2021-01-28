mod common;
mod config;
mod model;
mod video;

use crate::{common::*, config::Config, model::Model};

#[derive(Debug, Clone, StructOpt)]
/// Implementation for 'frame-flow' model.
pub struct Args {
    #[structopt(long, default_value = "config.json5")]
    pub config: PathBuf,
}

fn main() -> Result<()> {
    let Args { config } = Args::from_args();
    let Config { input_url, device } = Config::load(&config)?;

    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let model = Model::new(&root);

    let seq_size = 4;
    let mut video = VideoCapture::from_file(&input_url, 0)?;
    let frame_count = video.get(videoio::CAP_PROP_FRAME_COUNT)? as usize;

    let mut rng = rand::thread_rng();

    loop {
        let frame_index = rng.gen_range(0..(frame_count - 4));

        video.set(videoio::CAP_PROP_POS_FRAMES, frame_index as f64)?;

        let frames: Vec<_> = (0..seq_size)
            .map(|_| {
                let timestamp = video.get(videoio::CAP_PROP_POS_MSEC)?;
                let mut mat = Mat::default()?;
                let success = video.read(&mut mat)?;
                ensure!(success, "no frame at index {}", frame_index);
                Ok(mat)
            })
            .try_collect()?;
    }
}
