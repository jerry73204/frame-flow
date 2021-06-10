use crate::common::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub dataset: Dataset,
    pub train: Training,
    pub logging: Logging,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = json5::from_str(&fs::read_to_string(path)?)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub dataset_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub image_size: NonZeroUsize,
    pub image_dim: NonZeroUsize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Training {
    pub batch_size: NonZeroUsize,
    pub peek_len: usize,
    pub pred_len: usize,
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
    pub learning_rate: R64,
    pub warm_up_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logging {
    pub log_dir: PathBuf,
    pub save_image_steps: Option<NonZeroUsize>,
    pub save_checkpoint_steps: Option<NonZeroUsize>,
}
