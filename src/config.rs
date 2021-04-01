use crate::common::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub dataset: Dataset,
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = json5::from_str(&fs::read_to_string(path)?)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub dir: PathBuf,
    pub height: NonZeroUsize,
    pub width: NonZeroUsize,
}
