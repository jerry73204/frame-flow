use crate::common::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub dataset: Dataset,
    pub train: Training,
    pub model: Model,
    pub logging: Logging,
    pub loss: Loss,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = json5::from_str(&fs::read_to_string(path)?)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Dataset {
    Iii(IiiDataset),
    Simple(SimpleDataset),
    Mnist(MnistDataset),
}

impl Dataset {
    pub fn image_dim(&self) -> usize {
        match self {
            Self::Iii(dataset) => dataset.image_dim(),
            Self::Simple(dataset) => dataset.image_dim(),
            Dataset::Mnist(dataset) => dataset.image_dim(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IiiDataset {
    pub dataset_dir: PathBuf,
    pub classes_file: PathBuf,
    pub class_whitelist: Option<HashSet<String>>,
    pub blacklist_files: Option<HashSet<PathBuf>>,
    pub min_seq_len: Option<usize>,
}

impl IiiDataset {
    pub fn image_dim(&self) -> usize {
        3
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleDataset {
    pub dataset_dir: PathBuf,
    pub file_name_digits: NonZeroUsize,
}

impl SimpleDataset {
    pub fn image_dim(&self) -> usize {
        3
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnistDataset {
    pub dataset_dir: PathBuf,
}

impl MnistDataset {
    pub fn image_dim(&self) -> usize {
        3
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Training {
    pub cache_dir: PathBuf,
    pub batch_size: NonZeroUsize,
    pub image_size: NonZeroUsize,
    pub latent_dim: NonZeroUsize,
    pub peek_len: usize,
    pub pred_len: NonZeroUsize,
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
    pub lr_schedule: train::config::LearningRateSchedule,
    pub warm_up_steps: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logging {
    pub log_dir: PathBuf,
    pub save_image_steps: Option<NonZeroUsize>,
    pub save_checkpoint_steps: Option<NonZeroUsize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub detector: DetectionModel,
    pub generator: GeneratorModel,
    pub discriminator: DiscriminatorModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionModel {
    pub model_file: PathBuf,
    pub weights_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorModel {
    pub kind: GeneratorModelKind,
    pub weights_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeneratorModelKind {
    #[serde(rename = "n_layers")]
    NLayers,
    #[serde(rename = "unet")]
    UNet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminatorModel {
    pub weights_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Loss {
    pub detector: train::config::Loss,
    pub image_recon: GanLoss,
    // pub det_recon: GanLoss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GanLoss {
    #[serde(rename = "dc_gan")]
    DcGan,
    #[serde(rename = "relativistic_gan")]
    RelativisticGan,
    #[serde(rename = "wgan")]
    WGan,
    #[serde(rename = "wgan_gp")]
    WGanGp,
}
