use crate::{common::*, model::NormKind};

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
    #[serde(default = "default_warm_up_steps")]
    pub warm_up_steps: usize,
    #[serde(default = "default_label_flip_prob")]
    pub label_flip_prob: R64,
    pub train_detector_steps: usize,
    pub train_discriminator_steps: usize,
    pub train_generator_steps: usize,
    pub train_consistency_steps: usize,
    #[serde(default = "default_critic_noise_prob")]
    pub critic_noise_prob: R64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logging {
    pub log_dir: PathBuf,
    pub save_image_steps: Option<NonZeroUsize>,
    pub save_checkpoint_steps: Option<NonZeroUsize>,
    pub save_detector_checkpoint: bool,
    pub save_generator_checkpoint: bool,
    pub save_discriminator_checkpoint: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub detection_embedding: DetectionEmbedding,
    pub detector: DetectionModel,
    pub generator: GeneratorModel,
    pub discriminator: DiscriminatorModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEmbedding {
    pub channels: Vec<usize>,
    pub num_blocks: Vec<usize>,
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
    pub norm: NormKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeneratorModelKind {
    Resnet,
    UNet,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminatorModel {
    pub weights_file: Option<PathBuf>,
    pub norm: NormKind,
    pub num_blocks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Loss {
    pub detector: train::config::Loss,
    pub image_recon: GanLoss,
    // pub det_recon: GanLoss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GanLoss {
    DcGan,
    RaSGan,
    RaLsGan,
    WGan,
    WGanGp,
}

fn default_label_flip_prob() -> R64 {
    r64(0.0)
}

fn default_critic_noise_prob() -> R64 {
    r64(0.0)
}

fn default_warm_up_steps() -> usize {
    0
}
