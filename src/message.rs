use crate::common::*;

#[derive(Debug)]
pub enum LogMessage {
    Loss {
        step: usize,
        learning_rate: f64,
        sequence: Vec<LossLog>,
        transformer: TransformerLossLog,
    },
    Image {
        step: usize,
        sequence: Vec<ImageLog>,
    },
}

#[derive(Debug)]
pub struct LossLog {
    pub real_det_loss: Option<f64>,
    pub fake_det_loss: Option<f64>,
    pub discriminator_loss: Option<f64>,
    pub generator_loss: Option<f64>,
    pub detector_grads: Option<Vec<(String, f64)>>,
    pub generator_grads: Option<Vec<(String, f64)>>,
    pub discriminator_grads: Option<Vec<(String, f64)>>,
    pub detector_weights: Option<Vec<(String, f64)>>,
    pub generator_weights: Option<Vec<(String, f64)>>,
    pub discriminator_weights: Option<Vec<(String, f64)>>,
}

#[derive(Debug)]
pub struct TransformerLossLog {
    pub transformer_loss: Option<f64>,
    pub transformer_discriminator_loss: Option<f64>,
    pub transformer_weights: Option<Vec<(String, f64)>>,
    pub transformer_grads: Option<Vec<(String, f64)>>,
    pub transformer_discriminator_weights: Option<Vec<(String, f64)>>,
    pub transformer_discriminator_grads: Option<Vec<(String, f64)>>,
}

#[derive(Debug)]
pub struct ImageLog {
    pub true_image: Tensor,
    pub fake_image: Option<Tensor>,
}

#[derive(Debug, TensorLike)]
pub struct TrainingMessage {
    pub batch_index: usize,
    /// Sequence of batched images.
    pub image_batch_seq: Vec<Tensor>,
    /// Sequence of batched noise, each has shape `[batch_size, latent_dim]`.
    pub noise_seq: Vec<Tensor>,
    /// Sequence of batched sets of boxes.
    #[tensor_like(clone)]
    pub boxes_batch_seq: Vec<Vec<Vec<RatioRectLabel<R64>>>>,
}
