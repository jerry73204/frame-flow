use crate::common::*;

#[derive(Debug)]
pub enum LogMessage {
    Loss {
        step: usize,
        learning_rate: f64,
        sequence: Vec<LossLog>,
    },
    Image {
        step: usize,
        sequence: Vec<ImageLog>,
    },
}

#[derive(Debug)]
pub struct LossLog {
    pub det_loss: f64,
    pub det_recon_loss: f64,
    pub discriminator_loss: f64,
    pub generator_loss: f64,
    pub gen_grads: Vec<(String, f64)>,
    pub disc_grads: Vec<(String, f64)>,
}

#[derive(Debug)]
pub struct ImageLog {
    pub true_image: Tensor,
    pub fake_image: Tensor,
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
