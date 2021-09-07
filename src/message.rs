use crate::{common::*, model::DetectionSimilarity};

#[derive(Debug)]
pub enum LogMessage {
    Loss(Loss),
    Image {
        step: usize,
        sequence: Vec<ImageLog>,
    },
}

#[derive(Debug)]
pub struct Loss {
    pub step: usize,
    pub learning_rate: f64,

    pub detector_loss: Option<f64>,
    pub discriminator_loss: Option<f64>,
    pub generator_loss: Option<f64>,
    pub retraction_identity_loss: Option<f64>,
    pub retraction_identity_similarity: Option<DetectionSimilarity>,
    pub triangular_identity_loss: Option<f64>,
    pub triangular_identity_similarity: Option<DetectionSimilarity>,
    pub forward_consistency_loss: Option<f64>,
    pub forward_consistency_similarity: Option<DetectionSimilarity>,
    pub backward_consistency_gen_loss: Option<f64>,
    pub backward_consistency_disc_loss: Option<f64>,

    pub detector_weights: Option<WeightsAndGrads>,
    pub generator_weights: Option<WeightsAndGrads>,
    pub discriminator_weights: Option<WeightsAndGrads>,
    pub transformer_weights: Option<WeightsAndGrads>,
    pub image_seq_discriminator_weights: Option<WeightsAndGrads>,

    pub ground_truth_image_seq: Option<Vec<Tensor>>,
    pub generator_generated_image_seq: Option<Vec<Tensor>>,

    pub transformer_generated_image_seq: Option<Vec<Tensor>>,
    pub transformer_generated_det_seq: Option<Vec<DenseDetectionTensorList>>,
    pub transformer_attention_image_seq: Option<Vec<Tensor>>,
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

#[derive(Debug)]
pub struct WeightsAndGrads {
    pub weights: Vec<(String, f64)>,
    pub grads: Vec<(String, f64)>,
}
