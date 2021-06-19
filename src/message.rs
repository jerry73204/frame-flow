use crate::common::*;

#[derive(Debug)]
pub struct LogMessage {
    pub step: usize,
    pub dis_loss: f64,
    pub gen_loss: f64,
    pub learning_rate: f64,
    pub true_image: Option<Vec<Tensor>>,
    pub fake_image: Option<Vec<Tensor>>,
}

#[derive(Debug, TensorLike)]
pub struct TrainingMessage {
    pub sequence: Vec<Tensor>,
    pub noise: Tensor,
    pub batch_index: usize,
    pub record_index: usize,
}
