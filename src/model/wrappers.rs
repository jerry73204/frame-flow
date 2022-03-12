use crate::{
    common::*,
    model::{DetectionEmbedding, Discriminator, Generator},
};
use tch_goodies::{lr_schedule::LrScheduler, DenseDetectionTensorList, Ratio};
use yolo_dl::{loss::YoloLoss, model::YoloModel};

#[derive(Debug)]
pub struct DetectorWrapper {
    pub model: YoloModel,
}

impl DetectorWrapper {
    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<DenseDetectionTensorList> {
        let input = input * 0.5 + 0.5;
        self.model.forward_t(&input, train)
    }
}

#[derive(Debug)]
pub struct GeneratorWrapper {
    pub latent_dim: i64,
    pub embedding_model: DetectionEmbedding,
    pub generator_model: Generator,
}

impl GeneratorWrapper {
    pub fn forward_t(
        &self,
        input: &DenseDetectionTensorList,
        noise: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let Self {
            latent_dim,
            ref embedding_model,
            ref generator_model,
        } = *self;
        let device = input.device();

        let embedding = embedding_model.forward_t(input, train)?;
        let noise = {
            let (b, _c, h, w) = embedding.size4()?;
            let noise = match noise {
                Some(noise) => {
                    let noise = noise.borrow();
                    ensure!(noise.size1()? == latent_dim);
                    noise.view([1, latent_dim, 1, 1]).to_device(device)
                }
                None => Tensor::randn(&[b, latent_dim, 1, 1], (Kind::Float, device)),
            };
            noise.expand(&[b, latent_dim, h, w], false)
        };
        let input = Tensor::cat(&[embedding, noise], 1);
        let output = generator_model.forward_t(&input, train);
        Ok(output)
    }
}

#[derive(Debug)]
pub struct ImageSequenceDiscriminatorWrapper {
    pub model: Discriminator,
    pub input_len: usize,
}

impl ImageSequenceDiscriminatorWrapper {
    pub fn forward_t(&self, input: &[impl Borrow<Tensor>], train: bool) -> Result<Tensor> {
        let Self {
            input_len,
            ref model,
        } = *self;
        ensure!(input_len == input.len());

        let input = Tensor::cat(input, 1);
        let output = model.forward_t(&input, train);
        Ok(output)
    }
}
