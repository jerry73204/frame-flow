use crate::common::*;

#[derive(Debug)]
pub struct Adversarial {}

impl Adversarial {
    pub fn new<'a>(path: impl Borrow<nn::Path<'a>>) -> Self {
        todo!();
    }

    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<Tensor> {
        todo!();
    }
}
