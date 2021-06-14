use crate::common::*;
use tch_modules::{DarkBatchNorm, DarkBatchNormInit, InstanceNorm, InstanceNormInit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingKind {
    Reflect,
    Replicate,
    Zero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormKind {
    BatchNorm,
    InstanceNorm,
    None,
}

impl NormKind {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>, out_dim: i64) -> Norm {
        let norm = match self {
            Self::BatchNorm => {
                let norm = DarkBatchNormInit::default().build(path, out_dim);
                Norm::BatchNorm(norm)
            }
            Self::InstanceNorm => {
                let norm = InstanceNormInit::default().build(path, out_dim);
                Norm::InstanceNorm(norm)
            }
            Self::None => Norm::None,
        };

        norm
    }
}

#[derive(Debug)]
pub enum Norm {
    BatchNorm(DarkBatchNorm),
    InstanceNorm(InstanceNorm),
    None,
}

impl nn::ModuleT for Norm {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        match self {
            Self::BatchNorm(norm) => norm.forward_t(input, train),
            Self::InstanceNorm(norm) => norm.forward_t(input, train),
            Self::None => input.shallow_clone(),
        }
    }
}
