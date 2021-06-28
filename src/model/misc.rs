use crate::common::*;
use tch_modules::{DarkBatchNorm, DarkBatchNormInit, InstanceNorm, InstanceNormInit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingKind {
    Reflect,
    Replicate,
    Zero,
}

impl PaddingKind {
    pub fn build(self, lrtb: [usize; 4]) -> Pad2D {
        let [l, r, t, b] = lrtb;
        Pad2D {
            kind: self,
            lrtb: [l as i64, r as i64, t as i64, b as i64],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormKind {
    BatchNorm,
    InstanceNorm,
    None,
}

impl NormKind {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>, out_dim: i64) -> Norm {
        match self {
            Self::BatchNorm => {
                let norm = DarkBatchNormInit::default().build(path, out_dim);
                Norm::BatchNorm(norm)
            }
            Self::InstanceNorm => {
                let norm = InstanceNormInit::default().build(path, out_dim);
                Norm::InstanceNorm(norm)
            }
            Self::None => Norm::None,
        }
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

#[derive(Debug)]
pub struct Pad2D {
    kind: PaddingKind,
    lrtb: [i64; 4],
}

impl nn::Module for Pad2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self.kind {
            PaddingKind::Reflect => xs.reflection_pad2d(&self.lrtb),
            PaddingKind::Replicate => xs.replication_pad1d(&self.lrtb),
            PaddingKind::Zero => {
                let [l, r, t, b] = self.lrtb;
                xs.zero_pad2d(l, r, t, b)
            }
        }
    }
}
