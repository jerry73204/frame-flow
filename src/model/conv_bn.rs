use super::{
    batch_norm::{BatchNormND, BatchNormNDConfig},
    conv::{Conv1DInit, Conv2DInit, Conv3DInit, ConvND, ConvNDInit, ConvNDInitDyn, ConvParam},
};
use crate::common::*;

#[derive(Debug, Clone)]
pub struct ConvBnNDInit<Param: ConvParam> {
    pub conv: ConvNDInit<Param>,
    pub bn: BatchNormNDConfig,
    pub bn_first: bool,
}

pub type ConvBn1DInit = ConvBnNDInit<usize>;
pub type ConvBn2DInit = ConvBnNDInit<[usize; 2]>;
pub type ConvBn3DInit = ConvBnNDInit<[usize; 3]>;
pub type ConvBn4DInit = ConvBnNDInit<[usize; 4]>;
pub type ConvBnNDInitDyn = ConvBnNDInit<Vec<usize>>;

impl ConvBn1DInit {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: Conv1DInit::new(ksize),
            bn: Default::default(),
            bn_first: true,
        }
    }
}

impl<const DIM: usize> ConvBnNDInit<[usize; DIM]> {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: ConvNDInit::<[usize; DIM]>::new(ksize),
            bn: Default::default(),
            bn_first: true,
        }
    }
}

impl ConvBnNDInitDyn {
    pub fn new(ndim: usize, ksize: usize) -> Self {
        Self {
            conv: ConvNDInitDyn::new(ndim, ksize),
            bn: Default::default(),
            bn_first: true,
        }
    }
}

impl<Param: ConvParam> ConvBnNDInit<Param> {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<ConvBnND> {
        let path = path.borrow();
        let Self { conv, bn, bn_first } = self;
        let nd = conv.dim()?;

        Ok(ConvBnND {
            conv: conv.build(path / "conv", in_dim, out_dim)?,
            bn: BatchNormND::new(
                path / "bn",
                nd,
                if bn_first { in_dim } else { out_dim } as i64,
                bn,
            ),
            bn_first,
        })
    }
}

#[derive(Debug)]
pub struct ConvBnND {
    conv: ConvND,
    bn: BatchNormND,
    bn_first: bool,
}

impl ConvBnND {
    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            ref conv,
            ref mut bn,
            bn_first,
        } = *self;

        let output = if bn_first {
            conv.forward(&bn.forward_t(input, train)?)
        } else {
            bn.forward_t(&conv.forward(input), train)?
        };

        Ok(output)
    }
}
