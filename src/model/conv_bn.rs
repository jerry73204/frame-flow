use super::{
    batch_norm::{BatchNormND, BatchNormNDConfig},
    conv::{
        ConvND, ConvNDInit, ConvNDInit1D, ConvNDInit2D, ConvNDInit3D, ConvNDInitDyn, ConvParam,
    },
};
use crate::common::*;

#[derive(Debug, Clone)]
pub struct ConvBnNDInit<Param: ConvParam> {
    pub conv: ConvNDInit<Param>,
    pub bn: BatchNormNDConfig,
    pub bn_first: bool,
}

pub type ConvBnNDInit1D = ConvBnNDInit<usize>;
pub type ConvBnNDInit2D = ConvBnNDInit<[usize; 2]>;
pub type ConvBnNDInit3D = ConvBnNDInit<[usize; 3]>;
pub type ConvBnNDInit4D = ConvBnNDInit<[usize; 4]>;
pub type ConvBnNDInitDyn = ConvBnNDInit<Vec<usize>>;

impl ConvBnNDInit1D {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: ConvNDInit1D::new(ksize),
            bn: Default::default(),
            bn_first: true,
        }
    }
}

impl ConvBnNDInit2D {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: ConvNDInit2D::new(ksize),
            bn: Default::default(),
            bn_first: true,
        }
    }
}

impl ConvBnNDInit3D {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: ConvNDInit3D::new(ksize),
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
    pub conv: ConvND,
    pub bn: BatchNormND,
    pub bn_first: bool,
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
