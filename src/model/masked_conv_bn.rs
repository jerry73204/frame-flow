use super::{
    batch_norm::{BatchNormND, BatchNormNDConfig},
    conv::{ConvND, ConvNDInit, ConvNDInitDyn, ConvParam},
};
use crate::common::*;

#[derive(Debug, Clone)]
pub struct MaskedConvBnNDInit<Param: ConvParam> {
    pub conv: ConvNDInit<Param>,
    pub bn: BatchNormNDConfig,
    pub bn_first: bool,
}

impl<Param: ConvParam> MaskedConvBnNDInit<Param> {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<MaskedConvBnND> {
        let path = path.borrow();
        let Self {
            conv: conv_init,
            bn: bn_init,
            bn_first,
        } = self;
        let nd = conv_init.dim()?;
        let bn = BatchNormND::new(
            path / "bn",
            nd,
            if bn_first { in_dim } else { out_dim } as i64,
            bn_init,
        );

        let conv = conv_init.clone().build(path / "conv", in_dim, out_dim)?;

        let mask_divisor = conv_init.ksize.usize_iter().product::<usize>();
        let mask_conv = {
            let conv = ConvNDInitDyn {
                groups: 1,
                bias: false,
                ws_init: nn::Init::Const(1.0),
                bs_init: nn::Init::Const(0.0),
                ..conv_init.into_dyn()
            }
            .build(path / "mask_conv", 1, 1)?;
            conv.set_trainable(false);
            conv
        };

        Ok(MaskedConvBnND {
            conv,
            mask_conv,
            mask_divisor: mask_divisor as f64,
            bn,
            bn_first,
        })
    }
}

#[derive(Debug)]
pub struct MaskedConvBnND {
    conv: ConvND,
    mask_conv: ConvND,
    mask_divisor: f64,
    bn: BatchNormND,
    bn_first: bool,
}

impl MaskedConvBnND {
    pub fn forward(
        &mut self,
        input: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let Self {
            ref conv,
            ref mask_conv,
            ref mut bn,
            mask_divisor,
            bn_first,
        } = *self;

        let output = if bn_first {
            conv.forward(&bn.forward_t(input, train)?)
        } else {
            bn.forward_t(&conv.forward(input), train)?
        };

        let output_mask = mask.map(|mask| mask_conv.forward(mask) / mask_divisor);
        Ok((output, output_mask))
    }
}
