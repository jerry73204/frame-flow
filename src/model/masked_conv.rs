use super::conv::{ConvND, ConvNDInit, ConvNDInitDyn, ConvParam};
use crate::common::*;

#[derive(Debug, Clone)]
pub struct MaskedConvNDInit<Param: ConvParam> {
    pub conv: ConvNDInit<Param>,
}

impl<Param: ConvParam> MaskedConvNDInit<Param> {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<MaskedConvND> {
        let path = path.borrow();
        let Self { conv: conv_init } = self;
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

        Ok(MaskedConvND {
            conv,
            mask_conv,
            mask_divisor: mask_divisor as f64,
        })
    }
}

#[derive(Debug)]
pub struct MaskedConvND {
    conv: ConvND,
    mask_conv: ConvND,
    mask_divisor: f64,
}

impl MaskedConvND {
    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let Self {
            ref conv,
            ref mask_conv,
            mask_divisor,
        } = *self;
        let output = conv.forward(input);
        let output_mask = mask.map(|mask| mask_conv.forward(mask) / mask_divisor);
        (output, output_mask)
    }
}
