use super::{
    attention::{Attention, AttentionInit},
    conv::{ConvND, ConvNDInitDyn},
    conv_bn::{ConvBnND, ConvBnNDInitDyn},
    tensor_list::TensorList,
};
use crate::common::*;

#[derive(Debug, Clone)]
pub struct DecoderBlockInit {
    pub ndims: usize,
    pub input_channels: usize,
    pub context_channels: usize,
    pub output_channels: usize,
    pub repeat: usize,
    pub num_heads: usize,
    pub attention_channels: usize,
    pub keyvalue_channels: usize,
    pub attention_ksize: usize,
    pub conv_transposed: bool,
}

impl DecoderBlockInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<DecoderBlock> {
        let path = path.borrow();
        let Self {
            ndims,
            input_channels,
            context_channels,
            attention_channels,
            keyvalue_channels,
            output_channels,
            repeat,
            num_heads,
            attention_ksize,
            conv_transposed,
        } = self;

        let shortcut_conv = ConvNDInitDyn {
            transposed: conv_transposed,
            ..ConvNDInitDyn::new(ndims, 1)
        }
        .build(path / "shortcut_conv", input_channels, attention_channels)?;

        let pre_attention_conv = ConvBnNDInitDyn {
            conv: ConvNDInitDyn {
                transposed: conv_transposed,
                ..ConvNDInitDyn::new(ndims, 1)
            },
            ..ConvBnNDInitDyn::new(ndims, 1)
        }
        .build(
            path / "pre_attention_conv",
            input_channels,
            attention_channels,
        )?;
        let post_attention_conv = ConvBnNDInitDyn {
            conv: ConvNDInitDyn {
                transposed: conv_transposed,
                ..ConvNDInitDyn::new(ndims, 1)
            },
            ..ConvBnNDInitDyn::new(ndims, 1)
        }
        .build(
            path / "post_attention_conv",
            attention_channels,
            attention_channels,
        )?;
        let attentions: Vec<_> = (0..repeat)
            .map(|index| {
                AttentionInit {
                    num_heads,
                    input_channels: attention_channels,
                    context_channels,
                    output_channels: attention_channels,
                    key_channels: keyvalue_channels,
                    value_channels: keyvalue_channels,
                    input_conv: ConvNDInitDyn {
                        transposed: conv_transposed,
                        ..ConvNDInitDyn::new(ndims, attention_ksize)
                    },
                    context_conv: ConvNDInitDyn {
                        transposed: conv_transposed,
                        ..ConvNDInitDyn::new(ndims, attention_ksize)
                    },
                }
                .build(path / format!("attention_{}", index))
            })
            .try_collect()?;

        let merge_conv = ConvNDInitDyn {
            transposed: conv_transposed,
            ..ConvNDInitDyn::new(ndims, 1)
        }
        .build(path / "merge_conv", attention_channels * 2, output_channels)?;

        Ok(DecoderBlock {
            attentions,
            shortcut_conv,
            merge_conv,
            pre_attention_conv,
            post_attention_conv,
        })
    }
}

#[derive(Debug)]
pub struct DecoderBlock {
    attentions: Vec<Attention>,
    shortcut_conv: ConvND,
    merge_conv: ConvND,
    pre_attention_conv: ConvBnND,
    post_attention_conv: ConvBnND,
}

impl DecoderBlock {
    pub fn forward_t(
        &mut self,
        input: impl Borrow<Tensor>,
        contexts: impl TensorList,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<DecoderBlockOutput> {
        let Self {
            ref attentions,
            ref shortcut_conv,
            ref merge_conv,
            ref mut pre_attention_conv,
            ref mut post_attention_conv,
        } = *self;
        let contexts = contexts.into_owned_tensors();
        ensure!(
            contexts.len() == attentions.len(),
            "the number of context tensors does not match the number of attentions"
        );
        let input = input.borrow();
        let mask = mask.map(|mask| mask.detach());

        let shortcut = shortcut_conv.forward(input);
        let (branch, output_mask) = {
            dbg!(input.size());
            let xs = pre_attention_conv.forward_t(input, train)?;
            dbg!(xs.size());

            let (xs, mask) = izip!(attentions.iter(), contexts).try_fold(
                (xs, mask),
                |(xs, mask), (attention, context)| -> Result<_> {
                    let (xs, mask) = attention.forward(&xs, context.borrow(), mask.as_ref())?;
                    Ok((xs, mask))
                },
            )?;

            let xs = post_attention_conv.forward_t(&xs, train)?;

            (xs, mask)
        };
        let merge = Tensor::cat(&[shortcut, branch], 1);
        let output = merge_conv.forward(&merge);

        Ok(DecoderBlockOutput {
            feature: output,
            mask: output_mask,
        })
    }
}

#[derive(Debug)]
pub struct DecoderBlockOutput {
    pub feature: Tensor,
    pub mask: Option<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_block_test() -> Result<()> {
        let b = 6;
        let cx = 3;
        let cxx = 4;
        let hx = 12;
        let wx = 16;
        let cy = 5;
        let repeat = 2;

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();

        let mut block = DecoderBlockInit {
            input_channels: cx,
            context_channels: cxx,
            output_channels: cy,
            repeat,
            num_heads: 10,
            attention_channels: cx * 2,
            keyvalue_channels: cx * 2,
            ndims: 2,
            attention_ksize: 3,
            conv_transposed: false,
        }
        .build(&root)?;

        let contexts: Vec<_> = (0..repeat)
            .map(|_| Tensor::rand(&[b, cxx as i64, hx, wx], FLOAT_CPU))
            .collect();

        let input = Tensor::rand(&[b, cx as i64, hx, wx], FLOAT_CPU);
        let DecoderBlockOutput { feature, mask: _ } =
            block.forward_t(&input, contexts, None, true)?;

        ensure!(
            feature.size() == vec![b, cy as i64, hx as i64, wx as i64],
            "incorrect output shape"
        );

        Ok(())
    }
}
