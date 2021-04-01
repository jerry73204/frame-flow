use super::attention::{Attention, AttentionInit};
use crate::common::*;
use tch_goodies::module::{ConvBnND, ConvBnNDInitDyn, ConvND, ConvNDInitDyn};

#[derive(Debug, Clone)]
pub struct EncoderBlockInit {
    pub ndims: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub repeat: usize,
    pub num_heads: usize,
    pub attention_channels: usize,
    pub keyvalue_channels: usize,
    pub attention_ksize: usize,
    pub conv_transposed: bool,
}

impl EncoderBlockInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<EncoderBlock> {
        let path = path.borrow();
        let Self {
            ndims,
            input_channels,
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
                    context_channels: attention_channels,
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

        Ok(EncoderBlock {
            attentions,
            shortcut_conv,
            merge_conv,
            pre_attention_conv,
            post_attention_conv,
        })
    }
}

#[derive(Debug)]
pub struct EncoderBlock {
    attentions: Vec<Attention>,
    shortcut_conv: ConvND,
    merge_conv: ConvND,
    pre_attention_conv: ConvBnND,
    post_attention_conv: ConvBnND,
}

impl EncoderBlock {
    pub fn forward_t(
        &mut self,
        input: impl Borrow<Tensor>,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<EncoderBlockOutput> {
        let Self {
            ref attentions,
            ref shortcut_conv,
            ref merge_conv,
            ref mut pre_attention_conv,
            ref mut post_attention_conv,
        } = *self;
        let input = input.borrow();
        let mask = mask.map(|mask| mask.detach());

        let shortcut = shortcut_conv.forward(input);
        let (branch, output_mask, contexts) = {
            let xs = pre_attention_conv.forward_t(input, train)?;
            let mut contexts = vec![];

            let (xs, mask) =
                attentions
                    .iter()
                    .try_fold((xs, mask), |(xs, mask), attention| -> Result<_> {
                        let (xs, mask) = attention.forward(&xs, &xs, mask.as_ref())?;
                        contexts.push(xs.shallow_clone());
                        Ok((xs, mask))
                    })?;

            let xs = post_attention_conv.forward_t(&xs, train)?;

            (xs, mask, contexts)
        };
        let merge = Tensor::cat(&[shortcut, branch], 1);
        let output = merge_conv.forward(&merge);

        Ok(EncoderBlockOutput {
            feature: output,
            mask: output_mask,
            contexts,
        })
    }
}

#[derive(Debug)]
pub struct EncoderBlockOutput {
    pub feature: Tensor,
    pub mask: Option<Tensor>,
    pub contexts: Vec<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_block_test() -> Result<()> {
        let b = 6;
        let cx = 3;
        let hx = 12;
        let wx = 16;
        let cy = 5;

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();

        let mut block = EncoderBlockInit {
            input_channels: cx,
            output_channels: cy,
            repeat: 2,
            num_heads: 15,
            attention_channels: cx * 2,
            keyvalue_channels: cx * 2,
            ndims: 2,
            attention_ksize: 3,
            conv_transposed: false,
        }
        .build(&root)?;

        let input = Tensor::rand(&[b, cx as i64, hx, wx], FLOAT_CPU);
        let EncoderBlockOutput {
            feature,
            mask: _,
            contexts: _,
        } = block.forward_t(&input, None, true)?;

        ensure!(
            feature.size() == vec![b, cy as i64, hx as i64, wx as i64],
            "incorrect output shape"
        );

        Ok(())
    }
}
