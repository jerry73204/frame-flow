use super::attention::{Attention, AttentionInit};
use crate::common::*;
use tch_goodies::module::{ConvBn, ConvBnInitDyn, ConvND, ConvNDInitDyn};

#[derive(Debug, Clone)]
pub struct BlockInit {
    pub ndims: usize,
    pub input_channels: usize,
    pub context_channels: usize,
    pub output_channels: usize,
    pub repeat: usize,
    pub num_heads: usize,
    pub keyvalue_channels: usize,
    pub conv_ksize: usize,
    pub conv_transposed: bool,
}

impl BlockInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<Block> {
        let path = path.borrow();
        let Self {
            ndims,
            input_channels,
            context_channels,
            keyvalue_channels,
            output_channels,
            repeat,
            num_heads,
            conv_ksize,
            conv_transposed,
        } = self;

        let shortcut_conv = ConvNDInitDyn {
            transposed: conv_transposed,
            ..ConvNDInitDyn::new(ndims, 1)
        }
        .build(path / "shortcut_conv", input_channels, context_channels)?;

        let pre_attention_conv = ConvBnInitDyn {
            conv: ConvNDInitDyn {
                transposed: conv_transposed,
                ..ConvNDInitDyn::new(ndims, 1)
            },
            ..ConvBnInitDyn::new(ndims, 1)
        }
        .build(
            path / "pre_attention_conv",
            input_channels,
            context_channels,
        )?;
        let post_attention_conv = ConvBnInitDyn {
            conv: ConvNDInitDyn {
                transposed: conv_transposed,
                ..ConvNDInitDyn::new(ndims, 1)
            },
            ..ConvBnInitDyn::new(ndims, 1)
        }
        .build(
            path / "post_attention_conv",
            context_channels,
            context_channels,
        )?;
        let attentions: Vec<_> = (0..repeat)
            .map(|index| {
                AttentionInit {
                    num_heads,
                    input_channels: context_channels,
                    context_channels,
                    output_channels: context_channels,
                    key_channels: keyvalue_channels,
                    value_channels: keyvalue_channels,
                    input_conv: ConvNDInitDyn {
                        transposed: conv_transposed,
                        ..ConvNDInitDyn::new(ndims, conv_ksize)
                    },
                    context_conv: ConvNDInitDyn {
                        // stride: vec![context_stride; ndims],
                        transposed: conv_transposed,
                        ..ConvNDInitDyn::new(ndims, conv_ksize)
                    },
                }
                .build(path / format!("attention_{}", index))
            })
            .try_collect()?;

        let merge_conv = ConvNDInitDyn {
            transposed: conv_transposed,
            ..ConvNDInitDyn::new(ndims, 1)
        }
        .build(path / "merge_conv", context_channels * 2, output_channels)?;

        Ok(Block {
            attentions,
            shortcut_conv,
            merge_conv,
            pre_attention_conv,
            post_attention_conv,
        })
    }
}

#[derive(Debug)]
pub struct Block {
    attentions: Vec<Attention>,
    shortcut_conv: ConvND,
    merge_conv: ConvND,
    pre_attention_conv: ConvBn,
    post_attention_conv: ConvBn,
}

impl Block {
    pub fn num_contexts(&self) -> usize {
        self.attentions.len()
    }

    pub fn forward_t(
        &mut self,
        input: impl Borrow<Tensor>,
        contexts: impl OptionalTensorList,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<BlockOutput> {
        let Self {
            ref attentions,
            ref shortcut_conv,
            ref merge_conv,
            ref mut pre_attention_conv,
            ref mut post_attention_conv,
        } = *self;
        let input_contexts = contexts.into_optional_tensor_list();
        ensure!(
            input_contexts
                .as_ref()
                .map(|contexts| contexts.len() == attentions.len())
                .unwrap_or(true),
            "the number of context tensors does not match the number of attentions"
        );
        let input = input.borrow();
        let mask = mask.map(|mask| mask.detach());

        let shortcut = shortcut_conv.forward(input);
        let (branch, output_contexts, output_mask) = {
            let xs = pre_attention_conv.forward_t(input, train)?;
            let mut output_contexts = vec![];

            let (xs, mask) = match input_contexts {
                Some(input_contexts) => izip!(attentions.iter(), input_contexts).try_fold(
                    (xs, mask),
                    |(xs, mask), (attention, input_context)| -> Result<_> {
                        let (xs, mask) = attention.forward(&xs, &input_context, mask.as_ref())?;
                        output_contexts.push(xs.shallow_clone());
                        Ok((xs, mask))
                    },
                )?,
                None => attentions.iter().try_fold(
                    (xs, mask),
                    |(xs, mask), attention| -> Result<_> {
                        let (xs, mask) = attention.forward(&xs, &xs, mask.as_ref())?;
                        output_contexts.push(xs.shallow_clone());
                        Ok((xs, mask))
                    },
                )?,
            };

            let xs = post_attention_conv.forward_t(&xs, train)?;
            (xs, output_contexts, mask)
        };
        let merge = Tensor::cat(&[shortcut, branch], 1);
        let output = merge_conv.forward(&merge);

        Ok(BlockOutput {
            feature: output,
            contexts: output_contexts,
            mask: output_mask,
        })
    }
}

#[derive(Debug)]
pub struct BlockOutput {
    pub feature: Tensor,
    pub contexts: Vec<Tensor>,
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

        let mut block = BlockInit {
            input_channels: cx,
            context_channels: cxx,
            output_channels: cy,
            repeat,
            num_heads: 10,
            keyvalue_channels: cx * 2,
            ndims: 2,
            conv_ksize: 3,
            conv_transposed: false,
        }
        .build(&root)?;

        let in_contexts: Vec<_> = (0..repeat)
            .map(|_| Tensor::rand(&[b, cxx as i64, hx, wx], FLOAT_CPU))
            .collect();

        let input = Tensor::rand(&[b, cx as i64, hx, wx], FLOAT_CPU);
        let BlockOutput {
            feature: output,
            contexts: out_contexts,
            ..
        } = block.forward_t(&input, in_contexts, None, true)?;

        ensure!(
            output.size() == vec![b, cy as i64, hx as i64, wx as i64],
            "incorrect output shape"
        );

        out_contexts.iter().try_for_each(|context| -> Result<_> {
            ensure!(context.size() == vec![b, cxx as i64, hx, wx]);
            Ok(())
        })?;

        Ok(())
    }
}
