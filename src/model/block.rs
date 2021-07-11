use super::attention::{Attention, AttentionGrad, AttentionInit};
use crate::common::*;
use tch_modules::{
    ConvBn, ConvBnGrad, ConvBnInitDyn, ConvND, ConvNDGrad, ConvNDInitDyn, DarkBatchNorm,
    DarkBatchNormGrad, DarkBatchNormInit,
};

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
        // let var_min = Some(r64(1e-4));
        let var_min = None;

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
            bn: DarkBatchNormInit {
                var_min,
                ..Default::default()
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
            bn: DarkBatchNormInit {
                var_min,
                ..Default::default()
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
        let attention_bns: Vec<_> = (0..repeat)
            .map(|index| {
                DarkBatchNormInit {
                    var_min,
                    ..Default::default()
                }
                .build(
                    path / format!("attention_bn_{}", index),
                    context_channels as i64,
                )
            })
            .collect();

        let merge_conv = ConvNDInitDyn {
            transposed: conv_transposed,
            ..ConvNDInitDyn::new(ndims, 1)
        }
        .build(path / "merge_conv", context_channels * 2, output_channels)?;

        Ok(Block {
            attentions,
            attention_bns,
            shortcut_conv,
            merge_conv,
            pre_attention_conv,
            post_attention_conv,
        })
    }
}

#[derive(Debug)]
pub struct Block {
    attention_bns: Vec<DarkBatchNorm>,
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
        &self,
        input: impl Borrow<Tensor>,
        contexts: impl OptionalTensorList,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<BlockOutput> {
        let Self {
            ref attention_bns,
            ref attentions,
            ref shortcut_conv,
            ref merge_conv,
            ref pre_attention_conv,
            ref post_attention_conv,
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
            let xs = pre_attention_conv.forward_t(input, train);
            let mut output_contexts = vec![];

            let (xs, mask) = match input_contexts {
                Some(input_contexts) => {
                    izip!(attentions.iter(), attention_bns.iter(), input_contexts)
                        .enumerate()
                        .try_fold(
                            (xs, mask),
                            |(xs, mask), (_index, (attention, bn, input_context))| -> Result<_> {
                                // eprintln!("attention {}", index);
                                // dbg!(xs.mean(Kind::Float));
                                let xs = bn.forward_t(&xs, train);
                                // dbg!(xs.mean(Kind::Float));
                                let (xs, mask) =
                                    attention.forward(&xs, &input_context, mask.as_ref())?;
                                output_contexts.push(xs.shallow_clone());
                                Ok((xs, mask))
                            },
                        )?
                }
                None => izip!(attentions.iter(), attention_bns.iter())
                    .enumerate()
                    .try_fold(
                        (xs, mask),
                        |(xs, mask), (_index, (attention, bn))| -> Result<_> {
                            // eprintln!("attention (no context) {}", index);
                            // dbg!(xs.mean(Kind::Float));
                            let xs = bn.forward_t(&xs, train);
                            // dbg!(xs.mean(Kind::Float));
                            let (xs, mask) = attention.forward(&xs, &xs, mask.as_ref())?;
                            output_contexts.push(xs.shallow_clone());
                            Ok((xs, mask))
                        },
                    )?,
            };

            let xs = post_attention_conv.forward_t(&xs, train);
            (xs, output_contexts, mask)
        };
        // dbg!(branch.max(), branch.min());

        let merge = Tensor::cat(&[shortcut, branch], 1);
        let output = merge_conv.forward(&merge);
        // dbg!(output.max(), output.min());

        Ok(BlockOutput {
            feature: output,
            contexts: output_contexts,
            mask: output_mask,
        })
    }

    // pub fn clamp_bn_var(&mut self) {
    //     let Self {
    //         attention_bns,
    //         pre_attention_conv,
    //         post_attention_conv,
    //         ..
    //     } = self;

    //     attention_bns.iter_mut().for_each(|bn| {
    //         bn.clamp_bn_var();
    //     });
    //     pre_attention_conv.clamp_bn_var();
    //     post_attention_conv.clamp_bn_var();
    // }

    // pub fn denormalize_bn(&mut self) {
    //     let Self {
    //         attention_bns,
    //         pre_attention_conv,
    //         post_attention_conv,
    //         ..
    //     } = self;

    //     attention_bns.iter_mut().for_each(|bn| {
    //         bn.denormalize_bn();
    //     });
    //     pre_attention_conv.denormalize_bn();
    //     post_attention_conv.denormalize_bn();
    // }

    pub fn grad(&self) -> BlockGrad {
        let Self {
            attention_bns,
            attentions,
            shortcut_conv,
            merge_conv,
            pre_attention_conv,
            post_attention_conv,
        } = self;

        BlockGrad {
            attention_bns: attention_bns.iter().map(|bn| bn.grad()).collect(),
            attentions: attentions.iter().map(|att| att.grad()).collect(),
            shortcut_conv: shortcut_conv.grad(),
            merge_conv: merge_conv.grad(),
            pre_attention_conv: pre_attention_conv.grad(),
            post_attention_conv: post_attention_conv.grad(),
        }
    }
}

#[derive(Debug)]
pub struct BlockOutput {
    pub feature: Tensor,
    pub contexts: Vec<Tensor>,
    pub mask: Option<Tensor>,
}

#[derive(Debug, TensorLike)]
pub struct BlockGrad {
    pub attention_bns: Vec<DarkBatchNormGrad>,
    pub attentions: Vec<AttentionGrad>,
    pub shortcut_conv: ConvNDGrad,
    pub merge_conv: ConvNDGrad,
    pub pre_attention_conv: ConvBnGrad,
    pub post_attention_conv: ConvBnGrad,
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

        let block = BlockInit {
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
