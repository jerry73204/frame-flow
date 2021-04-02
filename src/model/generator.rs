use super::block::{Block, BlockInit, BlockOutput};
use crate::common::*;
use tch_goodies::module::{ConvND, ConvNDInitDyn};

#[derive(Debug, Clone)]
pub struct GeneratorInit<const DEPTH: usize> {
    pub ndims: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub num_heads: usize,
    pub strides: [usize; DEPTH],
    pub block_channels: [usize; DEPTH],
    pub context_channels: [usize; DEPTH],
    pub repeats: [usize; DEPTH],
}

impl<const DEPTH: usize> GeneratorInit<DEPTH> {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<Generator> {
        ensure!(DEPTH >= 1, "zero depth is not allowed");

        let path = path.borrow();
        let Self {
            ndims,
            num_heads,
            input_channels,
            output_channels,
            strides,
            block_channels,
            context_channels,
            repeats,
        } = self;
        let first_attention_channels = block_channels[0];
        let last_attention_channels = *block_channels.last().unwrap();

        let first_conv = ConvNDInitDyn::new(ndims, 1).build(
            path / "first_conv",
            input_channels,
            first_attention_channels,
        )?;
        let last_conv = ConvNDInitDyn {
            transposed: true,
            ..ConvNDInitDyn::new(ndims, 1)
        }
        .build(
            path / "last_conv",
            first_attention_channels,
            output_channels,
        )?;

        // encoder blocks
        let enc_blocks = izip!(
            array::IntoIter::new(repeats),
            array::IntoIter::new(block_channels),
            iter::once(first_attention_channels).chain(array::IntoIter::new(block_channels)),
            array::IntoIter::new(context_channels),
        )
        .enumerate()
        .map(|(index, (repeat, out_c, in_c, context_c))| -> Result<_> {
            let block = BlockInit {
                ndims,
                input_channels: in_c,
                context_channels: context_c,
                output_channels: out_c,
                repeat,
                num_heads,
                keyvalue_channels: context_c,
                conv_ksize: 3,
                conv_transposed: false,
            }
            .build(path / format!("enc_block_{}", index))?;
            Ok(block)
        })
        .try_collect()?;

        // down sample modules
        let down_samples: Vec<_> = izip!(
            array::IntoIter::new(strides),
            array::IntoIter::new(block_channels),
        )
        .enumerate()
        .map(|(index, (stride, inout_c))| -> Result<_> {
            let conv = ConvNDInitDyn {
                stride: vec![stride; ndims],
                transposed: false,
                ..ConvNDInitDyn::new(ndims, 1)
            }
            .build(path / format!("down_sample_{}", index), inout_c, inout_c)?;
            Ok(conv)
        })
        .try_collect()?;

        // decoder blocks
        let dec_blocks: Vec<_> = izip!(
            array::IntoIter::new(repeats).rev(),
            array::IntoIter::new(block_channels).rev(),
            array::IntoIter::new(block_channels)
                .rev()
                .chain(iter::once(first_attention_channels))
                .skip(1),
            array::IntoIter::new(context_channels).rev(),
        )
        .enumerate()
        .map(|(index, (repeat, in_c, out_c, context_c))| -> Result<_> {
            let block = BlockInit {
                ndims,
                input_channels: in_c,
                context_channels: context_c,
                output_channels: out_c,
                repeat,
                num_heads,
                keyvalue_channels: context_c,
                conv_ksize: 3,
                conv_transposed: true,
            }
            .build(path / format!("enc_block_{}", index))?;

            Ok(block)
        })
        .try_collect()?;

        // up sample modules
        let up_samples: Vec<_> = izip!(
            array::IntoIter::new(strides).rev(),
            array::IntoIter::new(block_channels).rev(),
        )
        .enumerate()
        .map(|(index, (stride, inout_c))| -> Result<_> {
            let conv = ConvNDInitDyn {
                stride: vec![stride; ndims],
                transposed: true,
                ..ConvNDInitDyn::new(ndims, 1)
            }
            .build(path / format!("down_sample_{}", index), inout_c, inout_c)?;
            Ok(conv)
        })
        .try_collect()?;

        // top block

        let top_block = BlockInit {
            ndims,
            input_channels: last_attention_channels,
            context_channels: last_attention_channels,
            output_channels: last_attention_channels,
            repeat: 3,
            num_heads,
            keyvalue_channels: last_attention_channels,
            conv_ksize: 3,
            conv_transposed: false,
        }
        .build(path / "top_block")?;

        Ok(Generator {
            enc_blocks,
            dec_blocks,
            down_samples,
            up_samples,
            top_block,
            first_conv,
            last_conv,
            num_contexts: repeats.into(),
        })
    }
}

#[derive(Debug)]
pub struct Generator {
    enc_blocks: Vec<Block>,
    dec_blocks: Vec<Block>,
    down_samples: Vec<ConvND>,
    up_samples: Vec<ConvND>,
    top_block: Block,
    first_conv: ConvND,
    last_conv: ConvND,
    num_contexts: Vec<usize>,
}

impl Generator {
    pub fn forward_t(
        &mut self,
        input: &Tensor,
        contexts: impl OptionalTensorList,
        train: bool,
    ) -> Result<GeneratorOutput> {
        let Self {
            enc_blocks,
            dec_blocks,
            down_samples,
            up_samples,
            top_block,
            first_conv,
            last_conv,
            num_contexts,
        } = self;
        let input_contexts = contexts.into_optional_tensor_list();

        ensure!(
            input_contexts
                .as_ref()
                .map(|contexts| {
                    let total: usize = num_contexts.iter().cloned().sum();
                    contexts.len() == total
                })
                .unwrap_or(true),
            "number of contexts does not match"
        );

        // first conv
        let xs = first_conv.forward(input);

        // encoder
        let (xs, contexts_vec, mut output_paddings_vec) = {
            let mut contexts_vec = vec![];
            let mut output_paddings_vec = vec![];

            let xs = match input_contexts {
                Some(input_contexts) => {
                    let input_contexts_iter = num_contexts.iter().scan(0, |index, len| {
                        let curr = *index;
                        let next = curr + len;
                        *index += next;
                        Some(&input_contexts[curr..next])
                    });

                    izip!(
                        enc_blocks.iter_mut(),
                        down_samples.iter_mut(),
                        input_contexts_iter
                    )
                    .try_fold(
                        xs,
                        |xs, (block, down_sample, in_contexts)| -> Result<_> {
                            let BlockOutput {
                                feature: xs,
                                contexts: out_contexts,
                                ..
                            } = block.forward_t(xs, in_contexts, None, train)?;
                            let before_shape = xs.size();
                            let xs = down_sample.forward(&xs);
                            let after_shape = xs.size();
                            let output_paddings: Vec<_> =
                                izip!(&before_shape[2..], &after_shape[2..], down_sample.stride())
                                    .map(|(&before_size, &after_size, &stride)| {
                                        before_size - ((after_size - 1) * stride + 1)
                                    })
                                    .collect();

                            contexts_vec.push(out_contexts);
                            output_paddings_vec.push(output_paddings);

                            Ok(xs)
                        },
                    )?
                }
                None => izip!(enc_blocks.iter_mut(), down_samples.iter_mut()).try_fold(
                    xs,
                    |xs, (block, down_sample)| -> Result<_> {
                        let BlockOutput {
                            feature: xs,
                            contexts: out_contexts,
                            ..
                        } = block.forward_t(xs, NONE_TENSORS, None, train)?;
                        let before_shape = xs.size();
                        let xs = down_sample.forward(&xs);
                        let after_shape = xs.size();
                        let output_paddings: Vec<_> =
                            izip!(&before_shape[2..], &after_shape[2..], down_sample.stride())
                                .map(|(&before_size, &after_size, &stride)| {
                                    before_size - ((after_size - 1) * stride + 1)
                                })
                                .collect();

                        contexts_vec.push(out_contexts);
                        output_paddings_vec.push(output_paddings);

                        Ok(xs)
                    },
                )?,
            };

            (xs, contexts_vec, output_paddings_vec)
        };

        // top
        let BlockOutput { feature: xs, .. } = top_block.forward_t(xs, NONE_TENSORS, None, train)?;

        // reverse contexts
        let contexts_vec: Vec<_> = contexts_vec
            .into_iter()
            .rev()
            .map(|mut contexts| {
                contexts.reverse();
                contexts
            })
            .collect();
        output_paddings_vec.reverse();

        // decoder
        let (xs, mut output_contexts) = {
            let mut output_contexts = vec![];

            let xs = izip!(
                dec_blocks.iter_mut(),
                up_samples.iter_mut(),
                contexts_vec,
                output_paddings_vec
            )
            .try_fold(
                xs,
                |xs, (block, up_sample, contexts, output_paddings)| -> Result<_> {
                    let xs = up_sample.forward_ext(&xs, Some(&output_paddings));

                    let BlockOutput {
                        feature: xs,
                        contexts: out_contexts,
                        ..
                    } = block.forward_t(&xs, &contexts, None, train)?;

                    output_contexts.extend(out_contexts);
                    Ok(xs)
                },
            )?;

            (xs, output_contexts)
        };

        // last conv
        let xs = last_conv.forward(&xs);

        // reverse output contexts
        output_contexts.reverse();

        Ok(GeneratorOutput {
            output: xs,
            contexts: output_contexts,
        })
    }
}

#[derive(Debug)]
pub struct GeneratorOutput {
    pub output: Tensor,
    pub contexts: Vec<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_test() -> Result<()> {
        let bs = 2;
        let cx = 3;
        let cy = 4;
        let hx = 11;
        let wx = 13;

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let mut generator = GeneratorInit {
            ndims: 2,
            input_channels: cx,
            output_channels: cy,
            num_heads: 7,
            strides: [2, 2],
            block_channels: [8, 16],
            context_channels: [5, 10],
            repeats: [2, 2],
        }
        .build(root)?;

        // first pass, without contexts
        let input = Tensor::rand(&[bs, cx as i64, hx, wx], FLOAT_CPU);
        let GeneratorOutput {
            output, contexts, ..
        } = generator.forward_t(&input, NONE_TENSORS, true)?;
        ensure!(
            output.size() == vec![bs, cy as i64, hx, wx],
            "incorrect output shape"
        );

        // second pass, with contexts
        let GeneratorOutput {
            output, contexts, ..
        } = generator.forward_t(&input, contexts, true)?;
        ensure!(
            output.size() == vec![bs, cy as i64, hx, wx],
            "incorrect output shape"
        );

        // third pass, with contexts
        let GeneratorOutput { output, .. } = generator.forward_t(&input, contexts, true)?;
        ensure!(
            output.size() == vec![bs, cy as i64, hx, wx],
            "incorrect output shape"
        );

        Ok(())
    }
}
