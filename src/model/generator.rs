use super::{
    dec_block::{DecoderBlock, DecoderBlockInit, DecoderBlockOutput},
    enc_block::{EncoderBlock, EncoderBlockInit, EncoderBlockOutput},
};
use crate::common::*;
use tch_goodies::module::{ConvND, ConvNDInitDyn};

#[derive(Debug, Clone)]
pub struct GeneratorInit<const DEPTH: usize> {
    pub ndims: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub num_heads: usize,
    pub strides: [usize; DEPTH],
    pub channels: [usize; DEPTH],
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
            channels,
            repeats,
        } = self;
        let channel_scale = 2;
        let first_attention_channels = channels[0];
        let last_attention_channels = *channels.last().unwrap();

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

        let (enc_blocks, context_channels_vec) = izip!(
            array::IntoIter::new(repeats),
            array::IntoIter::new(channels),
            iter::once(first_attention_channels).chain(array::IntoIter::new(channels)),
        )
        .enumerate()
        .map(|(index, (repeat, out_c, in_c))| -> Result<_> {
            let context_channels = in_c * channel_scale;
            let block = EncoderBlockInit {
                ndims,
                input_channels: in_c,
                output_channels: out_c,
                repeat,
                num_heads,
                attention_channels: context_channels,
                keyvalue_channels: in_c * channel_scale,
                attention_ksize: 3,
                conv_transposed: false,
            }
            .build(path / format!("enc_block_{}", index))?;
            Ok((block, context_channels))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip_n_vec();

        let down_samples: Vec<_> = izip!(
            array::IntoIter::new(strides),
            array::IntoIter::new(channels),
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

        let dec_blocks: Vec<_> = izip!(
            array::IntoIter::new(repeats).rev(),
            array::IntoIter::new(channels).rev(),
            array::IntoIter::new(channels)
                .rev()
                .chain(iter::once(first_attention_channels))
                .skip(1),
            context_channels_vec,
        )
        .enumerate()
        .map(|(index, (repeat, in_c, out_c, context_c))| -> Result<_> {
            let block = DecoderBlockInit {
                ndims,
                input_channels: in_c,
                context_channels: context_c,
                output_channels: out_c,
                repeat,
                num_heads,
                attention_channels: in_c * channel_scale,
                keyvalue_channels: in_c * channel_scale,
                attention_ksize: 3,
                conv_transposed: true,
            }
            .build(path / format!("enc_block_{}", index))?;

            Ok(block)
        })
        .try_collect()?;

        let up_samples: Vec<_> = izip!(
            array::IntoIter::new(strides).rev(),
            array::IntoIter::new(channels).rev(),
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

        let top_block = EncoderBlockInit {
            ndims,
            input_channels: last_attention_channels,
            output_channels: last_attention_channels,
            repeat: 3,
            num_heads,
            attention_channels: last_attention_channels * channel_scale,
            keyvalue_channels: last_attention_channels * channel_scale,
            attention_ksize: 3,
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
        })
    }
}

#[derive(Debug)]
pub struct Generator {
    enc_blocks: Vec<EncoderBlock>,
    dec_blocks: Vec<DecoderBlock>,
    down_samples: Vec<ConvND>,
    up_samples: Vec<ConvND>,
    top_block: EncoderBlock,
    first_conv: ConvND,
    last_conv: ConvND,
}

impl Generator {
    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            enc_blocks,
            dec_blocks,
            down_samples,
            up_samples,
            top_block,
            first_conv,
            last_conv,
        } = self;

        let xs = first_conv.forward(input);

        // encoder
        let (xs, contexts_vec, mut output_paddings_vec) =
            izip!(enc_blocks.iter_mut(), down_samples.iter_mut()).try_fold(
                (xs, vec![], vec![]),
                |(xs, mut contexts_vec, mut output_paddings_vec),
                 (block, down_sample)|
                 -> Result<_> {
                    let EncoderBlockOutput {
                        feature: xs,
                        contexts,
                        ..
                    } = block.forward_t(xs, None, train)?;
                    contexts_vec.push(contexts);

                    let before_shape = xs.size();

                    let xs = down_sample.forward(&xs);

                    let after_shape = xs.size();

                    let output_paddings: Vec<_> =
                        izip!(&before_shape[2..], &after_shape[2..], down_sample.stride())
                            .map(|(&before_size, &after_size, &stride)| {
                                before_size - ((after_size - 1) * stride + 1)
                            })
                            .collect();

                    output_paddings_vec.push(output_paddings);

                    Ok((xs, contexts_vec, output_paddings_vec))
                },
            )?;

        // top
        let EncoderBlockOutput { feature: xs, .. } = top_block.forward_t(xs, None, train)?;

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

                let DecoderBlockOutput { feature: xs, .. } =
                    block.forward_t(&xs, &contexts, None, train)?;

                Ok(xs)
            },
        )?;

        let xs = last_conv.forward(&xs);

        Ok(xs)
    }
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
            channels: [8, 16],
            repeats: [2, 2],
        }
        .build(root)?;

        let input = Tensor::rand(&[bs, cx as i64, hx, wx], FLOAT_CPU);
        let output = generator.forward_t(&input, true)?;

        ensure!(
            output.size() == vec![bs, cy as i64, hx, wx],
            "incorrect output shape"
        );

        Ok(())
    }
}
