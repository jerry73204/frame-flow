use super::{
    attention::{Attention, AttentionInit},
    conv::{ConvND, ConvNDInitDyn},
    conv_bn::{ConvBnND, ConvBnNDInitDyn},
};
use crate::common::*;

pub struct EncoderBlockInit {
    pub ndims: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub repeat: usize,
    pub num_heads: usize,
    pub attention_channels_scale: f64,
    pub keyvalue_channels_scale: f64,
    pub attention_ksize: usize,
}

impl EncoderBlockInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<EncoderBlock> {
        let path = path.borrow();
        let Self {
            ndims,
            input_channels,
            attention_channels_scale,
            keyvalue_channels_scale,
            output_channels,
            repeat,
            num_heads,
            attention_ksize,
        } = self;

        let attention_channels = (input_channels as f64 * attention_channels_scale).ceil() as usize;
        let key_channels = (input_channels as f64 * keyvalue_channels_scale).ceil() as usize;
        let value_channels = key_channels;

        let shortcut_conv = ConvNDInitDyn::new(ndims, 1).build(
            path / "shortcut_conv",
            input_channels,
            attention_channels,
        )?;

        let pre_attention_conv = ConvBnNDInitDyn::new(ndims, 1).build(
            path / "pre_attention_conv",
            input_channels,
            attention_channels,
        )?;
        let post_attention_conv = ConvBnNDInitDyn::new(ndims, 1).build(
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
                    key_channels,
                    value_channels,
                    input_conv: ConvNDInitDyn::new(ndims, attention_ksize),
                    context_conv: ConvNDInitDyn::new(ndims, attention_ksize),
                }
                .build(path / format!("attention_{}", index))
            })
            .try_collect()?;

        let merge_conv = ConvNDInitDyn::new(ndims, 1).build(
            path / "merge_conv",
            attention_channels * 2,
            output_channels,
        )?;

        Ok(EncoderBlock {
            shortcut_conv,
            attentions,
            merge_conv,
            pre_attention_conv,
            post_attention_conv,
        })
    }
}

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
        input: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let Self {
            ref attentions,
            ref shortcut_conv,
            ref merge_conv,
            ref mut pre_attention_conv,
            ref mut post_attention_conv,
        } = *self;

        let shortcut = shortcut_conv.forward(&input);
        let (branch, output_mask) = {
            let xs = pre_attention_conv.forward_t(input, train)?;

            let (xs, mask) = attentions.iter().try_fold(
                (xs, mask.map(|mask| mask.shallow_clone())),
                |(xs, mask), attention| attention.forward(&xs, &xs, mask.as_ref()),
            )?;

            let xs = post_attention_conv.forward_t(&xs, train)?;

            (xs, mask)
        };
        let merge = Tensor::cat(&[shortcut, branch], 1);
        let output = merge_conv.forward(&merge);

        Ok((output, output_mask))
    }
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
            attention_channels_scale: 2.0,
            keyvalue_channels_scale: 2.0,
            ndims: 2,
            attention_ksize: 3,
        }
        .build(&root)?;

        let input = Tensor::rand(&[b, cx as i64, hx, wx], FLOAT_CPU);
        let (output, _output_mask) = block.forward_t(&input, None, true)?;

        ensure!(
            output.size() == vec![b, cy as i64, hx as i64, wx as i64],
            "incorrect output shape"
        );

        Ok(())
    }
}
