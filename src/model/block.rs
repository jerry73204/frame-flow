use super::{
    attention::{Attention, AttentionInit},
    batch_norm::BatchNormND,
    conv::{ConvND, ConvNDInit, ConvParam},
};
use crate::common::*;

pub struct BlockInit<Param1, Param2>
where
    Param1: ConvParam,
    Param2: ConvParam,
{
    pub num_heads: usize,
    pub input_channels: usize,
    pub attention_channels: usize,
    pub output_channels: usize,
    pub attention_channel_scale: f64,
    // pub key_channels: usize,
    // pub value_channels: usize,
    pub attention_conv: ConvNDInit<Param1>,
    pub post_conv: ConvNDInit<Param2>,
}

impl<Param1, Param2> BlockInit<Param1, Param2>
where
    Param1: ConvParam,
    Param2: ConvParam,
{
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<Block> {
        let path = path.borrow();
        let Self {
            input_channels,
            attention_channels,
            output_channels,
            attention_channel_scale,
            // key_channels,
            // value_channels,
            num_heads,
            attention_conv,
            post_conv,
        } = self;

        ensure!(
            attention_conv.dim()? == post_conv.dim()?,
            "attention and post convolution dimension mismatch"
        );
        let feature_dim = attention_conv.dim()?;
        let key_channels = (input_channels as f64 * attention_channel_scale).ceil() as usize;
        let value_channels = key_channels;

        let norm1 = BatchNormND::new(
            path / "attention_norm",
            feature_dim,
            input_channels as i64,
            Default::default(),
        );
        let attention = AttentionInit {
            num_heads,
            input_channels,
            context_channels: input_channels,
            output_channels: attention_channels,
            key_channels,
            value_channels,
            input_conv: attention_conv.clone(),
            context_conv: attention_conv.clone(),
        }
        .build(path / "attention")?;
        let norm2 = BatchNormND::new(
            path / "post_norm",
            feature_dim,
            attention_channels as i64,
            Default::default(),
        );
        let post =
            post_conv
                .clone()
                .build(path / "post_conv", attention_channels, output_channels)?;
        let post_mask_conv = {
            let conv = ConvNDInit {
                groups: 1,
                bias: false,
                ws_init: nn::Init::Const(1.0),
                bs_init: nn::Init::Const(0.0),
                ..post_conv
            }
            .build(path / "post_mask_conv", 1, 1)?;
            conv.set_trainable(false);
            conv
        };

        Ok(Block {
            attention,
            post,
            post_mask_conv,
            norm1,
            norm2,
        })
    }
}

pub struct Block {
    attention: Attention,
    post: ConvND,
    post_mask_conv: ConvND,
    norm1: BatchNormND,
    norm2: BatchNormND,
}

impl Block {
    pub fn forward_t(
        &mut self,
        input: &Tensor,
        input_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let Self {
            ref attention,
            ref post,
            ref mut norm1,
            ref mut norm2,
            ref post_mask_conv,
        } = *self;

        let shortcut1 = input;
        let (branch1, out_mask1) = {
            let xs = norm1.forward_t(input, train)?;
            attention.forward(&xs, input, input_mask)?
        };
        let merge1 = shortcut1 + branch1;

        let shortcut2 = &merge1;
        let branch2 = {
            let xs = norm2.forward_t(&merge1, train)?;
            post.forward(&xs)
        };
        let merge2 = shortcut2 + branch2;

        let out_mask2 = out_mask1.map(|mask| post_mask_conv.forward(&mask));

        Ok((merge2, out_mask2))
    }
}
