use super::{
    attention::{Attention, AttentionInit},
    conv::ConvNDInit,
};
use crate::common::*;

pub struct BlockInit<const DIM: usize> {
    pub input_channels: usize,
    pub output_channels: usize,
    pub key_channels: usize,
    pub value_channels: usize,
    pub num_heads: usize,
    pub conv: ConvNDInit<DIM>,
}

impl<const DIM: usize> BlockInit<DIM> {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<Block> {
        let path = path.borrow();
        let Self {
            input_channels,
            output_channels,
            key_channels,
            value_channels,
            num_heads,
            conv,
        } = self;

        let attention = AttentionInit {
            num_heads,
            input_channels,
            context_channels: input_channels,
            output_channels,
            key_channels,
            value_channels,
            input_conv: conv.clone(),
            context_conv: conv,
        }
        .build(path / "attention")?;

        Ok(Block { attention })
    }
}

pub struct Block {
    attention: Attention,
}

impl Block {
    pub fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self { ref attention } = *self;

        let addon = attention.forward(input, input, None)?;

        todo!();
    }
}
