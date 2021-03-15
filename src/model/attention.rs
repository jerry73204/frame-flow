use crate::common::*;

pub struct AttentionInit {
    pub num_heads: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub key_channels: usize,
    pub value_channels: usize,
    pub input_block_size: usize,
    pub state_block_size: usize,
}

impl AttentionInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Attention {
        let path = path.borrow();
        let AttentionInit {
            num_heads,
            input_channels,
            output_channels,
            key_channels,
            value_channels,
            input_block_size,
            state_block_size,
        } = self;

        let num_heads = num_heads as i64;
        let input_channels = input_channels as i64;
        let output_channels = output_channels as i64;
        let key_channels = key_channels as i64;
        let value_channels = value_channels as i64;

        let query_weight = path.var(
            "query_weight",
            &[num_heads, input_channels, key_channels],
            nn::Init::KaimingUniform,
        );
        let key_weight = path.var(
            "key_weight",
            &[num_heads, input_channels, key_channels],
            nn::Init::KaimingUniform,
        );
        let value_weight = path.var(
            "value_weight",
            &[num_heads, input_channels, value_channels],
            nn::Init::KaimingUniform,
        );
        let merge_weight = path.var(
            "merge_weight",
            &[num_heads, value_channels, output_channels],
            nn::Init::KaimingUniform,
        );

        Attention {
            query_weight,
            key_weight,
            value_weight,
            merge_weight,
            key_channels: key_channels as i64,
            input_block_size: input_block_size as i64,
            state_block_size: state_block_size as i64,
        }
    }
}

pub struct Attention {
    query_weight: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    merge_weight: Tensor,
    key_channels: i64,
    input_block_size: i64,
    state_block_size: i64,
}

impl Attention {
    pub fn forward_t(&self, input: &Tensor, state: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            ref query_weight,
            ref key_weight,
            ref value_weight,
            ref merge_weight,
            key_channels,
            input_block_size,
            state_block_size,
            ..
        } = *self;

        let input_shape = input.size4().with_context(|| {
            format!(
                "expect state shape [B, C, H, W], but get {:?}",
                input.size()
            )
        })?;
        let state_shape = state.size4().with_context(|| {
            format!(
                "expect input shape [B, C, H, W], but get {:?}",
                state.size()
            )
        })?;
        ensure!(input_shape == state_shape, "input and state shape mismatch");

        // unfolding
        let input_padding = input_block_size / 2;
        let state_padding = state_block_size / 2;

        let input_unfolded = input.unfold2d(
            &[input_block_size, input_block_size],
            &[1, 1], // dilation
            &[input_padding, input_padding],
            &[1, 1], // stride
        );
        let state_unfolded = state.unfold2d(
            &[state_block_size, state_block_size],
            &[1, 1], // dilation
            &[state_padding, state_padding],
            &[1, 1], // stride
        );

        let query = Tensor::einsum("hck,bcpqij->bhkpqij", &[query_weight, &state_unfolded]);

        let key = Tensor::einsum("hck,bcpqij->bhkpqij", &[key_weight, &input_unfolded]);

        let value = Tensor::einsum("hck,bcpqij->bhkpqij", &[value_weight, &input_unfolded]);

        let attention = Tensor::einsum("bhkpqij,bhkrsij->bhpqrsij", &[&key, &query])
            .g_div1((key_channels as f64).sqrt())
            .multi_softmax(&[2, 3], Kind::Float);

        let head_outputs = Tensor::einsum("bhpqrsij,bhvpqij->bhvrsij", &[&attention, &value]);

        let output = Tensor::einsum("bhvrsij,hvw->bwrsij", &[&head_outputs, &merge_weight]);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_test() -> Result<()> {
        let b = 6;
        let c = 3;
        let h = 40;
        let w = 50;
        let o = 2;
        let sb = 5i64; // state block size

        let input = Tensor::randn(&[b, c, h, w], FLOAT_CPU);
        let state = Tensor::randn(&[b, c, h, w], FLOAT_CPU);

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let attention = AttentionInit {
            num_heads: 8,
            input_channels: c as usize,
            output_channels: o as usize,
            key_channels: 4,
            value_channels: 6,
            input_block_size: 7,
            state_block_size: sb as usize,
        }
        .build(&root);

        let output = attention.forward_t(&input, &state, false)?;
        assert_eq!(output.size(), vec![b, o, sb, sb, h, w]);

        Ok(())
    }
}
