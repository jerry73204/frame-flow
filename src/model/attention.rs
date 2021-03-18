use super::conv::{ConvND, ConvNDInit, ConvParam};
use crate::common::*;

pub struct AttentionInit<InputConvParam, ContextConvParam>
where
    InputConvParam: ConvParam,
    ContextConvParam: ConvParam,
{
    pub num_heads: usize,
    pub input_channels: usize,
    pub context_channels: usize,
    pub output_channels: usize,
    pub key_channels: usize,
    pub value_channels: usize,
    pub input_conv: ConvNDInit<InputConvParam>,
    pub context_conv: ConvNDInit<ContextConvParam>,
}

impl<InputConvParam, ContextConvParam> AttentionInit<InputConvParam, ContextConvParam>
where
    InputConvParam: ConvParam,
    ContextConvParam: ConvParam,
{
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<Attention> {
        let path = path.borrow();
        let AttentionInit {
            num_heads,
            input_channels,
            context_channels,
            output_channels,
            key_channels,
            value_channels,
            input_conv,
            context_conv,
        } = self;

        let mask_conv = {
            // let mask_dim = INPUT_DIM + CONTEXT_DIM;
            // let arr = [0i64; INPUT_DIM + CONTEXT_DIM];
            // let mask_conv = ConvNDInit {

            // };
        };

        let query_conv = input_conv.build(
            path / "query_conv",
            input_channels,
            num_heads * key_channels,
        )?;
        let key_conv = context_conv.clone().build(
            path / "key_conv",
            context_channels,
            num_heads * key_channels,
        )?;
        let value_conv = context_conv.build(
            path / "value_conv",
            context_channels,
            num_heads * value_channels,
        )?;
        let merge_weight = path.var(
            "merge_weight",
            &[
                num_heads as i64,
                value_channels as i64,
                output_channels as i64,
            ],
            nn::Init::KaimingUniform,
        );

        Ok(Attention {
            num_heads: num_heads as i64,
            key_channels: key_channels as i64,
            value_channels: value_channels as i64,
            output_channels: output_channels as i64,
            query_conv,
            key_conv,
            value_conv,
            merge_weight,
        })
    }
}

pub struct Attention {
    num_heads: i64,
    key_channels: i64,
    value_channels: i64,
    output_channels: i64,
    merge_weight: Tensor,
    query_conv: ConvND,
    key_conv: ConvND,
    value_conv: ConvND,
}

impl Attention {
    pub fn forward(
        &self,
        input: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = input.size()[0];
        ensure!(context.size()[0] == batch_size, "batch size mismatch");

        let Self {
            num_heads,
            key_channels,
            value_channels,
            output_channels,
            ref query_conv,
            ref key_conv,
            ref value_conv,
            ref merge_weight,
        } = *self;

        // check mask shape
        let mask = mask
            .map(|mask| {
                let input_shape = &input.size()[2..];
                let context_shape = &context.size()[2..];
                let expect_shape: Vec<_> = input_shape
                    .iter()
                    .cloned()
                    .chain(context_shape.iter().cloned())
                    .collect();
                ensure!(mask.size() == expect_shape, "mask shape mismatch");
                Ok(mask)
            })
            .transpose()?;

        // convolutions
        let (query, new_input_shape) = {
            let orig = query_conv.forward(input);
            let shape = orig.size()[2..].to_owned();
            let new = orig.view([batch_size, num_heads, key_channels, -1]);
            (new, shape)
        };
        let (key, new_context_shape) = {
            let orig = key_conv.forward(context);
            let shape = orig.size()[2..].to_owned();
            let new = orig.view([batch_size, num_heads, key_channels, -1]);
            (new, shape)
        };
        let value = {
            let orig = value_conv.forward(context);
            debug_assert_eq!(orig.size()[2..], new_context_shape);
            orig.view([batch_size, num_heads, value_channels, -1])
        };

        // compute attention
        let attention = query
            // [b, h, c, m] -> [b, h, m, c]
            .permute(&[0, 1, 3, 2])
            .matmul(&key)
            .g_div1((key_channels as f64).sqrt())
            .softmax(2, Kind::Float);

        // mask attention
        let attention = match mask {
            Some(mask) => {
                // let mask = mask.borrow();
                attention
            }
            None => attention,
        };

        // compute output per head
        let head_outputs = Tensor::einsum("bhxy,bhvy->bhvx", &[&attention, &value]);

        // merge head outputs
        let output = Tensor::einsum("hvo,bhvx->box", &[merge_weight, &head_outputs]);
        let output = {
            let output_shape: Vec<_> = vec![batch_size, output_channels]
                .into_iter()
                .chain(new_input_shape)
                .collect();
            output.view(&*output_shape)
        };

        Ok(output)
    }
}

// fn array_cat<T, const LHS_SIZE: usize, const RHS_SIZE: usize>(
//     lhs: [T; LHS_SIZE],
//     rhs: [T; RHS_SIZE],
// ) -> [T; LHS_SIZE + RHS_SIZE] {
//     todo!();
// }

// trait ArrayCat<Rhs> {
//     const OUT: usize;
//     type Output;
// }

// impl<T, const LHS: usize, const RHS: usize> ArrayCat<[T; RHS]> for [T; LHS] {
//     const OUT: usize = LHS + RHS;
//     type Output = [T; Self::OUT];
// }

#[cfg(test)]
mod tests {
    use super::{super::conv::ConvNDInit2D, *};

    #[test]
    fn attention_test() -> Result<()> {
        let b = 11;
        let cx = 3;
        let cc = 5;
        let hx = 40;
        let wx = 50;
        let hc = 30;
        let wc = 70;
        let o = 2;
        let ki = 7;
        let kc = 9;

        let input = Tensor::randn(&[b, cx, hx, wx], FLOAT_CPU);
        let context = Tensor::randn(&[b, cc, hc, wc], FLOAT_CPU);

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let attention = AttentionInit {
            num_heads: 8,
            input_channels: cx as usize,
            context_channels: cc as usize,
            output_channels: o as usize,
            key_channels: 4,
            value_channels: 6,
            input_conv: ConvNDInit2D::new(ki),
            context_conv: ConvNDInit2D::new(kc),
        }
        .build(&root)?;

        let output = attention.forward(&input, &context, None)?;
        assert_eq!(output.size(), vec![b, o, hx, wx]);

        Ok(())
    }
}
