use super::conv::{ConvND, ConvNDInit, ConvNDInitDyn, ConvParam};
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

        let mask_input_conv = ConvNDInitDyn {
            groups: 1,
            bias: false,
            ws_init: nn::Init::Const(1.0),
            bs_init: nn::Init::Const(0.0),
            ..input_conv.clone().into_dyn()
        }
        .build(path / "mask_input_conv", 1, 1)
        .unwrap();
        mask_input_conv.set_trainable(false);

        let mask_context_conv = ConvNDInitDyn {
            groups: 1,
            bias: false,
            ws_init: nn::Init::Const(1.0),
            bs_init: nn::Init::Const(0.0),
            ..context_conv.clone().into_dyn()
        }
        .build(path / "mask_input_conv", 1, 1)
        .unwrap();
        mask_context_conv.set_trainable(false);

        let mask_divisor = input_conv.ksize.usize_iter().product::<usize>()
            * context_conv.ksize.usize_iter().product::<usize>();

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
            mask_input_conv,
            mask_context_conv,
            merge_weight,
            mask_divisor: mask_divisor as f64,
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
    mask_input_conv: ConvND,
    mask_context_conv: ConvND,
    mask_divisor: f64,
}

impl Attention {
    pub fn forward<'a>(
        &self,
        input: &Tensor,
        context: &Tensor,
        mask: impl Into<Option<&'a Tensor>>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let batch_size = input.size()[0];
        let input_mask = mask.into();
        ensure!(context.size()[0] == batch_size, "batch size mismatch");

        let Self {
            num_heads,
            key_channels,
            value_channels,
            output_channels,
            mask_divisor,
            ref query_conv,
            ref key_conv,
            ref value_conv,
            ref mask_input_conv,
            ref mask_context_conv,
            ref merge_weight,
        } = *self;

        let input_shape = &input.size()[2..];
        let context_shape = &context.size()[2..];
        let input_numel: i64 = input_shape.iter().product();
        // let context_numel: i64 = context_shape.iter().product();

        // check mask shape
        if let Some(input_mask) = input_mask {
            let expect_shape: Vec<_> = input_shape
                .iter()
                .cloned()
                .chain(context_shape.iter().cloned())
                .collect();
            ensure!(!input_mask.requires_grad(), "mask must be non-trainable");
            ensure!(input_mask.size() == expect_shape, "mask shape mismatch");
        }

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
        let new_input_numel: i64 = new_input_shape.iter().product();
        let new_context_numel: i64 = new_context_shape.iter().product();

        // compute attention
        let (attention, output_mask) = {
            let logit = query
                // [b, h, c, mx] -> [b, h, mx, c]
                .permute(&[0, 1, 3, 2])
                // [b, h, mx, c] * [b, h, c, my] -> [b, h, mx, my]
                .matmul(&key)
                .g_div1((key_channels as f64).sqrt());

            // mask attention
            let (masked_logit, output_mask) = match input_mask {
                Some(mask) => {
                    // transform mask by convolutions
                    let shape1: Vec<_> = vec![input_numel, 1]
                        .into_iter()
                        .chain(context_shape.iter().cloned())
                        .collect();
                    let mask = mask.view(&*shape1);
                    let mask = mask_context_conv.forward(&mask);

                    let shape2: Vec<_> = vec![new_context_numel, 1]
                        .into_iter()
                        .chain(input_shape.iter().cloned())
                        .collect();
                    let mask = mask
                        .view([input_numel, 1, new_context_numel])
                        .permute(&[2, 1, 0])
                        .reshape(&*shape2);
                    let mask = mask_input_conv.forward(&mask);

                    let mask = mask
                        .view([new_context_numel, new_input_numel])
                        .permute(&[1, 0])
                        .reshape(&[1, 1, new_input_numel, new_context_numel]);
                    let mask = mask / mask_divisor;

                    // apply masking
                    let masked_logit = &logit + &mask.log();

                    // transform mask shape
                    let shape3: Vec<_> = new_input_shape
                        .iter()
                        .cloned()
                        .chain(new_context_shape.iter().cloned())
                        .collect();
                    let output_mask = mask.view(&*shape3);

                    (masked_logit, Some(output_mask))
                }
                None => (logit.shallow_clone(), None),
            };

            let attention = masked_logit.softmax(3, Kind::Float);
            (attention, output_mask)
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

        Ok((output, output_mask))
    }
}

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
        let mask = Tensor::ones(&[hx, wx, hc, wc], FLOAT_CPU).set_requires_grad(false);

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let attention = AttentionInit {
            num_heads: 8,
            input_channels: cx as usize,
            context_channels: cc as usize,
            output_channels: o as usize,
            key_channels: 4,
            value_channels: 6,
            input_conv: ConvNDInit2D {
                stride: [2, 2],
                ..ConvNDInit2D::new(ki)
            },
            context_conv: ConvNDInit2D {
                stride: [2, 2],
                ..ConvNDInit2D::new(kc)
            },
        }
        .build(&root)?;

        let (output, out_mask) = attention.forward(&input, &context, &mask)?;
        let out_mask = out_mask.unwrap();

        assert_eq!(output.size(), vec![b, o, hx / 2, wx / 2]);
        assert_eq!(out_mask.size(), vec![hx / 2, wx / 2, hc / 2, wc / 2]);
        assert_abs_diff_eq!(f64::from(out_mask.max()), 1.0);

        Ok(())
    }
}
