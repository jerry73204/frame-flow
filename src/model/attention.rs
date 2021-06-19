use crate::common::*;
use tch_modules::{ConvND, ConvNDGrad, ConvNDInit, ConvNDInitDyn, ConvParam};

#[derive(Debug, Clone)]
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

        // let bound = (6f64 / (num_heads * value_channels + output_channels) as f64).sqrt();

        let merge_weight = path.var(
            "merge_weight",
            &[
                num_heads as i64,
                value_channels as i64,
                output_channels as i64,
            ],
            // nn::Init::Uniform {
            //     lo: -bound,
            //     up: bound,
            // },
            nn::Init::Randn {
                mean: 0.0,
                stdev: (1.0 / (num_heads * value_channels) as f64).sqrt(),
            },
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

#[derive(Debug)]
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

        // dbg!(input.mean(Kind::Float), context.mean(Kind::Float));
        // dbg!(input.std(true), context.std(true));

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

        let key = key.softmax(3, Kind::Float);

        // let key =
        //     key / (key_channels as f64).sqrt() / (new_context_numel as f64).sqrt() * 2f64.sqrt();
        // let value = value / (new_context_numel as f64).sqrt();
        // let query = query / (key_channels as f64).sqrt();

        // let key = key.mish();
        // let value = value.mish();
        // let query = query.mish();

        // dbg!(query.mean(Kind::Float));
        // dbg!(key.mean(Kind::Float));
        // dbg!(value.mean(Kind::Float));

        debug_assert!(!query.has_nan(), "NaN detected");
        debug_assert!(!key.has_nan(), "NaN detected");
        debug_assert!(!value.has_nan(), "NaN detected");

        // transform mask by convolutions
        let output_mask = input_mask.map(|mask| {
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
                .reshape(&[new_input_numel, new_context_numel]);
            mask / mask_divisor
        });

        let head_outputs = match &output_mask {
            Some(mask) => Tensor::einsum("bhky,bhvy,bhkx,xy->bhvx", &[&key, &value, &query, mask]),
            None => Tensor::einsum("bhky,bhvy,bhkx->bhvx", &[&key, &value, &query]),
        };
        let head_outputs = head_outputs * 2f64.sqrt() / ((key_channels) as f64).sqrt();

        // dbg!(head_outputs.mean(Kind::Float));
        // dbg!(head_outputs.std(true));
        debug_assert!(!head_outputs.has_nan(), "NaN detected");

        // transform mask shape
        let output_mask = output_mask.map(|mask| {
            let shape: Vec<_> = new_input_shape
                .iter()
                .cloned()
                .chain(new_context_shape.iter().cloned())
                .collect();
            mask.view(&*shape)
        });

        // merge head outputs
        let output = Tensor::einsum("hvo,bhvx->box", &[merge_weight, &head_outputs]);
        // dbg!(output.mean(Kind::Float));
        // dbg!(output.std(true));
        debug_assert!(!output.has_nan(), "NaN detected");

        let output = {
            let output_shape: Vec<_> = vec![batch_size, output_channels]
                .into_iter()
                .chain(new_input_shape)
                .collect();
            output.view(&*output_shape)
        };

        debug_assert!(!output.has_nan(), "NaN detected");

        Ok((output, output_mask))
    }

    pub fn grad(&self) -> AttentionGrad {
        let Self {
            merge_weight,
            query_conv,
            key_conv,
            value_conv,
            ..
        } = self;

        AttentionGrad {
            merge_weight: merge_weight.grad(),
            query_conv: query_conv.grad(),
            key_conv: key_conv.grad(),
            value_conv: value_conv.grad(),
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct AttentionGrad {
    pub merge_weight: Tensor,
    pub query_conv: ConvNDGrad,
    pub key_conv: ConvNDGrad,
    pub value_conv: ConvNDGrad,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch_modules::Conv2DInit;

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
            input_conv: Conv2DInit {
                stride: [2, 2],
                ..Conv2DInit::new(ki)
            },
            context_conv: Conv2DInit {
                stride: [2, 2],
                ..Conv2DInit::new(kc)
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
