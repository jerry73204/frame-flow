use crate::common::*;
use tch_modules::{ConvND, ConvNDGrad, ConvNDInit, GroupNorm, GroupNormInit};

pub use attention::*;
pub use self_attention::*;

mod self_attention {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct SelfAttentionInit<const DIM: usize> {
        pub ksize: usize,
        pub n_heads: usize,
        pub key_c: usize,
        pub value_c: usize,
        pub bias: bool,
    }

    impl<const DIM: usize> SelfAttentionInit<DIM> {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
        ) -> Result<SelfAttention> {
            let Self {
                ksize,
                n_heads,
                key_c,
                value_c,
                bias,
            } = self;
            Ok(SelfAttention {
                attn: AttentionInit::<DIM> {
                    ksize,
                    n_heads,
                    key_c,
                    value_c,
                    bias,
                }
                .build(path, in_c, in_c, out_c)?,
            })
        }
    }

    impl SelfAttention {
        pub fn f_forward<'a>(&self, input: &Tensor) -> Result<Tensor> {
            self.attn.forward(input, input)
        }
    }

    impl nn::Module for SelfAttention {
        fn forward<'a>(&self, input: &Tensor) -> Tensor {
            self.f_forward(input).unwrap()
        }
    }

    #[derive(Debug)]
    pub struct SelfAttention {
        attn: Attention,
    }
}

mod attention {
    use super::*;

    pub type AttentionInit1 = AttentionInit<1>;
    pub type AttentionInit2 = AttentionInit<2>;
    pub type AttentionInit3 = AttentionInit<3>;

    #[derive(Debug, Clone)]
    pub struct AttentionInit<const DIM: usize> {
        pub ksize: usize,
        pub n_heads: usize,
        pub key_c: usize,
        pub value_c: usize,
        pub bias: bool,
    }

    impl<const DIM: usize> AttentionInit<DIM> {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            ctx_c: usize,
            out_c: usize,
        ) -> Result<Attention> {
            let path = path.borrow();
            let AttentionInit {
                ksize,
                n_heads,
                key_c,
                value_c,
                bias,
            } = self;

            let query_conv = ConvNDInit::<[usize; DIM]> {
                bias,
                ..ConvNDInit::<[usize; DIM]>::new(ksize)
            }
            .build(path / "query_conv", in_c, n_heads * key_c)?;
            let key_conv = ConvNDInit::<[usize; DIM]> {
                bias,
                ..ConvNDInit::<[usize; DIM]>::new(ksize)
            }
            .build(path / "key_conv", ctx_c, n_heads * key_c)?;
            let value_conv = ConvNDInit::<[usize; DIM]> {
                bias,
                ..ConvNDInit::<[usize; DIM]>::new(ksize)
            }
            .build(path / "value_conv", ctx_c, n_heads * value_c)?;
            let output_conv = ConvNDInit::<[usize; DIM]>::new(ksize).build(
                path / "output_conv",
                n_heads * value_c,
                out_c,
            )?;

            let input_norm =
                GroupNormInit::default().build(path / "input_norm", in_c as i64, in_c as i64);
            let context_norm =
                GroupNormInit::default().build(path / "context_norm", ctx_c as i64, ctx_c as i64);

            Ok(Attention {
                n_heads: n_heads as i64,
                key_c: key_c as i64,
                value_c: value_c as i64,
                query_conv,
                key_conv,
                value_conv,
                output_conv,
                input_norm,
                context_norm,
            })
        }
    }

    #[derive(Debug)]
    pub struct Attention {
        n_heads: i64,
        key_c: i64,
        value_c: i64,
        query_conv: ConvND,
        key_conv: ConvND,
        value_conv: ConvND,
        output_conv: ConvND,
        input_norm: GroupNorm,
        context_norm: GroupNorm,
    }

    impl Attention {
        pub fn forward<'a>(&self, input: &Tensor, context: &Tensor) -> Result<Tensor> {
            let bsize = input.size()[0];
            ensure!(context.size()[0] == bsize, "batch size mismatch");

            // dbg!(input.mean(Kind::Float), context.mean(Kind::Float));
            // dbg!(input.std(true), context.std(true));

            let Self {
                n_heads,
                key_c,
                value_c,
                ref query_conv,
                ref key_conv,
                ref value_conv,
                ref output_conv,
                ref input_norm,
                ref context_norm,
            } = *self;

            let input_shape = &input.size()[2..];
            // let context_shape = &context.size()[2..];
            // let input_numel: i64 = input_shape.iter().product();

            // norm
            // let input = input_norm.forward(input);
            // let context = context_norm.forward(context);

            let input = <_ as nn::Module>::forward(input_norm, input);
            let context = <_ as nn::Module>::forward(context_norm, context);

            // convolutions
            let query = query_conv.forward(&input);
            let key = key_conv.forward(&context);
            let value = value_conv.forward(&context);

            // flatten
            let query = query.view([bsize, n_heads, key_c, -1]);
            let key = key.view([bsize, n_heads, key_c, -1]);
            let value = value.view([bsize, n_heads, value_c, -1]);

            debug_assert!(!query.has_nan(), "NaN detected");
            debug_assert!(!key.has_nan(), "NaN detected");
            debug_assert!(!value.has_nan(), "NaN detected");

            // sum over attention
            let attn = Tensor::einsum("bhkx,bhky->bhxy", &[query, key]).softmax(3, Kind::Float);
            let sum = Tensor::einsum("bhxy,bhvy->bhvx", &[attn, value]);
            let sum = {
                let new_shape: Vec<_> =
                    chain!(vec![bsize, n_heads * value_c], input_shape.iter().cloned()).collect();
                sum.reshape(&*new_shape)
            };
            debug_assert!(!sum.has_nan(), "NaN detected");

            // merge head outputs
            let output = output_conv.forward(&sum);
            debug_assert!(!output.has_nan(), "NaN detected");

            Ok(output)
        }

        pub fn grad(&self) -> AttentionGrad {
            let Self {
                query_conv,
                key_conv,
                value_conv,
                output_conv,
                ..
            } = self;

            AttentionGrad {
                query_conv: query_conv.grad(),
                key_conv: key_conv.grad(),
                value_conv: value_conv.grad(),
                output_conv: output_conv.grad(),
            }
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct AttentionGrad {
        pub query_conv: ConvNDGrad,
        pub key_conv: ConvNDGrad,
        pub value_conv: ConvNDGrad,
        pub output_conv: ConvNDGrad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_test() -> Result<()> {
        let b: usize = 11;
        let cx: usize = 3;
        let cc: usize = 5;
        let hx: usize = 40;
        let wx: usize = 50;
        let hc: usize = 30;
        let wc: usize = 70;
        let o: usize = 2;

        let input = Tensor::randn(&[b as i64, cx as i64, hx as i64, wx as i64], FLOAT_CPU);
        let context = Tensor::randn(&[b as i64, cc as i64, hc as i64, wc as i64], FLOAT_CPU);

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let attention = AttentionInit::<2> {
            ksize: 3,
            n_heads: 8,
            key_c: 4,
            value_c: 6,
            bias: true,
        }
        .build(&root, cx, cc, o)?;

        let output = attention.forward(&input, &context)?;
        assert_eq!(
            output.size(),
            vec![b as i64, o as i64, hx as i64, wx as i64]
        );

        Ok(())
    }
}
