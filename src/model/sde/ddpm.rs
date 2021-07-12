use crate::{common::*, model::NormKind};
use tch_modules::{GroupNorm, GroupNormInit};

use attention_block::*;
use nin::*;
use variance_scaling::*;

mod variance_scaling {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Mode {
        FanIn,
        FanOut,
        FanAvg,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Distribution {
        Normal,
        Uniform,
    }

    #[derive(Debug)]
    pub struct VarianceScaling {
        pub scale: f64,
        pub mode: Mode,
        pub dist: Distribution,
        pub in_axis: usize,
        pub out_axis: usize,
    }

    impl VarianceScaling {
        pub fn init(&self, shape: &[i64], (kind, device): (Kind, Device)) -> Tensor {
            let Self {
                scale,
                mode,
                dist,
                in_axis,
                out_axis,
            } = *self;

            let in_c = shape[in_axis] as f64;
            let out_c = shape[out_axis] as f64;
            let nelem: i64 = shape.iter().cloned().product();
            let receptive_field_size = nelem as f64 / in_c / out_c;
            let fan_in = in_c * receptive_field_size;
            let fan_out = out_c * receptive_field_size;

            let denominator = match mode {
                Mode::FanIn => fan_in,
                Mode::FanOut => fan_out,
                Mode::FanAvg => (fan_in + fan_out) / 2.0,
            };
            let variance = scale / denominator;

            match dist {
                Distribution::Normal => Tensor::randn(shape, (kind, device)) * variance.sqrt(),
                Distribution::Uniform => {
                    (Tensor::rand(shape, (kind, device)) * 2.0 - 1.0) * (variance * 3.0).sqrt()
                }
            }
        }
    }
}

mod nin {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct NinInit {
        pub weight_init: nn::Init,
        pub bias_init: nn::Init,
    }

    impl Default for NinInit {
        fn default() -> Self {
            Self {
                weight_init: nn::Init::KaimingUniform,
                bias_init: nn::Init::Const(0.0),
            }
        }
    }

    impl NinInit {
        pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>, in_c: usize, out_c: usize) -> Nin {
            let Self {
                weight_init,
                bias_init,
            } = self;
            let path = path.borrow();
            let in_c = in_c as i64;
            let out_c = out_c as i64;

            let weight = path.var("weight", &[in_c, out_c], weight_init);
            let bias = path.var("bias", &[out_c], bias_init);

            Nin { weight, bias }
        }
    }

    #[derive(Debug)]
    pub struct Nin {
        weight: Tensor,
        bias: Tensor,
    }

    impl nn::Module for Nin {
        fn forward(&self, input: &Tensor) -> Tensor {
            Tensor::einsum("bchw,cd->bdhw", &[input, &self.weight]) + &self.bias
        }
    }
}

mod attention_block {
    use super::*;

    #[derive(Debug)]
    pub struct AttentionBlock {
        group_norm: GroupNorm,
        nin: [Nin; 4],
    }

    impl AttentionBlock {
        pub fn new<'a>(path: impl Borrow<nn::Path<'a>>, channels: usize) -> Self {
            let path = path.borrow();
            let init = NinInit::default();
            let group_norm = GroupNormInit {
                eps: r64(1e-6),
                ..Default::default()
            }
            .build(path / "norm", channels as i64, 32);
            let nin = [
                init.clone().build(path / "nin0", channels, channels),
                init.clone().build(path / "nin1", channels, channels),
                init.clone().build(path / "nin2", channels, channels),
                init.build(path / "nin3", channels, channels),
            ];

            Self { group_norm, nin }
        }
    }
    impl nn::Module for AttentionBlock {
        fn forward(&self, input: &Tensor) -> Tensor {
            let (bsize, in_c, in_h, in_w) = input.size4().unwrap();
            let Self {
                group_norm,
                nin: [nin0, nin1, nin2, nin3],
            } = self;

            let xs = group_norm.forward(input);
            let qs = nin0.forward(&xs);
            let ks = nin1.forward(&xs);
            let vs = nin2.forward(&xs);

            let ws = (Tensor::einsum("bchw,bcij->bhwij", &[qs, ks]) * (in_c as f64).powf(-0.5))
                .reshape(&[bsize, in_h, in_w, in_h * in_w])
                .softmax(3, Kind::Float)
                .reshape(&[bsize, in_h, in_w, in_h, in_w]);
            let bs = {
                let bs = Tensor::einsum("bhwij,bcij->bchw", &[ws, vs]);
                nin3.forward(&bs)
            };

            input + bs
        }
    }
}

mod resnet_block_ddpm {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ResnetBlockDDPMInit {
        pub dropout: f64,
        pub temb_dim: Option<usize>,
        pub conv_shortcut: bool,
    }

    impl ResnetBlockDDPMInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
        ) -> ResnetBlockDDPM {
            let path = path.borrow();
            let Self {
                dropout,
                temb_dim,
                conv_shortcut,
            } = self;

            todo!();
        }
    }

    impl Default for ResnetBlockDDPMInit {
        fn default() -> Self {
            Self {
                dropout: 0.1,
                temb_dim: None,
                conv_shortcut: false,
            }
        }
    }

    #[derive(Debug)]
    pub struct ResnetBlockDDPM {}

    impl ResnetBlockDDPM {}
}

#[derive(Debug)]
pub struct DDPM {
    act: Activation,
    norm: NormKind,
}

impl DDPM {
    pub fn forward_t(input: &Tensor, train: bool) -> Tensor {
        todo!();
    }
}
