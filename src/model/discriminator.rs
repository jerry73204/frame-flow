use super::misc::NormKind;
use crate::common::*;
use nn::Module;

use tch_modules::{ConvBn, ConvBnGrad, ConvBnInitDyn, ConvND, ConvNDGrad, ConvNDInitDyn};

pub use custom::*;
pub use discriminator::*;
pub use n_layers::*;
pub use pixel::*;

mod discriminator {
    use super::*;

    #[derive(Debug)]
    pub enum Discriminator {
        NLayers(NLayerDiscriminator),
        Pixel(PixelDiscriminator),
    }

    impl nn::ModuleT for Discriminator {
        fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
            match self {
                Discriminator::NLayers(model) => model.forward_t(input, train),
                Discriminator::Pixel(model) => model.forward_t(input, train),
            }
        }
    }

    impl From<PixelDiscriminator> for Discriminator {
        fn from(v: PixelDiscriminator) -> Self {
            Self::Pixel(v)
        }
    }

    impl From<NLayerDiscriminator> for Discriminator {
        fn from(v: NLayerDiscriminator) -> Self {
            Self::NLayers(v)
        }
    }
}

mod n_layers {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct NLayerDiscriminatorInit<const N_BLOCKS: usize> {
        pub norm_kind: NormKind,
    }

    impl<const N_BLOCKS: usize> NLayerDiscriminatorInit<N_BLOCKS> {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            inner_c: usize,
        ) -> NLayerDiscriminator {
            let path = path.borrow();
            let Self { norm_kind } = self;
            let bias = norm_kind == NormKind::InstanceNorm;
            let in_c = in_c as i64;
            let inner_c = inner_c as i64;
            let ksize = 3;
            let padding = 1;

            let seq = {
                let path = path / "first_block";
                nn::seq_t()
                    .add(nn::conv2d(
                        &path / "conv",
                        in_c,
                        inner_c,
                        ksize,
                        nn::ConvConfig {
                            stride: 2,
                            padding,
                            ..Default::default()
                        },
                    ))
                    .add_fn(|xs| xs.leaky_relu())
            };

            let seq = (0..N_BLOCKS)
                .scan(1, |saved_mult, index| {
                    let prev_mult = *saved_mult;
                    let curr_mult = 2i64.pow(index.min(3) as u32);
                    *saved_mult = curr_mult;
                    Some((index, prev_mult, curr_mult))
                })
                .fold(seq, |seq, (index, prev_mult, curr_mult)| {
                    let path = path / format!("block_{}", index);

                    seq.add(nn::conv2d(
                        &path / "conv",
                        inner_c * prev_mult,
                        inner_c * curr_mult,
                        ksize,
                        nn::ConvConfig {
                            stride: 2,
                            padding,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c * curr_mult))
                    .add_fn(|xs| xs.leaky_relu())
                });

            let seq = {
                let path = path / format!("block_{}", N_BLOCKS);
                let prev_mult = 2i64.pow((N_BLOCKS - 1).min(3) as u32);
                let curr_mult = 2i64.pow(N_BLOCKS.min(3) as u32);

                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c * prev_mult,
                    inner_c * curr_mult,
                    ksize,
                    nn::ConvConfig {
                        stride: 1,
                        padding,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c * curr_mult))
                .add_fn(|xs| xs.leaky_relu())
            };

            let seq = {
                let path = path / "last_block";
                let mult = 2i64.pow(N_BLOCKS.min(3) as u32);

                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c * mult,
                    1,
                    ksize,
                    nn::ConvConfig {
                        stride: 1,
                        padding,
                        ..Default::default()
                    },
                ))
                .add_fn(|xs| {
                    let batch_size = xs.size()[0];
                    xs.view([batch_size])
                })
            };

            NLayerDiscriminator { seq }
        }
    }

    impl<const N_BLOCKS: usize> Default for NLayerDiscriminatorInit<N_BLOCKS> {
        fn default() -> Self {
            Self {
                norm_kind: NormKind::BatchNorm,
            }
        }
    }

    #[derive(Debug)]
    pub struct NLayerDiscriminator {
        seq: nn::SequentialT,
    }

    impl NLayerDiscriminator {
        pub fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
            self.seq.forward_t(input, train)
        }
    }
}

mod pixel {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct PixelDiscriminatorInit {
        pub norm_kind: NormKind,
    }

    impl PixelDiscriminatorInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            inner_c: usize,
        ) -> PixelDiscriminator {
            let path = path.borrow();
            let Self { norm_kind } = self;
            let bias = norm_kind == NormKind::InstanceNorm;
            let in_c = in_c as i64;
            let inner_c = inner_c as i64;

            let seq = nn::seq_t()
                .add(nn::conv2d(
                    path / "conv1",
                    in_c,
                    inner_c,
                    1,
                    nn::ConvConfig {
                        stride: 1,
                        padding: 0,
                        ..Default::default()
                    },
                ))
                .add_fn(|xs| xs.leaky_relu())
                .add(nn::conv2d(
                    path / "conv2",
                    inner_c,
                    inner_c * 2,
                    1,
                    nn::ConvConfig {
                        stride: 1,
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(path / "norm1", inner_c * 2))
                .add_fn(|xs| xs.leaky_relu())
                .add(nn::conv2d(
                    path / "conv2",
                    inner_c,
                    inner_c * 2,
                    1,
                    nn::ConvConfig {
                        stride: 1,
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ));

            PixelDiscriminator { seq }
        }
    }

    impl Default for PixelDiscriminatorInit {
        fn default() -> Self {
            Self {
                norm_kind: NormKind::BatchNorm,
            }
        }
    }

    #[derive(Debug)]
    pub struct PixelDiscriminator {
        seq: nn::SequentialT,
    }

    impl nn::ModuleT for PixelDiscriminator {
        fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
            self.seq.forward_t(input, train)
        }
    }
}

mod custom {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct CustomDiscriminatorInit<const DEPTH: usize> {
        pub ndims: usize,
        pub ksize: usize,
        pub input_channels: usize,
        pub channels: [usize; DEPTH],
        pub strides: [usize; DEPTH],
    }

    impl<const DEPTH: usize> CustomDiscriminatorInit<DEPTH> {
        pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<CustomDiscriminator> {
            ensure!(DEPTH > 0, "zero depth is not allowed");

            let path = path.borrow();
            let Self {
                ndims,
                ksize,
                input_channels,
                channels,
                strides,
            } = self;
            ensure!(
                (1..=3).contains(&ndims),
                "ndims must be one of 1, 2, 3, but get ndims = {}",
                ndims
            );

            let last_channels = *channels.last().unwrap();

            let tuples: Vec<_> = izip!(
                iter::once(input_channels).chain(array::IntoIter::new(channels)),
                array::IntoIter::new(channels),
                array::IntoIter::new(strides),
            )
            .enumerate()
            .map(|(index, (in_c, out_c, stride))| -> Result<_> {
                let conv = ConvBnInitDyn::new(ndims, ksize).build(
                    path / format!("conv_{}", index),
                    in_c,
                    out_c,
                )?;

                let down_sample = ConvNDInitDyn {
                    stride: vec![stride; ndims],
                    ..ConvNDInitDyn::new(ndims, 1)
                }
                .build(path / format!("down_sample_{}", index), out_c, out_c)?;

                Ok((conv, down_sample))
            })
            .try_collect()?;

            let (convs, down_samples) = tuples.into_iter().unzip_n_vec();

            let linear = nn::linear(path / "linear", last_channels as i64, 1, Default::default());

            Ok(CustomDiscriminator {
                ndims,
                convs,
                down_samples,
                linear,
            })
        }
    }

    #[derive(Debug)]
    pub struct CustomDiscriminator {
        ndims: usize,
        convs: Vec<ConvBn>,
        down_samples: Vec<ConvND>,
        linear: nn::Linear,
    }

    impl CustomDiscriminator {
        pub fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
            let Self {
                ndims,
                ref convs,
                ref down_samples,
                ref linear,
            } = *self;

            let xs = izip!(convs, down_samples).try_fold(
                input.shallow_clone(),
                |xs, (conv, down_sample)| -> Result<_> {
                    let xs = conv.forward_t(&xs, train);
                    let xs = down_sample.forward(&xs);
                    Ok(xs)
                },
            )?;

            let xs = match ndims {
                1 => xs.adaptive_avg_pool1d(&[1]),
                2 => xs.adaptive_avg_pool2d(&[1, 1]),
                3 => xs.adaptive_avg_pool3d(&[1, 1, 1]),
                _ => unreachable!(),
            };
            let xs = xs.view(&xs.size()[0..2]);
            let xs = linear.forward(&xs);

            Ok(xs)
        }

        pub fn clamp_bn_var(&mut self) {
            let Self { convs, .. } = self;

            convs.iter_mut().for_each(|conv| {
                conv.clamp_bn_var();
            });
        }

        pub fn denormalize_bn(&mut self) {
            let Self { convs, .. } = self;

            convs.iter_mut().for_each(|conv| {
                conv.denormalize_bn();
            });
        }

        pub fn grad(&self) -> CustomDiscriminatorGrad {
            let Self {
                convs,
                down_samples,
                linear: nn::Linear { ws, bs, .. },
                ..
            } = self;

            CustomDiscriminatorGrad {
                convs: convs.iter().map(|conv| conv.grad()).collect(),
                down_samples: down_samples.iter().map(|down| down.grad()).collect(),
                linear: LinearGrad {
                    ws: ws.grad(),
                    bs: bs.grad(),
                },
            }
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct LinearGrad {
        pub ws: Tensor,
        pub bs: Tensor,
    }

    #[derive(Debug, TensorLike)]
    pub struct CustomDiscriminatorGrad {
        pub convs: Vec<ConvBnGrad>,
        pub down_samples: Vec<ConvNDGrad>,
        pub linear: LinearGrad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminator_test() -> Result<()> {
        let bs = 2;
        let cx = 3;
        let hx = 11;
        let wx = 13;

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();

        let discriminator = CustomDiscriminatorInit {
            ndims: 2,
            ksize: 3,
            input_channels: cx,
            channels: [16, 32],
            strides: [2, 2],
        }
        .build(&root)?;

        let input = Tensor::rand(&[bs, cx as i64, hx, wx], FLOAT_CPU);
        let output = discriminator.forward_t(&input, true)?;

        ensure!(output.size() == vec![bs, 1], "incorrect output shape");

        Ok(())
    }
}
