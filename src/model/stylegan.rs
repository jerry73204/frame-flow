use crate::common::*;

pub use activation::*;
pub use conv_2d_resample::*;
pub use discriminator::*;
pub use discriminator_block::*;
pub use discriminator_epilogue::*;
pub use fir_filter::*;
pub use generator::*;
pub use mapping_network::*;
pub use minibatch_std_layer::*;
pub use misc::*;
pub use modulated_conv2d::*;
pub use stylegan_conv_2d::*;
pub use stylegan_linear::*;
pub use synthesis_block::*;
pub use synthesis_layer::*;
pub use synthesis_network::*;
pub use to_rgb_layer::*;
pub use up_fir_down_2d::*;

type TensorIter = Box<dyn Iterator<Item = Tensor> + Send>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    Orig,
    Skip,
    Resnet,
}

mod fir_filter {
    use super::*;

    #[derive(Debug)]
    pub struct FirFilter(Tensor);

    impl FirFilter {
        pub fn size(&self) -> (i64, i64) {
            self.0.size2().unwrap()
        }
    }

    impl Clone for FirFilter {
        fn clone(&self) -> Self {
            Self(self.0.detach().copy())
        }
    }

    impl Default for FirFilter {
        fn default() -> Self {
            Self(Tensor::ones(&[1, 1], (Kind::Float, Device::Cpu)))
        }
    }

    impl TryFrom<Tensor> for FirFilter {
        type Error = Error;

        fn try_from(from: Tensor) -> Result<Self, Self::Error> {
            ensure!(from.kind() == Kind::Float && from.numel() > 0);
            let from = from.detach().copy().set_requires_grad(false);
            let tensor = match from.dim() {
                0 => from.view([1, 1]),
                1 => from.outer(&from),
                2 => from,
                dim => bail!("filter dimension must be one of 0, 1, 2, but get {}", dim),
            };
            Ok(Self(tensor))
        }
    }

    impl From<f64> for FirFilter {
        fn from(from: f64) -> Self {
            Self(Tensor::from(from).view([1, 1]))
        }
    }

    impl<const N: usize> From<[f64; N]> for FirFilter {
        fn from(from: [f64; N]) -> Self {
            let tensor = Tensor::of_slice(from.as_ref());
            Self(tensor)
        }
    }

    impl<const R: usize, const C: usize> From<[[f64; C]; R]> for FirFilter {
        fn from(from: [[f64; C]; R]) -> Self {
            let tensor: Tensor = from.into_cv();
            Self(tensor)
        }
    }

    #[derive(Debug, Clone)]
    pub struct FirFilterInit {
        pub filter: FirFilter,
        pub normalize: bool,
        pub flip: bool,
        pub gain: f64,
    }

    impl FirFilterInit {
        pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<Tensor> {
            let path = path.borrow();
            let Self {
                filter,
                normalize,
                flip,
                gain,
            } = self;
            ensure!(gain > 0.0);

            let filter = {
                let weights = filter.0;
                let mut filter = path.zeros_no_train("fir_filter", &weights.size());
                filter.copy_(&weights);
                filter
            };
            let filter = if normalize {
                &filter / &filter.sum(Kind::Float)
            } else {
                filter
            };
            let filter = if flip { filter.flip(&[0, 1]) } else { filter };
            let filter = filter * gain;

            Ok(filter)
        }
    }

    impl Default for FirFilterInit {
        fn default() -> Self {
            Self {
                filter: Default::default(),
                normalize: true,
                flip: false,
                gain: 1.0,
            }
        }
    }
}

mod up_fir_down_2d {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct UpFirDn2DInit {
        pub filter: FirFilterInit,
        pub up: usize,
        pub down: usize,
        pub padding: [i64; 4],
    }

    impl Default for UpFirDn2DInit {
        fn default() -> Self {
            Self {
                filter: Default::default(),
                up: 1,
                down: 1,
                padding: [0; 4],
            }
        }
    }

    impl UpFirDn2DInit {
        pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<UpFirDn2D> {
            let path = path.borrow();

            let Self {
                filter,
                up,
                down,
                padding,
            } = self;

            ensure!(up > 0);
            ensure!(down > 0);
            let up = up as i64;
            let down = down as i64;
            let weight = filter.build(path)?;

            Ok(UpFirDn2D {
                weight,
                up: [up; 2],
                down: [down; 2],
                padding,
            })
        }
    }

    #[derive(Debug)]
    pub struct UpFirDn2D {
        weight: Tensor,
        up: [i64; 2],
        down: [i64; 2],
        padding: [i64; 4],
    }

    impl nn::Module for UpFirDn2D {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let Self {
                ref weight,
                up: [up_y, up_x],
                down: [down_y, down_x],
                padding: [pad_l, pad_r, pad_t, pad_b],
            } = *self;

            let (bsize, in_c, in_h, in_w) = xs.size4().unwrap();

            // up sample
            let xs = xs
                .reshape(&[bsize, in_c, in_h, 1, in_w, 1])
                .constant_pad_nd(&[0, up_x - 1, 0, 0, 0, up_y - 1])
                .reshape(&[bsize, in_c, in_h * up_y, in_w * up_x]);

            // compute padding
            let pos_neg = |num: i64| {
                if num.is_negative() {
                    (0, -num)
                } else {
                    (num, 0)
                }
            };

            let (pad_l_pos, pad_l_neg) = pos_neg(pad_l);
            let (pad_r_pos, pad_r_neg) = pos_neg(pad_r);
            let (pad_t_pos, pad_t_neg) = pos_neg(pad_t);
            let (pad_b_pos, pad_b_neg) = pos_neg(pad_b);

            // pad
            let xs = xs.constant_pad_nd(&[pad_l_pos, pad_r_pos, pad_t_pos, pad_b_pos]);

            // crop
            let xs = {
                let (_, _, xh, xw) = xs.size4().unwrap();
                xs.i((
                    ..,
                    ..,
                    pad_t_neg..(xh - pad_b_neg),
                    pad_l_neg..(xw - pad_r_neg),
                ))
            };

            // convolution
            let weight = weight.unsqueeze(0).unsqueeze(0).repeat(&[in_c, 1, 1, 1]);

            let xs = xs.convolution::<Tensor>(
                &weight,
                None,    // bias
                &[1, 1], // stride
                &[0, 0], // padding
                &[0, 0], // dilation
                false,   // transposed
                &[0, 0], // output_padding
                in_c,    // num_groups
            );

            // down sample
            xs.slice(2, None, None, down_y).slice(3, None, None, down_x)
        }
    }
}

mod conv_2d_resample {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Conv2DResampleInit {
        pub filter: FirFilterInit,
        pub padding: [i64; 4],
        pub up: usize,
        pub down: usize,
        pub groups: usize,
        pub flip_weight: bool,
        pub ws_init: nn::Init,
    }

    impl Default for Conv2DResampleInit {
        fn default() -> Self {
            Self {
                filter: Default::default(),
                padding: default_padding(),
                up: default_up(),
                down: default_down(),
                groups: 1,
                flip_weight: default_flip_weight(),
                ws_init: nn::Init::KaimingUniform,
            }
        }
    }

    impl Conv2DResampleInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            ksize: usize,
        ) -> Result<Conv2DResample> {
            let Self {
                filter,
                padding,
                up,
                down,
                groups,
                flip_weight,
                ws_init,
            } = self;
            ensure!(groups >= 1);
            let path = path.borrow();
            let (fh, fw) = filter.filter.size();
            let out_c = out_c as i64;
            let in_c = in_c as i64;
            let ksize = ksize as i64;
            let groups = groups as i64;

            let conv_padding = {
                let up = up as i64;
                let down = down as i64;

                let padding = if up > 1 {
                    let [pad_l, pad_r, pad_t, pad_b] = padding;
                    [
                        pad_l + (fw + up - 1) / 2,
                        pad_r + (fw - up) / 2,
                        pad_t + (fh + up - 1) / 2,
                        pad_b + (fh - up) / 2,
                    ]
                } else {
                    padding
                };

                if down > 1 {
                    let [pad_l, pad_r, pad_t, pad_b] = padding;
                    [
                        pad_l + (fw + down - 1) / 2,
                        pad_r + (fw - down) / 2,
                        pad_t + (fh + down - 1) / 2,
                        pad_b + (fh - down) / 2,
                    ]
                } else {
                    padding
                }
            };

            let weight = {
                let weight =
                    path.sub("mid_conv")
                        .var("weight", &[out_c, in_c, ksize, ksize], ws_init);

                if flip_weight {
                    weight.flip(&[2, 3])
                } else {
                    weight
                }
            };

            let up_conv = UpFirDn2DInit {
                filter: filter.clone(),
                up,
                down: 1,
                padding: conv_padding,
            }
            .build(path / "up_conv")?;

            let down_conv = UpFirDn2DInit {
                filter,
                up: 1,
                down,
                padding: [0, 0, 0, 0],
            }
            .build(path / "down_conv")?;

            Ok(Conv2DResample {
                up_conv,
                down_conv,
                weight,
                groups,
            })
        }
    }

    #[derive(Debug)]
    pub struct Conv2DResample {
        pub(super) up_conv: UpFirDn2D,
        pub(super) down_conv: UpFirDn2D,
        pub(super) weight: Tensor,
        groups: i64,
    }

    impl nn::Module for Conv2DResample {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let Self {
                ref up_conv,
                ref down_conv,
                ref weight,
                groups,
            } = *self;
            conv2d_resample(xs, up_conv, down_conv, weight, groups)
        }
    }

    impl Conv2DResample {
        pub fn set_trainable(&self, trainable: bool) {
            let _ = self.weight.set_requires_grad(trainable);
        }
    }

    pub fn conv2d_resample(
        xs: &Tensor,
        up_conv: &UpFirDn2D,
        down_conv: &UpFirDn2D,
        weight: &Tensor,
        groups: i64,
    ) -> Tensor {
        let xs = up_conv.forward(xs);
        let xs = xs.convolution::<Tensor>(
            weight,
            None,
            &[1, 1, 1, 1],
            &[0, 0, 0, 0],
            &[1, 1, 1, 1],
            false,
            &[0, 0, 0, 0],
            groups,
        );
        down_conv.forward(&xs)
    }
}

mod modulated_conv2d {
    use super::*;

    pub struct ModulatedConv2DInit {
        pub up: usize,
        pub down: usize,
        pub padding: [i64; 4],
        pub flip_weight: bool,
        pub demodulate: bool,
        pub filter: FirFilterInit,
        pub ws_init: nn::Init,
    }

    impl Default for ModulatedConv2DInit {
        fn default() -> Self {
            Self {
                up: default_up(),
                down: default_down(),
                padding: default_padding(),
                flip_weight: default_flip_weight(),
                demodulate: true,
                filter: Default::default(),
                ws_init: nn::Init::KaimingUniform,
            }
        }
    }

    pub struct ModulatedConv2D {
        conv: Conv2DResample,
        demodulate: bool,
    }

    impl ModulatedConv2DInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            ksize: usize,
        ) -> Result<ModulatedConv2D> {
            let path = path.borrow();
            let Self {
                up,
                down,
                filter,
                padding,
                demodulate,
                flip_weight,
                ws_init,
            } = self;
            ensure!(ksize > 0);

            let conv = Conv2DResampleInit {
                up,
                down,
                padding,
                ws_init,
                filter,
                flip_weight,
                groups: 1,
            }
            .build(path, in_c, out_c, ksize)?;
            let output = ModulatedConv2D { conv, demodulate };

            Ok(output)
        }
    }

    impl ModulatedConv2D {
        pub fn f_forward(
            &self,
            xs: &Tensor,
            styles: &Tensor,
            noise: Option<&Tensor>,
        ) -> Result<Tensor> {
            let (bsize, in_c, _, _) = xs.size4()?;
            ensure!(styles.size2()? == (bsize, in_c));

            let Self {
                ref conv,
                demodulate,
            } = *self;

            let dcoefs = if demodulate {
                let weight = conv.weight.unsqueeze(0); // [1, out_c, in_c, kh, kw]
                let weight = weight * styles.reshape(&[bsize, 1, -1, 1, 1]); // [bsize, out_c, in_c, kh, kw]
                let dcoefs = (weight
                    .square()
                    .sum_dim_intlist(&[2, 3, 4], false, Kind::Float)
                    + 1e-8)
                    .rsqrt(); // [bsize, out_c]
                Some(dcoefs)
            } else {
                None
            };

            let xs = xs * styles.reshape(&[bsize, -1, 1, 1]);
            let xs = conv.forward(&xs);
            let xs = match dcoefs {
                Some(dcoefs) => xs * dcoefs,
                None => xs,
            };
            let xs = match noise {
                Some(noise) => xs + noise,
                None => xs,
            };

            Ok(xs)
        }
    }
}

mod synthesis_layer {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum NoiseMode {
        Random,
        Const,
        None,
    }

    #[derive(Debug, Clone)]
    pub struct SynthesisLayerInit {
        pub ksize: usize,
        pub up: usize,
        pub activation: FixedActivationInit,
        pub noise_mode: NoiseMode,
        pub filter: FirFilterInit,
    }

    impl Default for SynthesisLayerInit {
        fn default() -> Self {
            Self {
                ksize: 3,
                up: 1,
                activation: Default::default(),
                noise_mode: NoiseMode::Random,
                filter: Default::default(),
            }
        }
    }

    impl SynthesisLayerInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            w_dim: usize,
            resol: usize,
        ) -> Result<SynthesisLayer> {
            let path = path.borrow();
            let device = path.device();
            let Self {
                ksize,
                up,
                activation,
                noise_mode,
                filter,
            } = self;
            let resol = resol as i64;

            let conv = ModulatedConv2DInit {
                up,
                padding: [ksize as i64 / 2; 4],
                flip_weight: up == 1,
                filter,
                ..Default::default()
            }
            .build(path, in_c, out_c, ksize)?;
            let bias = path.zeros("bias", &[out_c as i64]);
            let noise_strength = path.zeros("noise_strength", &[]);
            let linear = nn::linear(
                path / "linear",
                in_c as i64,
                w_dim as i64,
                Default::default(),
            );
            let activation = activation.build()?;

            let noise_fn: Box<dyn Fn(i64) -> Option<Tensor> + Send> = match noise_mode {
                NoiseMode::Random => Box::new(move |bsize: i64| {
                    Some(Tensor::randn(
                        &[bsize, 1, resol, resol],
                        (Kind::Float, device),
                    ))
                }),
                NoiseMode::Const => {
                    let noise_const = path.randn("noise_const", &[resol, resol], 0.0, 1.0);
                    Box::new(move |_| Some(noise_const.shallow_clone()))
                }
                NoiseMode::None => Box::new(|_| None),
            };

            let forward_fn = Box::new(move |xs: &Tensor, styles: &Tensor| -> Result<Tensor> {
                let in_resol = resol / up as i64;
                let bsize = {
                    let (bsize, _, in_h, in_w) = xs.size4()?;
                    ensure!((in_h, in_w) == (in_resol, in_resol));
                    bsize
                };

                let styles: Tensor = linear.forward(styles);
                let noise = noise_fn(bsize).map(|noise| noise * &noise_strength);

                let xs = conv.f_forward(xs, &styles, noise.as_ref())?;
                let xs = xs + bias.view([1, -1, 1, 1]);
                let xs = activation.forward(&xs);

                Ok(xs)
            });

            Ok(SynthesisLayer { forward_fn })
        }
    }

    pub struct SynthesisLayer {
        forward_fn: Box<dyn Fn(&Tensor, &Tensor) -> Result<Tensor> + Send>,
    }

    impl SynthesisLayer {
        pub fn f_forward(&self, xs: &Tensor, styles: &Tensor) -> Result<Tensor> {
            (self.forward_fn)(xs, styles)
        }
    }
}

mod synthesis_block {
    use super::*;

    type InputFn = Box<dyn Fn(Option<&Tensor>, i64) -> Result<Tensor> + Send>;
    type MainFn = Box<dyn Fn(&Tensor, &mut TensorIter) -> Result<Tensor> + Send>;
    type ImageFn =
        Box<dyn Fn(&Tensor, Option<&Tensor>, &mut TensorIter) -> Result<Option<Tensor>> + Send>;
    type ForwardFn = Box<
        dyn Fn(
                Option<&Tensor>,
                Option<&Tensor>,
                &mut TensorIter,
                i64,
            ) -> Result<(Tensor, Option<Tensor>)>
            + Send,
    >;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Kind {
        Orig,
        Skip,
        Resnet,
    }

    #[derive(Debug, Clone)]
    pub struct SynthesisBlockInit {
        pub kind: ModelKind,
        pub filter: FirFilter,
        pub clamp: Option<f64>,
    }

    impl Default for SynthesisBlockInit {
        fn default() -> Self {
            Self {
                kind: ModelKind::Skip,
                filter: Default::default(),
                clamp: None,
            }
        }
    }

    impl SynthesisBlockInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            w_dim: usize,
            resol: usize,
            image_c: usize,
            is_last: bool,
        ) -> Result<SynthesisBlock> {
            let path = path.borrow();
            let Self {
                kind,
                filter,
                clamp,
            } = self;

            fn styles_err() -> Error {
                format_err!("invalid styles input length")
            }

            let input_fn: InputFn = if in_c == 0 {
                let out_c = out_c as i64;
                let resol = resol as i64;
                let r#const = path.randn("const", &[out_c, resol, resol], 0.0, 1.0);

                Box::new(move |_xs: Option<&Tensor>, bsize: i64| -> Result<_> {
                    let xs = r#const
                        .shallow_clone()
                        .unsqueeze(0)
                        .repeat(&[bsize, 1, 1, 1]);
                    Ok(xs)
                })
            } else {
                let in_c = in_c as i64;
                let resol = resol as i64;

                Box::new(move |xs: Option<&Tensor>, _bsize| -> Result<_> {
                    let xs = xs.ok_or_else(|| format_err!("expect xs to be Some, but get None"))?;
                    let (_, in_c_, in_h, in_w) = xs.size4()?;
                    ensure!((in_c_, in_h, in_w) == (in_c, resol / 2, resol / 2));
                    Ok(xs.shallow_clone())
                })
            };

            let main_fn: MainFn = {
                let conv1 = SynthesisLayerInit {
                    activation: FixedActivationInit {
                        clamp,
                        ..Default::default()
                    },
                    ..Default::default()
                }
                .build(path / "conv1", out_c, out_c, w_dim, resol)?;

                if in_c == 0 {
                    Box::new(move |xs: &Tensor, styles: &mut TensorIter| -> Result<_> {
                        let xs = conv1.f_forward(xs, &styles.next().ok_or_else(styles_err)?)?;
                        Ok(xs)
                    })
                } else {
                    let conv0 = SynthesisLayerInit {
                        up: 2,
                        activation: FixedActivationInit {
                            clamp,
                            ..Default::default()
                        },
                        filter: FirFilterInit {
                            filter: filter.clone(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                    .build(path / "conv0", in_c, out_c, w_dim, resol)?;

                    match kind {
                        ModelKind::Resnet => {
                            let skip = StyleGanConv2DInit {
                                up: 2,
                                bs_init: None,
                                activation: FixedActivationInit {
                                    gain: 0.5.sqrt(),
                                    ..Default::default()
                                },
                                filter: FirFilterInit {
                                    filter,
                                    ..Default::default()
                                },
                                ..Default::default()
                            }
                            .build(path / "skip", in_c, out_c, 1)?;

                            Box::new(move |xs: &Tensor, styles: &mut TensorIter| -> Result<_> {
                                let ys = skip.forward(xs);
                                let xs =
                                    conv0.f_forward(xs, &styles.next().ok_or_else(styles_err)?)?;
                                let xs =
                                    conv1.f_forward(&xs, &styles.next().ok_or_else(styles_err)?)?;
                                let xs = xs + ys;
                                Ok(xs)
                            })
                        }
                        ModelKind::Orig | ModelKind::Skip => {
                            Box::new(move |xs: &Tensor, styles: &mut TensorIter| -> Result<_> {
                                let xs =
                                    conv0.f_forward(xs, &styles.next().ok_or_else(styles_err)?)?;
                                let xs =
                                    conv1.f_forward(&xs, &styles.next().ok_or_else(styles_err)?)?;
                                Ok(xs)
                            })
                        }
                    }
                }
            };

            let image_fn: ImageFn = {
                let upfirdn2d_fn = {
                    let image_c = image_c as i64;
                    let resol = resol as i64;
                    let upfirdn2d = UpFirDn2DInit::default().build(path / "upfirdn2d")?;

                    move |image: Option<&Tensor>| -> Result<_> {
                        image
                            .map(|image| -> Result<_> {
                                let (_bsize, img_c_, img_h, img_w) = image.size4()?;
                                ensure!((img_c_, img_h, img_w) == (image_c, resol / 2, resol / 2));
                                let image = upfirdn2d.forward(image);
                                Ok(image)
                            })
                            .transpose()
                    }
                };

                if is_last || kind == ModelKind::Skip {
                    let to_rgb = ToRgbLayerInit {
                        clamp,
                        ..Default::default()
                    }
                    .build(path / "to_rgb", in_c, out_c, w_dim)?;

                    Box::new(
                        move |xs: &Tensor,
                              image: Option<&Tensor>,
                              styles: &mut TensorIter|
                              -> Result<Option<Tensor>> {
                            let image = upfirdn2d_fn(image)?;
                            let ys = {
                                let styles = styles.next().ok_or_else(styles_err)?;
                                to_rgb.f_forward(xs, &styles)?
                            };
                            let image = match image {
                                Some(image) => image + ys,
                                None => ys,
                            };
                            Ok(Some(image))
                        },
                    )
                } else {
                    Box::new(move |_xs, image, _styles| upfirdn2d_fn(image))
                }
            };

            let forward_fn: ForwardFn = Box::new(
                move |xs: Option<&Tensor>,
                      image: Option<&Tensor>,
                      styles: &mut TensorIter,
                      bsize: i64|
                      -> Result<_> {
                    let xs = input_fn(xs, bsize)?;
                    let xs = main_fn(&xs, styles)?;
                    let image = image_fn(&xs, image, styles)?;
                    Ok((xs, image))
                },
            );

            Ok(SynthesisBlock { forward_fn })
        }
    }

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct SynthesisBlock {
        #[derivative(Debug = "ignore")]
        forward_fn: ForwardFn,
    }

    impl SynthesisBlock {
        pub fn f_forward(
            &self,
            xs: Option<&Tensor>,
            image: Option<&Tensor>,
            styles: &mut TensorIter,
            bsize: usize,
        ) -> Result<(Tensor, Option<Tensor>)> {
            (self.forward_fn)(xs, image, styles, bsize as i64)
        }
    }
}

mod synthesis_network {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct SynthesisNetworkInit {
        pub block_channels_base_pow: usize,
        pub block_channels_max_pow: usize,
        pub block_init: SynthesisBlockInit,
    }

    impl Default for SynthesisNetworkInit {
        fn default() -> Self {
            Self {
                block_channels_base_pow: 15,
                block_channels_max_pow: 9,
                block_init: Default::default(),
            }
        }
    }

    impl SynthesisNetworkInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            w_dim: usize,
            image_resol_pow: usize,
            image_c: usize,
        ) -> Result<SynthesisNetwork> {
            let Self {
                block_channels_base_pow: block_c_base_pow,
                block_channels_max_pow: block_c_max_pow,
                block_init,
            } = self;
            let path = path.borrow();
            ensure!((2..block_c_base_pow).contains(&image_resol_pow));
            let image_resol = 2usize.pow(image_resol_pow as u32);

            let block_params: Vec<_> = (2..=image_resol_pow)
                .map(|block_resol_pow| {
                    let block_resol = 2usize.pow(block_resol_pow as u32);
                    let block_c_pow = cmp::min(block_c_base_pow - block_resol_pow, block_c_max_pow);
                    let block_c = 2usize.pow(block_c_pow as u32);
                    (block_resol, block_c)
                })
                .collect();

            let in_c_iter = chain!(
                iter::once(0),
                block_params.iter().map(|&(_, block_c)| block_c)
            );
            let out_c_iter = block_params.iter().map(|&(_, block_c)| block_c);
            let resol_iter = block_params.iter().map(|&(block_resol, _)| block_resol);

            let blocks: Vec<_> = izip!(in_c_iter, out_c_iter, resol_iter)
                .map(|(in_c, out_c, resol)| -> Result<_> {
                    let is_last = resol == image_resol;
                    let block = block_init
                        .clone()
                        .build(path, in_c, out_c, w_dim, resol, image_c, is_last)?;
                    Ok(block)
                })
                .try_collect()?;

            Ok(SynthesisNetwork { blocks })
        }
    }

    #[derive(Debug)]
    pub struct SynthesisNetwork {
        blocks: Vec<SynthesisBlock>,
    }

    impl SynthesisNetwork {
        pub fn f_forward(&self, styles: &mut TensorIter, bsize: usize) -> Result<Tensor> {
            let (_xs, image) = self.blocks.iter().try_fold(
                (None, None),
                |(xs, image): (Option<Tensor>, Option<Tensor>), block| -> Result<_> {
                    let (xs, image) =
                        block.f_forward(xs.as_ref(), image.as_ref(), styles, bsize)?;
                    Ok((Some(xs), image))
                },
            )?;
            ensure!(styles.next().is_none());
            let image = image.ok_or_else(|| format_err!("please report bug"))?;
            Ok(image)
        }
    }
}

mod stylegan_conv_2d {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct StyleGanConv2DInit {
        pub up: usize,
        pub down: usize,
        pub activation: FixedActivationInit,
        pub ws_init: nn::Init,
        pub bs_init: Option<nn::Init>,
        pub filter: FirFilterInit,
    }

    impl Default for StyleGanConv2DInit {
        fn default() -> Self {
            Self {
                up: default_up(),
                down: default_down(),
                activation: Default::default(),
                ws_init: nn::Init::KaimingUniform,
                bs_init: None,
                filter: Default::default(),
            }
        }
    }

    impl StyleGanConv2DInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            ksize: usize,
        ) -> Result<StyleGanConv2D> {
            let path = path.borrow();
            let Self {
                up,
                down,
                activation,
                ws_init,
                bs_init,
                filter,
            } = self;
            let padding = ksize as i64 / 2;
            let flip_weight = up == 1;

            let conv = Conv2DResampleInit {
                filter,
                padding: [padding; 4],
                up,
                down,
                flip_weight,
                ws_init,
                ..Default::default()
            }
            .build(path, in_c, out_c, ksize)?;
            let bias = bs_init.map(|bs_init| path.var("bias", &[out_c as i64], bs_init));
            let activation = activation.build()?;

            Ok(StyleGanConv2D {
                conv,
                bias,
                activation,
            })
        }
    }

    #[derive(Debug)]
    pub struct StyleGanConv2D {
        conv: Conv2DResample,
        bias: Option<Tensor>,
        activation: FixedActivation,
    }

    impl nn::Module for StyleGanConv2D {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let Self {
                ref conv,
                ref bias,
                ref activation,
            } = *self;

            let xs = conv.forward(xs);
            let xs = match bias {
                Some(bias) => xs + bias.view([1, -1, 1, 1]),
                None => xs,
            };
            activation.forward(&xs)
        }
    }

    impl StyleGanConv2D {
        pub fn set_trainable(&self, trainable: bool) {
            let Self { conv, bias, .. } = self;
            conv.set_trainable(trainable);
            if let Some(bias) = bias {
                let _ = bias.set_requires_grad(trainable);
            }
        }
    }
}

mod to_rgb_layer {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ToRgbLayerInit {
        pub ksize: usize,
        pub clamp: Option<f64>,
    }

    impl Default for ToRgbLayerInit {
        fn default() -> Self {
            Self {
                ksize: 1,
                clamp: None,
            }
        }
    }

    impl ToRgbLayerInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            w_dim: usize,
        ) -> Result<ToRgbLayer> {
            let path = path.borrow();
            let Self { ksize, clamp } = self;
            let linear = nn::linear(
                path / "linear",
                in_c as i64,
                w_dim as i64,
                Default::default(),
            );
            let conv = ModulatedConv2DInit {
                demodulate: false,
                ..Default::default()
            }
            .build(path / "conv", in_c, out_c, ksize)?;
            let bias = path.zeros("bias", &[out_c as i64]);
            let activation = FixedActivationInit {
                clamp,
                ..Default::default()
            }
            .build()?;

            Ok(ToRgbLayer {
                linear,
                conv,
                activation,
                bias,
            })
        }
    }

    pub struct ToRgbLayer {
        linear: nn::Linear,
        conv: ModulatedConv2D,
        bias: Tensor,
        activation: FixedActivation,
    }

    impl ToRgbLayer {
        pub fn f_forward(&self, xs: &Tensor, styles: &Tensor) -> Result<Tensor> {
            let Self {
                linear,
                conv,
                bias,
                activation,
            } = self;

            let styles = linear.forward(styles);
            let xs = conv.f_forward(xs, &styles, None)?;
            let xs = xs + bias;
            let xs = activation.forward(&xs);
            Ok(xs)
        }
    }
}

mod generator {
    use super::*;

    #[derive(Debug, Clone, Default)]
    pub struct StyleGanGeneratorInit {
        pub mapping_network_init: MappingNetworkInit,
        pub synthesis_network_init: SynthesisNetworkInit,
    }

    impl StyleGanGeneratorInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            z_dim: Option<usize>,
            c_dim: Option<usize>,
            w_dim: usize,
            image_resol_pow: usize,
            image_c: usize,
        ) -> Result<StyleGanGenerator> {
            let path = path.borrow();
            let Self {
                mapping_network_init,
                synthesis_network_init,
            } = self;

            let synthesis = synthesis_network_init.build(
                path / "synthesis",
                w_dim,
                image_resol_pow,
                image_c,
            )?;
            let mapping = mapping_network_init.build(
                path / "mapping",
                z_dim,
                c_dim,
                w_dim,
                todo!(), // TODO
            )?;

            Ok(StyleGanGenerator { mapping, synthesis })
        }
    }

    #[derive(Debug)]
    pub struct StyleGanGenerator {
        mapping: MappingNetwork,
        synthesis: SynthesisNetwork,
    }

    impl StyleGanGenerator {
        pub fn f_forward_t(
            &self,
            zs: Option<&Tensor>,
            cs: Option<&Tensor>,
            train: bool,
            skip_w_avg_update: bool,
        ) -> Result<Tensor> {
            let Self { mapping, synthesis } = self;
            let bsize = match (zs, cs) {
                (None, None) => bail!("empty input is not allowed"),
                (None, Some(cs)) => cs.size()[0],
                (Some(zs), None) => zs.size()[0],
                (Some(zs), Some(cs)) => {
                    ensure!(zs.size()[0] == cs.size()[0]);
                    zs.size()[0]
                }
            } as usize;
            let styles = mapping.f_forward(zs, cs, train, skip_w_avg_update)?;
            let image = synthesis.f_forward(
                todo!(), // styles
                bsize,
            )?;
            Ok(image)
        }
    }
}

mod activation {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct FixedActivationInit {
        pub activation: Activation,
        pub gain: f64,
        pub clamp: Option<f64>,
    }

    impl Default for FixedActivationInit {
        fn default() -> Self {
            Self {
                activation: Activation::LRelu,
                gain: 1.0,
                clamp: None,
            }
        }
    }

    impl FixedActivationInit {
        pub fn build(self) -> Result<FixedActivation> {
            let Self {
                activation,
                gain,
                clamp,
            } = self;
            ensure!(clamp.into_iter().all(|clamp| clamp >= 0.0));
            ensure!(gain > 0.0);

            let gain = activation.gain()? * gain;
            let clamp = clamp.map(|clamp| clamp * gain);

            Ok(FixedActivation {
                activation,
                gain,
                clamp,
            })
        }
    }

    #[derive(Debug)]
    pub struct FixedActivation {
        activation: Activation,
        gain: f64,
        clamp: Option<f64>,
    }

    impl nn::Module for FixedActivation {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let Self {
                activation,
                gain,
                clamp,
            } = *self;
            let xs = activation.forward(xs);
            let xs = xs * gain;
            match clamp {
                Some(clamp) => xs.clamp(-clamp, clamp),
                None => xs,
            }
        }
    }

    pub(super) trait ActivationExt {
        fn gain(&self) -> Result<f64>;
    }

    impl ActivationExt for Activation {
        fn gain(&self) -> Result<f64> {
            let act_gain = match self {
                Activation::Linear => 1.0,
                Activation::Relu => 2.0.sqrt(),
                Activation::LRelu => 2.0.sqrt(),
                Activation::Tanh => 1.0,
                Activation::Logistic => 1.0,
                Activation::Elu => 1.0,
                Activation::Selu => 1.0,
                Activation::Swish => 2.0.sqrt(),
                act => bail!("the activation '{:?}' is not supported", act),
            };
            Ok(act_gain)
        }
    }
}

mod mapping_network {

    use super::*;

    #[derive(Debug, Clone)]
    pub struct MappingNetworkInit {
        pub num_layers: usize,
        pub embed_feature_dim: Option<usize>,
        pub layer_feature_dim: Option<usize>,
        pub activation: Activation,
        pub lr_multiplier: f64,
        pub w_avg_beta: Option<f64>,
        pub truncation_psi: f64,
        pub truncation_cutoff: Option<usize>,
    }

    impl Default for MappingNetworkInit {
        fn default() -> Self {
            Self {
                num_layers: 8,
                embed_feature_dim: None,
                layer_feature_dim: None,
                activation: Activation::LRelu,
                lr_multiplier: 0.01,
                w_avg_beta: Some(0.995),
                truncation_psi: 1.0,
                truncation_cutoff: None,
            }
        }
    }

    impl MappingNetworkInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            z_dim: Option<usize>,
            c_dim: Option<usize>,
            w_dim: usize,
            num_ws: Option<usize>,
        ) -> Result<MappingNetwork> {
            let path = path.borrow();
            let Self {
                num_layers,
                embed_feature_dim,
                layer_feature_dim,
                activation,
                lr_multiplier,
                w_avg_beta,
                truncation_psi,
                truncation_cutoff,
            } = self;
            ensure!(num_layers >= 1);
            ensure!(w_dim > 0);
            ensure!(z_dim.into_iter().all(|dim| dim > 0));
            ensure!(c_dim.into_iter().all(|dim| dim > 0));
            ensure!(embed_feature_dim.into_iter().all(|dim| dim > 0));
            ensure!(layer_feature_dim.into_iter().all(|dim| dim > 0));

            let z_dim = z_dim.map(|z_dim| z_dim as i64);
            let layer_feature_dim = layer_feature_dim.unwrap_or(w_dim) as i64;
            let c_embed_dim = match (c_dim, embed_feature_dim) {
                (Some(c_dim), Some(embed_feature_dim)) => Some((c_dim, embed_feature_dim)),
                (Some(c_dim), None) => Some((c_dim, w_dim)),
                (None, Some(_)) => bail!("embed_feature_dim must be None if c_dim is None"),
                (None, None) => None,
            };
            let w_dim = w_dim as i64;

            let input_fn: Box<dyn Fn(_, _) -> _ + Send> = {
                let z_input_fn = match z_dim {
                    Some(z_dim) => Some(move |zs: Tensor| -> Result<Tensor> {
                        ensure!(matches!(zs.size2()?, (_, z_dim_) if z_dim_ == z_dim));
                        Ok(normalize_2nd_moment(&zs))
                    }),
                    None => None,
                };
                let c_input_fn = match c_embed_dim {
                    Some((c_dim, embed_feature_dim)) => {
                        let embed = StyleGanLinearInit::default().build(
                            path / "embed",
                            c_dim,
                            embed_feature_dim,
                        );

                        Some(move |cs: Tensor| -> Result<Tensor> {
                            ensure!(matches!(cs.size2()?, (_, c_dim_) if c_dim_ == c_dim as i64));
                            let ys = embed.forward(&cs);
                            let ys = normalize_2nd_moment(&ys);
                            Ok(ys)
                        })
                    }
                    None => None,
                };
                match (z_input_fn, c_input_fn) {
                    (None, None) => unreachable!(),
                    (None, Some(c_input_fn)) => Box::new(
                        move |_zs: Option<Tensor>, cs: Option<Tensor>| -> Result<_> {
                            let cs = cs.ok_or_else(|| format_err!("expect cs, but get None"))?;
                            let xs = c_input_fn(cs)?;
                            Ok(xs)
                        },
                    ),
                    (Some(z_input_fn), None) => Box::new(
                        move |zs: Option<Tensor>, _cs: Option<Tensor>| -> Result<_> {
                            let zs = zs.ok_or_else(|| format_err!("expect zs, but get None"))?;
                            let xs = z_input_fn(zs)?;
                            Ok(xs)
                        },
                    ),
                    (Some(z_input_fn), Some(c_input_fn)) => {
                        Box::new(move |zs: Option<Tensor>, cs: Option<Tensor>| -> Result<_> {
                            let cs = cs.ok_or_else(|| format_err!("expect cs, but get None"))?;
                            let zs = zs.ok_or_else(|| format_err!("expect zs, but get None"))?;
                            let xs = z_input_fn(zs)?;
                            let ys = c_input_fn(cs)?;
                            Ok(Tensor::cat(&[xs, ys], 1))
                        })
                    }
                }
            };

            let main_fn = {
                let channels_iter = {
                    let embed_dim = c_embed_dim.map(|(_, embed_dim)| embed_dim as i64);
                    let input_c = z_dim.unwrap_or(0) + embed_dim.unwrap_or(0);

                    chain!(
                        iter::once(input_c),
                        iter::repeat(layer_feature_dim).take(num_layers - 1),
                        iter::once(w_dim)
                    )
                };
                let layers: Vec<_> = izip!(channels_iter.clone(), channels_iter.skip(1))
                    .enumerate()
                    .map(|(index, (in_c, out_c))| {
                        StyleGanLinearInit {
                            activation,
                            lr_multiplier,
                            ..Default::default()
                        }
                        .build(
                            path / format!("layer{}", index),
                            in_c as usize,
                            out_c as usize,
                        )
                    })
                    .collect();
                move |xs: &Tensor| {
                    layers
                        .iter()
                        .fold(xs.shallow_clone(), |xs, layer| layer.forward(&xs))
                }
            };

            let update_fn = if let (Some(num_ws), Some(w_avg_beta)) = (num_ws, w_avg_beta) {
                let w_avg = path.zeros_no_train("w_avg", &[w_dim]);

                Some(move |xs: &Tensor, train: bool, skip_w_avg_update: bool| {
                    if (true, false) == (train, skip_w_avg_update) {
                        let new_w_avg = xs
                            .detach()
                            .mean_dim(&[0], false, Kind::Float)
                            .lerp(&w_avg, w_avg_beta);
                        w_avg.shallow_clone().copy_(&new_w_avg);
                    }

                    let xs = xs.unsqueeze(1).repeat(&[1, num_ws as i64, 1]);

                    match truncation_cutoff {
                        Some(truncation_cutoff) => {
                            let truncation_cutoff = truncation_cutoff as i64;
                            let xs_former = xs.i((.., ..truncation_cutoff));
                            let xs_latter = xs.i((.., truncation_cutoff..));
                            let xs_former = w_avg.lerp(&xs_former, truncation_psi);
                            Tensor::cat(&[xs_former, xs_latter], 1)
                        }
                        None => w_avg.lerp(&xs, truncation_psi),
                    }
                })
            } else {
                None
            };

            let forward_fn = Box::new(
                move |zs: Option<&Tensor>,
                      cs: Option<&Tensor>,
                      train: bool,
                      skip_w_avg_update: bool|
                      -> Result<Tensor> {
                    let zs = zs.map(|zs| zs.shallow_clone());
                    let cs = cs.map(|cs| cs.shallow_clone());
                    let xs = input_fn(zs, cs)?;
                    let xs = main_fn(&xs);
                    let xs = match &update_fn {
                        Some(update_fn) => update_fn(&xs, train, skip_w_avg_update),
                        None => xs,
                    };

                    Ok(xs)
                },
            );

            Ok(MappingNetwork { forward_fn })
        }
    }

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct MappingNetwork {
        #[derivative(Debug = "ignore")]
        forward_fn:
            Box<dyn Fn(Option<&Tensor>, Option<&Tensor>, bool, bool) -> Result<Tensor> + Send>,
    }

    impl MappingNetwork {
        pub fn f_forward(
            &self,
            zs: Option<&Tensor>,
            cs: Option<&Tensor>,
            train: bool,
            skip_w_avg_update: bool,
        ) -> Result<Tensor> {
            (self.forward_fn)(zs, cs, train, skip_w_avg_update)
        }
    }

    fn normalize_2nd_moment(xs: &Tensor) -> Tensor {
        const DIM: i64 = 1;
        const EPS: f64 = 1e-8;
        xs * (xs.square().mean_dim(&[DIM], true, xs.kind()) + EPS).rsqrt()
    }
}

mod stylegan_linear {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct StyleGanLinearInit {
        pub activation: Activation,
        pub lr_multiplier: f64,
        pub ws_init: nn::Init,
        pub bs_init: Option<nn::Init>,
    }

    impl Default for StyleGanLinearInit {
        fn default() -> Self {
            Self {
                activation: Activation::Linear,
                lr_multiplier: 1.0,
                ws_init: nn::Init::KaimingUniform,
                bs_init: Some(nn::Init::Const(0.0)),
            }
        }
    }

    impl StyleGanLinearInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
        ) -> StyleGanLinear {
            let path = path.borrow();
            let Self {
                activation,
                lr_multiplier,
                ws_init,
                bs_init,
            } = self;
            let in_c = in_c as i64;
            let out_c = out_c as i64;

            let weight = path.var("weight", &[out_c, in_c], ws_init);
            let bias = bs_init.map(|bs_init| path.var("bias", &[out_c], bs_init) / lr_multiplier);

            StyleGanLinear {
                weight,
                bias,
                activation,
            }
        }
    }

    #[derive(Debug)]
    pub struct StyleGanLinear {
        weight: Tensor,
        bias: Option<Tensor>,
        activation: Activation,
    }

    impl nn::Module for StyleGanLinear {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let Self {
                ref weight,
                ref bias,
                activation,
            } = *self;

            let xs = xs.matmul(&weight.tr());
            let xs = match bias.as_ref() {
                Some(bias) => xs + bias,
                None => xs,
            };
            let xs = activation.forward(&xs);
            xs
        }
    }
}

mod discriminator_block {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DiscriminatorBlockInit {
        pub activation: FixedActivationInit,
        pub filter: FirFilter,
        pub kind: ModelKind,
    }

    impl Default for DiscriminatorBlockInit {
        fn default() -> Self {
            Self {
                kind: ModelKind::Resnet,
                filter: Default::default(),
                activation: Default::default(),
            }
        }
    }

    impl DiscriminatorBlockInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: Option<usize>,
            out_c: usize,
            tmp_c: usize,
            resol: usize,
            image_c: usize,
        ) -> Result<DiscriminatorBlock> {
            let path = path.borrow();
            let Self {
                activation,
                filter,
                kind,
            } = self;
            ensure!(in_c.into_iter().all(|in_c| in_c > 0));
            let in_c = in_c.map(|in_c| in_c as i64);
            let resol = resol as i64;

            let input_fn: Box<dyn Fn(Option<&Tensor>) -> Result<Option<Tensor>> + Send> = match in_c
            {
                Some(in_c) => Box::new(move |xs: Option<&Tensor>| {
                    let xs = xs.ok_or_else(|| format_err!("expect xs, but get None"))?;
                    ensure!(
                        matches!(xs.size4()?, (_, in_c_, in_h, in_w) if in_c_ == in_c && in_h == resol && in_w == resol)
                    );
                    Ok(Some(xs.shallow_clone()))
                }),
                None => Box::new(move |xs: Option<&Tensor>| {
                    ensure!(xs.is_none());
                    Ok(None)
                }),
            };

            let image_fn: Box<dyn Fn(Option<Tensor>, &Tensor) -> Result<(Tensor, Tensor)> + Send> =
                if in_c.is_none() || kind == ModelKind::Skip {
                    let from_rgb = StyleGanConv2DInit {
                        activation: FixedActivationInit {
                            gain: 1.0,
                            ..activation
                        },
                        ..Default::default()
                    }
                    .build(path / "from_rgb", image_c, tmp_c, 1)?;

                    Box::new(move |xs: Option<Tensor>, image: &Tensor| {
                        let ys = from_rgb.forward(image);
                        let xs = match xs {
                            Some(xs) => xs + ys,
                            None => ys,
                        };
                        let image: Tensor = todo!();
                        Ok((xs, image))
                    })
                } else {
                    Box::new(move |xs: Option<Tensor>, image: &Tensor| {
                        let xs = xs.ok_or_else(|| format_err!("expect xs, but get None"))?;
                        let image = image.shallow_clone();
                        Ok((xs, image))
                    })
                };

            let main_fn: Box<dyn Fn(&Tensor) -> Tensor + Send> = {
                let gain = if kind == ModelKind::Resnet {
                    0.5.sqrt()
                } else {
                    1.0
                };

                let conv0 = StyleGanConv2DInit {
                    activation: FixedActivationInit {
                        gain: 1.0,
                        ..activation
                    },
                    ..Default::default()
                }
                .build(path / "conv0", tmp_c, tmp_c, 3)?;

                let conv1 = StyleGanConv2DInit {
                    down: 2,
                    activation: FixedActivationInit { gain, ..activation },
                    filter: FirFilterInit {
                        filter: filter.clone(),
                        ..Default::default()
                    },
                    ..Default::default()
                }
                .build(path / "conv1", tmp_c, out_c, 3)?;

                if kind == ModelKind::Resnet {
                    let skip = StyleGanConv2DInit {
                        down: 2,
                        activation: FixedActivationInit { gain, ..activation },
                        filter: FirFilterInit {
                            filter,
                            ..Default::default()
                        },
                        bs_init: None,
                        ..Default::default()
                    }
                    .build(path / "skip", tmp_c, out_c, 1)?;

                    Box::new(move |xs: &Tensor| {
                        let ys = skip.forward(xs);
                        let xs = conv0.forward(xs);
                        let xs = conv1.forward(&xs);
                        let xs = xs + ys;
                        xs
                    })
                } else {
                    Box::new(move |xs: &Tensor| {
                        let xs = conv0.forward(xs);
                        let xs = conv1.forward(&xs);
                        xs
                    })
                }
            };

            let forward_fn = Box::new(move |xs: Option<&Tensor>, image: &Tensor| {
                let xs = input_fn(xs)?;
                let (xs, image) = image_fn(xs, image)?;
                let xs = main_fn(&xs);
                Ok((xs, image))
            });

            Ok(DiscriminatorBlock { forward_fn })
        }
    }

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct DiscriminatorBlock {
        #[derivative(Debug = "ignore")]
        forward_fn: Box<dyn Fn(Option<&Tensor>, &Tensor) -> Result<(Tensor, Tensor)> + Send>,
    }

    impl DiscriminatorBlock {
        pub fn forward(&self, xs: Option<&Tensor>, image: &Tensor) -> Result<(Tensor, Tensor)> {
            (self.forward_fn)(xs, image)
        }
    }
}

mod discriminator_epilogue {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DiscriminatorEpilogueInit {
        pub kind: ModelKind,
        pub mbstd_init: Option<MinibatchStdLayerInit>,
        pub activation: Activation,
        pub clamp: Option<f64>,
    }

    impl Default for DiscriminatorEpilogueInit {
        fn default() -> Self {
            Self {
                kind: ModelKind::Resnet,
                mbstd_init: Some(MinibatchStdLayerInit {
                    minibatch_size: Some(4),
                    num_groups: Some(1),
                }),
                activation: Activation::LRelu,
                clamp: None,
            }
        }
    }

    impl DiscriminatorEpilogueInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            cmap_dim: Option<usize>,
            resol: usize,
            image_c: usize,
        ) -> Result<DiscriminatorEpilogue> {
            let path = path.borrow();
            let Self {
                kind,
                activation,
                clamp,
                mbstd_init,
            } = self;
            ensure!(cmap_dim.into_iter().all(|dim| dim > 0));

            let input_fn = move |xs: &Tensor| {
                let in_c = in_c as i64;
                let resol = resol as i64;
                ensure!(
                    matches!(xs.size4()?, (_, in_c_, in_h, in_w) if in_c_ == in_c && in_h == resol && in_w == resol)
                );

                Ok(xs.shallow_clone())
            };

            let image_fn: Box<dyn Fn(&Tensor, &Tensor) -> Result<Tensor> + Send> = match kind {
                ModelKind::Skip => {
                    let from_rgb = StyleGanConv2DInit {
                        activation: FixedActivationInit {
                            activation,
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                    .build(path / "from_rgb", image_c, in_c, 1)?;

                    Box::new(move |xs, image| {
                        let image_c = image_c as i64;
                        let resol = resol as i64;
                        ensure!(
                            matches!(image.size4()?, (_, image_c_, image_h, image_w) if image_c_ == image_c  && image_h == resol  && image_w == resol)
                        );

                        let xs = xs + from_rgb.forward(xs);
                        Ok(xs)
                    })
                }
                _ => Box::new(move |xs, _image| Ok(xs.shallow_clone())),
            };

            let conv_fn: Box<dyn Fn(&Tensor) -> Result<Tensor> + Send> = match mbstd_init {
                Some(mbstd_init) => {
                    let mbstd = mbstd_init.build()?;
                    let conv = StyleGanConv2DInit {
                        activation: FixedActivationInit {
                            activation,
                            clamp,
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                    .build(
                        path / "conv",
                        in_c + mbstd.num_groups(),
                        in_c,
                        3,
                    )?;

                    Box::new(move |xs| {
                        let xs = mbstd.f_forward(xs)?;
                        let xs = conv.forward(&xs);
                        Ok(xs)
                    })
                }
                None => {
                    let conv = StyleGanConv2DInit {
                        activation: FixedActivationInit {
                            activation,
                            clamp,
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                    .build(path / "conv", in_c, in_c, 3)?;

                    Box::new(move |xs| {
                        let xs = conv.forward(xs);
                        Ok(xs)
                    })
                }
            };

            let linear_fn = {
                let linear1 = StyleGanLinearInit {
                    activation,
                    ..Default::default()
                }
                .build(path / "linear1", in_c * resol * resol, in_c);
                let linear2 = StyleGanLinearInit {
                    activation,
                    ..Default::default()
                }
                .build(path / "linear2", in_c, cmap_dim.unwrap_or(1));

                move |xs: &Tensor| {
                    let xs = xs.flatten(1, -1);
                    let xs = linear1.forward(&xs);
                    let xs = linear2.forward(&xs);
                    xs
                }
            };

            let cond_fn: Box<dyn Fn(&Tensor, Option<&Tensor>) -> Result<Tensor> + Send> =
                match cmap_dim {
                    Some(cmap_dim) => Box::new(move |xs: &Tensor, cmap: Option<&Tensor>| {
                        let cmap = cmap.ok_or_else(|| format_err!("expect cmap, but get None"))?;
                        ensure!(
                            matches!(cmap.size2()?, (_, cmap_dim_) if cmap_dim_ == cmap_dim as i64)
                        );
                        let xs = (xs * cmap).sum_dim_intlist(&[1], true, Kind::Float)
                            * (cmap_dim as f64).sqrt().recip();
                        Ok(xs)
                    }),
                    None => {
                        Box::new(move |xs: &Tensor, _cmap: Option<&Tensor>| Ok(xs.shallow_clone()))
                    }
                };

            let forward_fn = Box::new(move |xs: &Tensor, image: &Tensor, cmap: Option<&Tensor>| {
                let xs = input_fn(xs)?;
                let xs = image_fn(&xs, image)?;
                let xs = conv_fn(&xs)?;
                let xs = linear_fn(&xs);
                let xs = cond_fn(&xs, cmap)?;
                Ok(xs)
            });

            Ok(DiscriminatorEpilogue { forward_fn })
        }
    }

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct DiscriminatorEpilogue {
        #[derivative(Debug = "ignore")]
        forward_fn: Box<dyn Fn(&Tensor, &Tensor, Option<&Tensor>) -> Result<Tensor> + Send>,
    }

    impl DiscriminatorEpilogue {
        pub fn f_forward(
            &self,
            xs: &Tensor,
            image: &Tensor,
            cmap: Option<&Tensor>,
        ) -> Result<Tensor> {
            (self.forward_fn)(xs, image, cmap)
        }
    }
}

mod minibatch_std_layer {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct MinibatchStdLayerInit {
        pub minibatch_size: Option<usize>,
        pub num_groups: Option<usize>,
    }

    impl MinibatchStdLayerInit {
        pub fn build(self) -> Result<MinibatchStdLayer> {
            let Self {
                minibatch_size,
                num_groups,
            } = self;

            ensure!(minibatch_size.into_iter().all(|size| size > 0));
            ensure!(num_groups.into_iter().all(|num| num > 0));

            Ok(MinibatchStdLayer {
                minibatch_size: minibatch_size.map(|size| size as i64),
                num_groups: num_groups.unwrap_or(1) as i64,
            })
        }
    }

    #[derive(Debug)]
    pub struct MinibatchStdLayer {
        minibatch_size: Option<i64>,
        num_groups: i64,
    }

    impl MinibatchStdLayer {
        pub fn f_forward(&self, xs: &Tensor) -> Result<Tensor> {
            let Self {
                minibatch_size: size_mb,
                num_groups: num_g,
            } = *self;

            let (in_b, in_c, in_h, in_w) = xs.size4()?;
            let size_mb = size_mb.unwrap_or(in_b);

            ensure!(in_c % num_g == 0);
            ensure!(in_b % size_mb == 0);

            let num_mb = in_b / size_mb;
            let size_g = in_c / num_g;

            let avg_stdev = {
                let ys = xs.reshape(&[size_mb, num_mb, num_g, size_g, in_h, in_w]); // [m, M, G, g, h, w]

                // mean, var, stdev per minibatch
                let mean = ys.mean_dim(&[0], true, Kind::Float); // [1, M, G, g, h, w]
                let var = (ys - mean).square().mean_dim(&[0], false, Kind::Float); // [M, G, g, h, w]
                let stdev = (var + 1e-8).sqrt(); // [M, G, g, h, w]

                // average stdev per minibatch per group
                let avg_stdev = stdev.mean_dim(&[2, 3, 4], false, Kind::Float); // [M, G]

                avg_stdev
            };

            let avg_stdev = avg_stdev // [M, G]
                .reshape(&[num_mb, num_g, 1, 1]) // [M, G, 1, 1]
                .repeat(&[size_mb, 1, in_h, in_w]); // [b, G, h, w]

            let output = Tensor::cat(&[xs, &avg_stdev], 1); // [b, c + G, h, w]

            Ok(output)
        }

        pub fn num_groups(&self) -> usize {
            self.num_groups as usize
        }
    }
}

mod discriminator {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct StyleGanDiscriminatorInit {
        pub kind: ModelKind,
        pub channel_base_pow: usize,
        pub channel_max_pow: usize,
        pub clamp: Option<f64>,
        pub cmap_dim: Option<usize>,
        pub block_init: DiscriminatorBlockInit,
        pub mapping_network_init: MappingNetworkInit,
        pub epilogue_init: DiscriminatorEpilogueInit,
    }

    impl Default for StyleGanDiscriminatorInit {
        fn default() -> Self {
            Self {
                kind: ModelKind::Resnet,
                channel_base_pow: 15,
                channel_max_pow: 9,
                clamp: None,
                cmap_dim: None,
                block_init: Default::default(),
                mapping_network_init: Default::default(),
                epilogue_init: Default::default(),
            }
        }
    }

    impl StyleGanDiscriminatorInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            c_dim: Option<usize>,
            image_resol_pow: usize,
            image_c: usize,
        ) -> Result<StyleGanDiscriminator> {
            let path = path.borrow();
            let Self {
                kind,
                channel_base_pow,
                channel_max_pow,
                clamp,
                cmap_dim,
                block_init,
                mapping_network_init,
                epilogue_init,
            } = self;
            ensure!(c_dim.into_iter().all(|dim| dim > 0));

            let image_resol = 2usize.pow(image_resol_pow as u32);

            let cmap_dim = cmap_dim.unwrap_or_else(|| {
                let pow = (channel_base_pow - 2).min(channel_max_pow);
                2usize.pow(pow as u32)
            });

            (3..=image_resol_pow).rev().map(|block_resol_pow| {
                let block_resol = 2usize.pow(block_resol_pow as u32);
                let block_c_pow = (channel_base_pow - block_resol_pow).min(channel_max_pow);
                let block_c = 2usize.pow(block_c_pow as u32);
            });

            todo!();
        }
    }

    #[derive(Debug)]
    pub struct StyleGanDiscriminator {}
}

mod misc {
    // pub(super) trait IntoSendBox
    // where
    //     Self: Send,
    // {
    //     fn into_send_box(self) -> Box<Self>;
    // }

    // impl<T: Send> IntoSendBox for T {
    //     fn into_send_box(self) -> Box<Self> {
    //         Box::new(self)
    //     }
    // }
}

fn default_flip_weight() -> bool {
    true
}

fn default_down() -> usize {
    1
}

fn default_up() -> usize {
    1
}

fn default_padding() -> [i64; 4] {
    [0; 4]
}
