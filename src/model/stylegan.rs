use crate::common::*;

pub use conv_2d_resample::*;
pub use fir_filter::*;
pub use misc::*;
pub use modulated_conv2d::*;
pub use stylegan_conv_2d::*;
pub use synthesis_block::*;
pub use synthesis_layer::*;
pub use synthesis_network::*;
pub use to_rgb_layer::*;
pub use up_fir_down_2d::*;

mod fir_filter {
    use super::*;

    pub fn default_fir_values() -> Tensor {
        Tensor::ones(&[1, 1], (Kind::Float, Device::Cpu))
    }

    pub trait FirFilterValues {
        fn to_tensor(self) -> Result<Tensor>;
    }

    impl FirFilterValues for Tensor {
        fn to_tensor(self) -> Result<Tensor> {
            ensure!(self.kind() == Kind::Float && self.numel() > 0 && !self.requires_grad());
            let tensor = match self.dim() {
                0 => self.view([1, 1]),
                1 => self.outer(&self),
                2 => self,
                dim => bail!("filter dimension must be one of 0, 1, 2, but get {}", dim),
            };
            Ok(tensor)
        }
    }

    impl FirFilterValues for f64 {
        fn to_tensor(self) -> Result<Tensor> {
            Ok(Tensor::from(self).view([1, 1]))
        }
    }

    impl<const N: usize> FirFilterValues for [f64; N] {
        fn to_tensor(self) -> Result<Tensor> {
            let tensor = Tensor::of_slice(self.as_ref());
            Ok(tensor.outer(&tensor))
        }
    }

    impl<const R: usize, const C: usize> FirFilterValues for [[f64; C]; R] {
        fn to_tensor(self) -> Result<Tensor> {
            let tensor: Tensor = self.into_cv();
            Ok(tensor)
        }
    }

    #[derive(Debug, Clone)]
    pub struct FirFilterOptions {
        pub normalize: bool,
        pub flip: bool,
        pub gain: f64,
    }

    impl Default for FirFilterOptions {
        fn default() -> Self {
            Self {
                normalize: true,
                flip: false,
                gain: 1.0,
            }
        }
    }

    pub fn make_fir_filter(
        values: impl FirFilterValues,
        options: FirFilterOptions,
    ) -> Result<Tensor> {
        let FirFilterOptions {
            normalize,
            flip,
            gain,
        } = options;

        let filter = values.to_tensor()?;
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

mod up_fir_down_2d {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct UpFirDn2DInit {
        pub filter_options: FirFilterOptions,
        pub up: usize,
        pub down: usize,
        pub padding: [i64; 4],
    }

    impl Default for UpFirDn2DInit {
        fn default() -> Self {
            Self {
                filter_options: Default::default(),
                up: 1,
                down: 1,
                padding: [0; 4],
            }
        }
    }

    impl UpFirDn2DInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            filter: impl FirFilterValues,
        ) -> Result<UpFirDn2D> {
            let path = path.borrow();

            let Self {
                filter_options,
                up,
                down,
                padding,
            } = self;

            ensure!(up > 0);
            ensure!(down > 0);
            let up = up as i64;
            let down = down as i64;
            let filter = make_fir_filter(filter, filter_options)?;

            let weight = {
                let mut weight = path.zeros_no_train("weight", &filter.size());
                weight.copy_(&filter);
                weight
            };

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
        pub filter_options: FirFilterOptions,
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
                filter_options: Default::default(),
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
            filter: impl FirFilterValues,
        ) -> Result<Conv2DResample> {
            let Self {
                filter_options,
                padding,
                up,
                down,
                groups,
                flip_weight,
                ws_init,
            } = self;
            ensure!(groups >= 1);
            let path = path.borrow();
            let filter = filter.to_tensor()?;
            let (fh, fw) = filter.size2().unwrap();
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
                filter_options: filter_options.clone(),
                up,
                down: 1,
                padding: conv_padding,
            }
            .build(path / "up_conv", filter.copy())?;

            let down_conv = UpFirDn2DInit {
                filter_options,
                up: 1,
                down,
                padding: [0, 0, 0, 0],
            }
            .build(path / "down_conv", filter)?;

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
        pub filter_options: FirFilterOptions,
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
                filter_options: Default::default(),
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
            filter: impl FirFilterValues,
        ) -> Result<ModulatedConv2D> {
            let path = path.borrow();
            let Self {
                up,
                down,
                filter_options,
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
                filter_options,
                flip_weight,
                groups: 1,
            }
            .build(path, in_c, out_c, ksize, filter)?;
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
        pub filter_options: FirFilterOptions,
    }

    impl SynthesisLayerInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            w_dim: usize,
            resolution: usize,
            filter: impl FirFilterValues,
        ) -> Result<SynthesisLayer> {
            let path = path.borrow();
            let Self {
                ksize,
                up,
                activation,
                noise_mode,
                filter_options,
            } = self;

            let conv = ModulatedConv2DInit {
                up,
                padding: [ksize as i64 / 2; 4],
                flip_weight: up == 1,
                filter_options,
                ..Default::default()
            }
            .build(path, in_c, out_c, ksize, filter)?;

            let bias = path.zeros("bias", &[out_c as i64]);

            let noise_const = if noise_mode == NoiseMode::Const {
                let resol = resolution as i64;
                Some(path.randn("noise_const", &[resol, resol], 0.0, 1.0))
            } else {
                None
            };

            let noise_strength = path.zeros("noise_strength", &[]);

            let linear = nn::linear(
                path / "linear",
                in_c as i64,
                w_dim as i64,
                Default::default(),
            );

            let activation = activation.build()?;

            Ok(SynthesisLayer {
                up: up as i64,
                resolution: resolution as i64,
                noise_mode,
                activation,
                conv,
                linear,
                bias,
                noise_const,
                noise_strength,
            })
        }
    }

    pub struct SynthesisLayer {
        up: i64,
        resolution: i64,
        noise_mode: NoiseMode,
        conv: ModulatedConv2D,
        linear: nn::Linear,
        activation: FixedActivation,
        bias: Tensor,
        noise_const: Option<Tensor>,
        noise_strength: Tensor,
    }

    impl SynthesisLayer {
        pub fn f_forward(&self, xs: &Tensor, styles: &Tensor) -> Result<Tensor> {
            let Self {
                up,
                noise_mode,
                resolution,
                ref conv,
                ref linear,
                ref bias,
                ref activation,
                ref noise_const,
                ref noise_strength,
            } = *self;
            let in_resol = resolution / up;
            let bsize = {
                let (bsize, _, in_h, in_w) = xs.size4()?;
                ensure!((in_h, in_w) == (in_resol, in_resol));
                bsize
            };

            let styles: Tensor = linear.forward(styles);
            let noise = match noise_mode {
                NoiseMode::Random => {
                    let noise = Tensor::randn(
                        &[bsize, 1, resolution, resolution],
                        (Kind::Float, xs.device()),
                    );
                    Some(noise)
                }
                NoiseMode::Const => noise_const.shallow_clone(),
                NoiseMode::None => None,
            }
            .map(|noise| noise * noise_strength);

            let xs = conv.f_forward(xs, &styles, noise.as_ref())?;
            let xs = xs + bias.view([1, -1, 1, 1]);
            let xs = activation.forward(&xs);

            Ok(xs)
        }
    }
}

mod synthesis_block {
    use super::*;

    type TensorIter = Box<dyn Iterator<Item = Tensor> + Send>;
    type InputFn = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send>;
    type MainFn = Box<dyn Fn(&Tensor, &mut TensorIter) -> Result<Tensor> + Send>;
    type ImageFn =
        Box<dyn Fn(&Tensor, Option<&Tensor>, &mut TensorIter) -> Result<Option<Tensor>> + Send>;
    type ForwardFn = Box<
        dyn Fn(&Tensor, Option<&Tensor>, &mut TensorIter) -> Result<(Tensor, Option<Tensor>)>
            + Send,
    >;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum SynthesisBlockKind {
        Orig,
        Skip,
        Resnet,
    }

    #[derive(Debug, Clone)]
    pub struct SynthesisBlockInit {
        pub kind: SynthesisBlockKind,
        pub synthesis_init: SynthesisLayerInit,
        pub clamp: Option<f64>,
    }

    impl SynthesisBlockInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            w_dim: usize,
            resolution: usize,
            image_c: usize,
            is_last: bool,
            filter: impl FirFilterValues,
        ) -> Result<SynthesisBlock> {
            let path = path.borrow();
            let Self {
                kind,
                synthesis_init,
                clamp,
            } = self;
            let filter = filter.to_tensor()?;

            fn styles_err() -> Error {
                format_err!("invalid styles input length")
            }

            let input_fn: InputFn = if in_c == 0 {
                let out_c = out_c as i64;
                let resolution = resolution as i64;
                let r#const = path.randn("const", &[out_c, resolution, resolution], 0.0, 1.0);

                Box::new(move |xs: &Tensor| -> Result<_> {
                    let (bsize, _, _, _) = xs.size4()?;
                    let xs = r#const
                        .shallow_clone()
                        .unsqueeze(0)
                        .repeat(&[bsize, 1, 1, 1]);
                    Ok(xs)
                })
            } else {
                let in_c = in_c as i64;
                let resolution = resolution as i64;

                Box::new(move |xs: &Tensor| -> Result<_> {
                    let (_, in_c_, in_h, in_w) = xs.size4()?;
                    ensure!((in_c_, in_h, in_w) == (in_c, resolution / 2, resolution / 2));
                    Ok(xs.shallow_clone())
                })
            };

            let main_fn: MainFn = {
                let conv1 = SynthesisLayerInit {
                    activation: FixedActivationInit {
                        clamp,
                        ..Default::default()
                    },
                    ..synthesis_init.clone()
                }
                .build(
                    path / "conv1",
                    out_c,
                    out_c,
                    w_dim,
                    resolution,
                    filter.shallow_clone(),
                )?;

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
                        ..synthesis_init
                    }
                    .build(
                        path / "conv0",
                        in_c,
                        out_c,
                        w_dim,
                        resolution,
                        filter.shallow_clone(),
                    )?;

                    match kind {
                        SynthesisBlockKind::Resnet => {
                            let skip = StyleGanConv2DInit {
                                up: 2,
                                bs_init: None,
                                activation: FixedActivationInit {
                                    gain: 0.5.sqrt(),
                                    ..Default::default()
                                },
                                ..Default::default()
                            }
                            .build(
                                path / "skip",
                                in_c,
                                out_c,
                                1,
                                filter.shallow_clone(),
                            )?;

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
                        SynthesisBlockKind::Orig | SynthesisBlockKind::Skip => {
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
                    let resolution = resolution as i64;
                    let upfirdn2d = UpFirDn2DInit::default().build(path / "upfirdn2d", filter)?;

                    move |image: Option<&Tensor>| -> Result<_> {
                        image
                            .map(|image| -> Result<_> {
                                let (_bsize, img_c_, img_h, img_w) = image.size4()?;
                                ensure!(
                                    (img_c_, img_h, img_w)
                                        == (image_c, resolution / 2, resolution / 2)
                                );
                                let image = upfirdn2d.forward(image);
                                Ok(image)
                            })
                            .transpose()
                    }
                };

                if is_last || kind == SynthesisBlockKind::Skip {
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
                move |xs: &Tensor, image: Option<&Tensor>, styles: &mut TensorIter| -> Result<_> {
                    let xs = input_fn(xs)?;
                    let xs = main_fn(&xs, styles)?;
                    let image = image_fn(&xs, image, styles)?;
                    Ok((xs, image))
                },
            );

            Ok(SynthesisBlock { forward_fn })
        }
    }

    pub struct SynthesisBlock {
        forward_fn: ForwardFn,
    }

    impl SynthesisBlock {
        pub fn f_forward(
            &self,
            xs: &Tensor,
            image: Option<&Tensor>,
            styles: impl IntoIterator<IntoIter = impl 'static + Iterator<Item = impl Borrow<Tensor>> + Send>
                + Send,
        ) -> Result<(Tensor, Option<Tensor>)> {
            let mut styles: TensorIter = Box::new(
                styles
                    .into_iter()
                    .map(|tensor| tensor.borrow().shallow_clone()),
            );
            (self.forward_fn)(xs, image, &mut styles)
        }
    }
}

// mod synthesis_network {
//     use super::*;

//     #[derive(Debug, Clone)]
//     pub struct SynthesisNetworkInit {}
// }

mod stylegan_conv_2d {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct StyleGanConv2DInit {
        pub up: usize,
        pub down: usize,
        pub activation: FixedActivationInit,
        pub ws_init: nn::Init,
        pub bs_init: Option<nn::Init>,
        pub filter_options: FirFilterOptions,
    }

    impl Default for StyleGanConv2DInit {
        fn default() -> Self {
            Self {
                up: default_up(),
                down: default_down(),
                activation: Default::default(),
                ws_init: nn::Init::KaimingUniform,
                bs_init: None,
                filter_options: Default::default(),
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
            filter: impl FirFilterValues,
        ) -> Result<StyleGanConv2D> {
            let path = path.borrow();
            let Self {
                up,
                down,
                activation,
                ws_init,
                bs_init,
                filter_options,
            } = self;
            let padding = ksize as i64 / 2;
            let flip_weight = up == 1;

            let conv = Conv2DResampleInit {
                filter_options,
                padding: [padding; 4],
                up,
                down,
                flip_weight,
                ws_init,
                ..Default::default()
            }
            .build(path, in_c, out_c, ksize, filter)?;
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
            .build(path / "conv", in_c, out_c, ksize, default_fir_values())?;
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

mod misc {
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
