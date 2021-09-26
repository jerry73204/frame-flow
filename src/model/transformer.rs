use super::misc::{NormKind, PaddingKind};
use crate::{common::*, message as msg, utils::*};
// use tch_modules::GroupNormInit;

use denormed_detection::*;
use detection_encode::*;

pub use motion_based::*;
pub use potential_based::*;

mod potential_based {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct PotentialBasedTransformerInit {
        pub ksize: usize,
        pub norm_kind: NormKind,
        pub padding_kind: PaddingKind,
    }

    impl Default for PotentialBasedTransformerInit {
        fn default() -> Self {
            Self {
                padding_kind: PaddingKind::Reflect,
                norm_kind: NormKind::InstanceNorm,
                ksize: 3,
            }
        }
    }

    impl PotentialBasedTransformerInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            input_len: usize,
            num_classes: usize,
            inner_c: usize,
        ) -> Result<PotentialBasedTransformer> {
            // const BORDER_SIZE_RATIO: f64 = 4.0 / 64.0;

            let path = path.borrow();
            let Self {
                ksize,
                norm_kind,
                padding_kind,
            } = self;
            // let device = path.device();
            let in_c = 5 + num_classes;
            ensure!(input_len > 0);

            let ctx_c = in_c * input_len;

            let gaussian_blur =
                GaussianBlur::new(path / "gaussian_blur", &[5, 5], &[1.0, 1.0]).unwrap();

            let feature_maps: Vec<_> = array::IntoIter::new([
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 4,
                    num_mid: 1,
                    num_up: 4,
                }
                .build(path / "feature_0", ctx_c, inner_c, inner_c), // 64 -> 64
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 4,
                    num_mid: 1,
                    num_up: 3,
                }
                .build(path / "feature_1", inner_c, inner_c, inner_c), // 64 -> 32
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 3,
                    num_mid: 1,
                    num_up: 2,
                }
                .build(path / "feature_2", inner_c, inner_c, inner_c), // 32 -> 16
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 2,
                    num_mid: 1,
                    num_up: 1,
                }
                .build(path / "feature_3", inner_c, inner_c, inner_c), // 16 -> 8
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 1,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "feature_4", inner_c, inner_c, inner_c), // 8 -> 4
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 1,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "feature_5", inner_c, inner_c, inner_c), // 4 -> 2
            ])
            .try_collect()
            .unwrap();

            let motion_maps: Vec<_> = array::IntoIter::new([
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 4,
                    num_mid: 1,
                    num_up: 4,
                }
                .build(path / "motion_0", inner_c + 1, 1, inner_c), // 64
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 3,
                    num_mid: 1,
                    num_up: 3,
                }
                .build(path / "motion_1", inner_c + 1, 1, inner_c), // 32
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 3,
                    num_mid: 1,
                    num_up: 3,
                }
                .build(path / "motion_2", inner_c + 1, 1, inner_c), // 16
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 2,
                    num_mid: 1,
                    num_up: 2,
                }
                .build(path / "motion_3", inner_c + 1, 1, inner_c), // 8
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 0,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "motion_4", inner_c + 1, 1, inner_c), // 4
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 0,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "motion_5", inner_c, 1, inner_c), // 2
            ])
            .try_collect()
            .unwrap();

            // let spectral_coefs = {
            //     const SIZE: i64 = 64;

            //     let coef = (Tensor::arange(SIZE, (Kind::Float, device))
            //         .view([1, SIZE])
            //         .expand(&[SIZE, SIZE], false)
            //         .pow(2)
            //         + Tensor::arange(SIZE, (Kind::Float, device))
            //             .view([SIZE, 1])
            //             .expand(&[SIZE, SIZE], false)
            //             .pow(2))
            //     .sqrt()
            //     .mul(f64::consts::PI * 2.0)
            //     .clamp_min(1.0)
            //     .reciprocal();

            //     [
            //         coef.i((0..64, 0..64)) * 8.0,
            //         coef.i((0..32, 0..32)) * 4.0,
            //         coef.i((0..16, 0..16)) * 2.0,
            //         coef.i((0..8, 0..8)) * 1.0,
            //         coef.i((0..4, 0..4)) * 1.0,
            //         coef.i((0..2, 0..2)) * 1.0,
            //     ]
            // };

            let scaling_coefs = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0];

            let forward_fn = Box::new(
                move |input: &[&DenseDetectionTensorList],
                      train: bool,
                      with_artifacts: bool|
                      -> Result<_> {
                    // sanity checks
                    ensure!(input.len() == input_len);
                    ensure!(input.iter().all(|list| list.tensors.len() == 1));
                    ensure!(input
                        .iter()
                        .map(|list| &list.tensors)
                        .flatten()
                        .all(|tensor| tensor.num_classes() == num_classes));
                    debug_assert!(input.iter().any(|&list| list.is_all_finite()));

                    // let device = input[0].device();
                    let bsize = input[0].batch_size() as i64;
                    let in_h = input[0].tensors[0].height() as i64;
                    let in_w = input[0].tensors[0].width() as i64;
                    let anchors = &input[0].tensors[0].anchors;
                    let num_anchors = anchors.len() as i64;
                    let num_classes = num_classes as i64;

                    ensure!(input
                        .iter()
                        .flat_map(|list| &list.tensors)
                        .all(|det| det.height() == in_h as usize
                            && det.width() == in_w as usize
                            && &det.anchors == anchors));

                    // unpack last input detection
                    let last_det = &input.last().unwrap().tensors[0];
                    let cy_ratio = &last_det.cy;
                    let cx_ratio = &last_det.cx;
                    let h_ratio = &last_det.h;
                    let w_ratio = &last_det.w;
                    let obj_logit = &last_det.obj_logit;
                    let class_logit = &last_det.class_logit;

                    // merge input sequence into a context tensor
                    let context = {
                        let context_vec: Vec<_> = input
                            .iter()
                            .flat_map(|list| &list.tensors)
                            .map(|det| -> Result<_> {
                                let (context, _) = det.encode_detection()?;
                                // let (bsize, num_entires, num_anchors, height, width) = context.size5().unwrap();

                                // scale batch_size to bsize * num_anchors
                                let context = context.repeat_interleave_self_int(num_anchors, 0);

                                // merge entry and anchor dimensions
                                let context = context.view([bsize * num_anchors, -1, in_h, in_w]);

                                Ok(context)
                            })
                            .try_collect()?;
                        Tensor::cat(&context_vec, 1)
                    };
                    debug_assert!(context.is_all_finite());

                    // generate features
                    let features: Vec<_> = {
                        feature_maps
                            .iter()
                            .scan(context, |prev, map| {
                                let next = map.forward_t(prev, train).lrelu();
                                *prev = next.shallow_clone();
                                Some(next)
                            })
                            .collect()
                    };

                    // generate down sampled objectness images
                    let down_sampled_obj_logits = {
                        let obj_logit = obj_logit.view([bsize * num_anchors, 1, in_h, in_w]);

                        let latter = (1..6).scan(obj_logit.shallow_clone(), |prev, _| {
                            let next = prev.slice(2, None, None, 2).slice(3, None, None, 2);
                            *prev = next.shallow_clone();
                            Some(next)
                        });

                        velcro::vec![obj_logit, ..latter]
                    };

                    // generate motion potentials
                    let cograd = |xs: &Tensor| -> Tensor {
                        let (dx, dy) = xs.spatial_gradient();
                        Tensor::cat(&[-dy, dx], 1)
                    };

                    let motion_potential_pixel = {
                        let mut tuples = izip!(
                            features.iter().rev(),
                            motion_maps.iter().rev(),
                            scaling_coefs.iter().cloned().rev(),
                            down_sampled_obj_logits.iter().rev(),
                        );
                        let (feature, map, scaling_coef, _) = tuples.next().unwrap();
                        let init_potential =
                            gaussian_blur.forward(&map.forward_t(feature, train)) * scaling_coef;
                        // let init_potential =
                        //     init_potential.clamp_spectrum(-1.0, 1.0) * spectral_coef;
                        // let init_motion = cograd(&init_potential);

                        tuples.fold(
                            init_potential,
                            |prev_potential_pixel, (feature, map, scaling_coef, obj_logit)| {
                                let (_, _, prev_h, prev_w) = prev_potential_pixel.size4().unwrap();
                                let next_h = prev_h * 2;
                                let next_w = prev_w * 2;

                                // upsample motion from prev step
                                let prev_potential_pixel = prev_potential_pixel
                                    .upsample_bilinear2d(&[next_h, next_w], false, None, None);
                                let prev_field_pixel = cograd(&prev_potential_pixel);

                                let prev_field_ratio = {
                                    // motion vector in ratio unit
                                    let dx_pixel = prev_field_pixel.i((.., 0..1, .., ..));
                                    let dy_pixel = prev_field_pixel.i((.., 1..2, .., ..));

                                    // scale [0..in_h, 0..in_w] = [0..1, 0..1]
                                    let dx_ratio = &dx_pixel / in_w as f64;
                                    let dy_ratio = &dy_pixel / in_h as f64;
                                    Tensor::cat(&[&dx_ratio, &dy_ratio], 1)
                                };

                                let warped_obj_logit = WarpInit::default()
                                    .build(&prev_field_ratio)
                                    .unwrap()
                                    .forward(obj_logit);
                                let corr =
                                    obj_logit.partial_correlation_2d(&warped_obj_logit, [5, 5]);

                                // cat prev motion to curr feature
                                let xs = Tensor::cat(&[feature, &corr], 1);

                                // predict potential
                                let addition = gaussian_blur.forward(&map.forward_t(&xs, train))
                                    * scaling_coef;
                                // dbg!(addition.size(), spectral_coef.size());
                                // let addition = addition.clamp_spectrum(
                                //     -1.0,
                                //     1.0,
                                // );
                                let next_potential = prev_potential_pixel + addition;

                                next_potential
                            },
                        )
                    };
                    let motion_field_pixel = cograd(&motion_potential_pixel);
                    debug_assert!(motion_field_pixel.is_all_finite());

                    // motion vector in ratio unit
                    let motion_dx_pixel = motion_field_pixel.i((.., 0..1, .., ..));
                    let motion_dy_pixel = motion_field_pixel.i((.., 1..2, .., ..));

                    // scale [0..in_h, 0..in_w] = [0..1, 0..1]
                    let motion_dx_ratio = &motion_dx_pixel / in_w as f64;
                    let motion_dy_ratio = &motion_dy_pixel / in_h as f64;
                    let motion_field_ratio = Tensor::cat(&[&motion_dx_ratio, &motion_dy_ratio], 1);

                    // dbg!((motion_potential_pixel.max(), motion_potential_pixel.min()));
                    // dbg!((motion_field_pixel.max(), motion_field_pixel.min()));
                    // dbg!((motion_field_ratio.max(), motion_field_ratio.min()));

                    // compute grid, where each value range in [-1, 1]
                    let warp = WarpInit::default().build(&motion_field_ratio).unwrap();

                    // let grid = {
                    //     let ident_grid = {
                    //         let theta = Tensor::from_cv([[[1f32, 0.0, 0.0], [0.0, 1.0, 0.0]]])
                    //             .expand(&[bsize, 2, 3], false)
                    //             .to_device(device);
                    //         Tensor::affine_grid_generator(&theta, &[bsize, 1, in_h, in_w], false)
                    //     };

                    //     // sample_grid defines boundaries in [-1, 1], while our motion
                    //     // vector defines boundaries in [0, 1].
                    //     &motion_field_ratio.permute(&[0, 2, 3, 1]) * 2.0 + ident_grid
                    // };

                    // let DenormedDetection {
                    //     cy: cy_denorm,
                    //     cx: cx_denorm,
                    //     h: h_denorm,
                    //     w: w_denorm,
                    //     obj_prob,
                    //     class_prob,
                    //     anchors,
                    // } = (&input.last().unwrap().tensors[0]).into();

                    // merge batch and anchor dimensions
                    let cy_ratio = cy_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let cx_ratio = cx_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let h_ratio = h_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let w_ratio = w_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let obj_logit = obj_logit
                        .permute(&[0, 2, 1, 3, 4])
                        .view([-1, 1, in_h, in_w]);
                    let class_logit =
                        class_logit
                            .permute(&[0, 2, 1, 3, 4])
                            .view([-1, num_classes, in_h, in_w]);

                    // warp values
                    let cy_ratio = warp.forward(&(&cy_ratio + &motion_dy_ratio));
                    let cx_ratio = warp.forward(&(&cx_ratio + &motion_dx_ratio));
                    let h_ratio = warp.forward(&h_ratio);
                    let w_ratio = warp.forward(&w_ratio);
                    let obj_logit = warp.forward(&obj_logit);
                    let class_logit = warp.forward(&class_logit);

                    // split batch and anchor dimensions
                    let cy_ratio = cy_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let cx_ratio = cx_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let h_ratio = h_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let w_ratio = w_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let obj_logit = obj_logit
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let class_logit = class_logit
                        .view([bsize, num_anchors, num_classes, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);

                    let warped_det = DenseDetectionTensorUnchecked {
                        cy: cy_ratio,
                        cx: cx_ratio,
                        h: h_ratio,
                        w: w_ratio,
                        obj_logit,
                        class_logit,
                        anchors: anchors.clone(),
                    }
                    .build()
                    .unwrap();

                    let artifacts = with_artifacts.then(|| msg::TransformerArtifacts {
                        motion_potential: Some(motion_potential_pixel),
                        motion_field: Some(motion_field_pixel),
                    });

                    let output: DenseDetectionTensorList = DenseDetectionTensorListUnchecked {
                        tensors: vec![warped_det],
                    }
                    .try_into()
                    .unwrap();

                    Ok((output, artifacts))
                },
            );

            Ok(PotentialBasedTransformer {
                input_len,
                forward_fn,
            })
        }
    }

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct PotentialBasedTransformer {
        input_len: usize,
        #[derivative(Debug = "ignore")]
        forward_fn: Box<
            dyn Fn(
                    &[&DenseDetectionTensorList],
                    bool,
                    bool,
                )
                    -> Result<(DenseDetectionTensorList, Option<msg::TransformerArtifacts>)>
                + Send,
        >,
    }

    impl PotentialBasedTransformer {
        pub fn input_len(&self) -> usize {
            self.input_len
        }

        pub fn forward_t(
            &self,
            input: &[impl Borrow<DenseDetectionTensorList>],
            train: bool,
            with_artifacts: bool,
        ) -> Result<(DenseDetectionTensorList, Option<msg::TransformerArtifacts>)> {
            let input: Vec<_> = input.iter().map(|list| list.borrow()).collect();
            (self.forward_fn)(&input, train, with_artifacts)
        }
    }
}

mod motion_based {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct MotionBasedTransformerInit {
        pub ksize: usize,
        pub norm_kind: NormKind,
        pub padding_kind: PaddingKind,
    }

    impl Default for MotionBasedTransformerInit {
        fn default() -> Self {
            Self {
                padding_kind: PaddingKind::Reflect,
                norm_kind: NormKind::InstanceNorm,
                ksize: 3,
            }
        }
    }

    impl MotionBasedTransformerInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            input_len: usize,
            num_classes: usize,
            inner_c: usize,
        ) -> Result<MotionBasedTransformer> {
            // const BORDER_SIZE_RATIO: f64 = 4.0 / 64.0;

            let path = path.borrow();
            let Self {
                ksize,
                norm_kind,
                padding_kind,
            } = self;
            // let device = path.device();
            let in_c = 5 + num_classes;
            ensure!(input_len > 0);

            let ctx_c = in_c * input_len;

            let feature_maps: Vec<_> = array::IntoIter::new([
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 4,
                    num_mid: 1,
                    num_up: 4,
                }
                .build(path / "feature_0", ctx_c, inner_c, inner_c), // 64 -> 64
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 4,
                    num_mid: 1,
                    num_up: 3,
                }
                .build(path / "feature_1", inner_c, inner_c, inner_c), // 64 -> 32
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 3,
                    num_mid: 1,
                    num_up: 2,
                }
                .build(path / "feature_2", inner_c, inner_c, inner_c), // 32 -> 16
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 2,
                    num_mid: 1,
                    num_up: 1,
                }
                .build(path / "feature_3", inner_c, inner_c, inner_c), // 16 -> 8
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 1,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "feature_4", inner_c, inner_c, inner_c), // 8 -> 4
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 1,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "feature_5", inner_c, inner_c, inner_c), // 4 -> 2
            ])
            .try_collect()
            .unwrap();

            let motion_maps: Vec<_> = array::IntoIter::new([
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 4,
                    num_mid: 1,
                    num_up: 4,
                }
                .build(path / "motion_0", inner_c + 1, 2, inner_c), // 64
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 3,
                    num_mid: 1,
                    num_up: 3,
                }
                .build(path / "motion_1", inner_c + 1, 2, inner_c), // 32
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 3,
                    num_mid: 1,
                    num_up: 3,
                }
                .build(path / "motion_2", inner_c + 1, 2, inner_c), // 16
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 2,
                    num_mid: 1,
                    num_up: 2,
                }
                .build(path / "motion_3", inner_c + 1, 2, inner_c), // 8
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 0,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "motion_4", inner_c + 1, 2, inner_c), // 4
                resnet::ResnetInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_down: 0,
                    num_mid: 1,
                    num_up: 0,
                }
                .build(path / "motion_5", inner_c, 2, inner_c), // 2
            ])
            .try_collect()
            .unwrap();

            let forward_fn = Box::new(
                move |input: &[&DenseDetectionTensorList],
                      train: bool,
                      with_artifacts: bool|
                      -> Result<_> {
                    // sanity checks
                    ensure!(input.len() == input_len);
                    ensure!(input.iter().all(|list| list.tensors.len() == 1));
                    ensure!(input
                        .iter()
                        .map(|list| &list.tensors)
                        .flatten()
                        .all(|tensor| tensor.num_classes() == num_classes));
                    debug_assert!(input.iter().any(|&list| list.is_all_finite()));

                    let device = input[0].device();
                    let bsize = input[0].batch_size() as i64;
                    let in_h = input[0].tensors[0].height() as i64;
                    let in_w = input[0].tensors[0].width() as i64;
                    let anchors = &input[0].tensors[0].anchors;
                    let num_anchors = anchors.len() as i64;
                    let num_classes = num_classes as i64;

                    ensure!(input
                        .iter()
                        .flat_map(|list| &list.tensors)
                        .all(|det| det.height() == in_h as usize
                            && det.width() == in_w as usize
                            && &det.anchors == anchors));

                    // merge input sequence into a context tensor
                    let context = {
                        let context_vec: Vec<_> = input
                            .iter()
                            .flat_map(|list| &list.tensors)
                            .map(|det| -> Result<_> {
                                let (context, _) = det.encode_detection()?;
                                // let (bsize, num_entires, num_anchors, height, width) = context.size5().unwrap();

                                // scale batch_size to bsize * num_anchors
                                let context = context.repeat_interleave_self_int(num_anchors, 0);

                                // merge entry and anchor dimensions
                                let context = context.view([bsize * num_anchors, -1, in_h, in_w]);

                                Ok(context)
                            })
                            .try_collect()?;
                        Tensor::cat(&context_vec, 1)
                    };
                    debug_assert!(context.is_all_finite());

                    // generate features
                    let features: Vec<_> = {
                        feature_maps
                            .iter()
                            .scan(context, |prev, map| {
                                let next = map.forward_t(prev, train).lrelu();
                                *prev = next.shallow_clone();
                                Some(next)
                            })
                            .collect()
                    };

                    // generate motion potentials
                    const SCALE: f64 = 0.01;

                    let motion_field = {
                        let mut pairs = izip!(features.iter().rev(), motion_maps.iter().rev());
                        let (feature, map) = pairs.next().unwrap();
                        let init_field = map.forward_t(feature, train).div(SCALE).tanh().mul(SCALE);

                        pairs.fold(init_field, |prev_field, (feature, map)| {
                            let (_, _, prev_h, prev_w) = prev_field.size4().unwrap();
                            let next_h = prev_h * 2;
                            let next_w = prev_w * 2;

                            // upsample motion from prev step
                            let prev_field =
                                prev_field.upsample_nearest2d(&[next_h, next_w], None, None);

                            // cat prev motion to curr feature
                            let xs = Tensor::cat(&[feature, &prev_field], 1);

                            // predict potential
                            let next_field =
                                prev_field + map.forward_t(&xs, train).div(SCALE).tanh().mul(SCALE);

                            next_field
                        })
                    };
                    debug_assert!(motion_field.is_all_finite());

                    // motion vector in ratio unit
                    let motion_dx = motion_field.i((.., 0..1, .., ..));
                    let motion_dy = motion_field.i((.., 1..2, .., ..));

                    // dbg!((motion_field.max(), motion_field.min()));

                    // compute grid, where each value range in [-1, 1]
                    let grid = {
                        let ident_grid = {
                            let theta = Tensor::from_cv([[[1f32, 0.0, 0.0], [0.0, 1.0, 0.0]]])
                                .expand(&[bsize, 2, 3], false)
                                .to_device(device);
                            Tensor::affine_grid_generator(&theta, &[bsize, 1, in_h, in_w], false)
                        };

                        // sample_grid defines boundaries in [-1, 1], while our motion
                        // vector defines boundaries in [0, 1].
                        &motion_field.permute(&[0, 2, 3, 1]) * 2.0 + ident_grid
                    };

                    // unpack last input detection
                    let last_det = &input.last().unwrap().tensors[0];
                    let cy_ratio = &last_det.cy;
                    let cx_ratio = &last_det.cx;
                    let h_ratio = &last_det.h;
                    let w_ratio = &last_det.w;
                    let obj_logit = &last_det.obj_logit;
                    let class_logit = &last_det.class_logit;

                    // merge batch and anchor dimensions
                    let cy_ratio = cy_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let cx_ratio = cx_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let h_ratio = h_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let w_ratio = w_ratio.permute(&[0, 2, 1, 3, 4]).view([-1, 1, in_h, in_w]);
                    let obj_logit = obj_logit
                        .permute(&[0, 2, 1, 3, 4])
                        .view([-1, 1, in_h, in_w]);
                    let class_logit =
                        class_logit
                            .permute(&[0, 2, 1, 3, 4])
                            .view([-1, num_classes, in_h, in_w]);

                    // warp values
                    let cy_ratio = (&cy_ratio + &motion_dy).grid_sampler(
                        &grid, // grid
                        1,     // nearest interpolation
                        0,     // pad zeros
                        false,
                    );
                    let cx_ratio = (&cx_ratio + &motion_dx).grid_sampler(
                        &grid, // grid
                        1,     // nearest interpolation
                        0,     // pad zeros
                        false,
                    );
                    let h_ratio = h_ratio.grid_sampler(
                        &grid, // grid
                        1,     // nearest interpolation
                        0,     // pad zeros
                        false,
                    );
                    let w_ratio = w_ratio.grid_sampler(
                        &grid, // grid
                        1,     // nearest interpolation
                        0,     // pad zeros
                        false,
                    );
                    let obj_logit = obj_logit.grid_sampler(
                        &grid, // grid
                        1,     // nearest interpolation
                        0,     // pad zeros
                        false,
                    );
                    let class_logit = class_logit.grid_sampler(
                        &grid, // grid
                        1,     // nearest interpolation
                        0,     // pad zeros
                        false,
                    );

                    // split batch and anchor dimensions
                    let cy_ratio = cy_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let cx_ratio = cx_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let h_ratio = h_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let w_ratio = w_ratio
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let obj_logit = obj_logit
                        .view([bsize, num_anchors, 1, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);
                    let class_logit = class_logit
                        .view([bsize, num_anchors, num_classes, in_h, in_w])
                        .permute(&[0, 2, 1, 3, 4]);

                    let warped_det = DenseDetectionTensorUnchecked {
                        cy: cy_ratio,
                        cx: cx_ratio,
                        h: h_ratio,
                        w: w_ratio,
                        obj_logit,
                        class_logit,
                        anchors: anchors.clone(),
                    }
                    .build()
                    .unwrap();

                    let artifacts = with_artifacts.then(|| msg::TransformerArtifacts {
                        motion_potential: None,
                        motion_field: Some(motion_field),
                    });

                    let output: DenseDetectionTensorList = DenseDetectionTensorListUnchecked {
                        tensors: vec![warped_det],
                    }
                    .try_into()
                    .unwrap();

                    Ok((output, artifacts))
                },
            );

            Ok(MotionBasedTransformer {
                input_len,
                forward_fn,
            })
        }
    }

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct MotionBasedTransformer {
        input_len: usize,
        #[derivative(Debug = "ignore")]
        forward_fn: Box<
            dyn Fn(
                    &[&DenseDetectionTensorList],
                    bool,
                    bool,
                )
                    -> Result<(DenseDetectionTensorList, Option<msg::TransformerArtifacts>)>
                + Send,
        >,
    }

    impl MotionBasedTransformer {
        pub fn input_len(&self) -> usize {
            self.input_len
        }

        pub fn forward_t(
            &self,
            input: &[impl Borrow<DenseDetectionTensorList>],
            train: bool,
            with_artifacts: bool,
        ) -> Result<(DenseDetectionTensorList, Option<msg::TransformerArtifacts>)> {
            let input: Vec<_> = input.iter().map(|list| list.borrow()).collect();
            (self.forward_fn)(&input, train, with_artifacts)
        }
    }
}

mod resnet {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ResnetInit {
        pub padding_kind: PaddingKind,
        pub norm_kind: NormKind,
        pub ksize: usize,
        pub num_down: usize,
        pub num_mid: usize,
        pub num_up: usize,
    }

    impl Default for ResnetInit {
        fn default() -> Self {
            Self {
                padding_kind: PaddingKind::Reflect,
                norm_kind: NormKind::InstanceNorm,
                ksize: 5,
                num_down: 0,
                num_mid: 3,
                num_up: 0,
            }
        }
    }

    impl ResnetInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            inner_c: usize,
        ) -> Result<Resnet> {
            let path = path.borrow();
            let Self {
                padding_kind,
                norm_kind,
                ksize,
                num_down,
                num_mid,
                num_up,
            } = self;
            let in_c = in_c as i64;
            let out_c = out_c as i64;
            let inner_c = inner_c as i64;
            let bias = norm_kind == NormKind::InstanceNorm;
            let padding = ksize / 2;
            let ksize = ksize as i64;
            let total_blocks = num_down + num_mid + num_up;

            ensure!(in_c > 0);
            ensure!(out_c > 0);
            ensure!(inner_c > 0);
            ensure!(total_blocks > 0);

            // first block
            let seq = {
                let path = path / "initial_block";

                nn::seq_t()
                    .add(nn::conv2d(
                        &path / "conv",
                        in_c,
                        inner_c,
                        1,
                        nn::ConvConfig {
                            stride: 1,
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c))
                    .add_fn(|xs| xs.lrelu())
                    .inspect(|xs| {
                        debug_assert!(xs.is_all_finite());
                    })
            };

            // down sampling blocks
            let seq = (0..num_down).fold(seq, |seq, index| {
                let path = path / format!("block_{}", index);

                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    inner_c,
                    ksize,
                    nn::ConvConfig {
                        stride: 2,
                        padding: padding as i64,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.lrelu())
                .inspect(|xs| {
                    debug_assert!(xs.is_all_finite());
                })
            });

            // resnet blocks
            let seq = (0..num_mid).fold(seq, |seq, index| {
                let path = path / format!("block_{}", index + num_down);

                let branch = nn::seq_t()
                    // first part
                    .add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / "conv1",
                        inner_c,
                        inner_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm1", inner_c))
                    .add_fn(|xs| xs.lrelu())
                    .inspect(|xs| {
                        debug_assert!(xs.is_all_finite());
                    })
                    // second part
                    .add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / "conv2",
                        inner_c,
                        inner_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm2", inner_c))
                    .inspect(|xs| {
                        debug_assert!(xs.is_all_finite());
                    })
                    .add_fn(|xs| xs.lrelu())
                    .inspect(|xs| {
                        debug_assert!(xs.is_all_finite());
                    });

                // addition
                seq.add_fn_t(move |xs, train| xs + branch.forward_t(xs, train))
                    .inspect(|xs| {
                        debug_assert!(xs.is_all_finite());
                    })
            });

            // up sampling blocks
            let seq = (0..num_up).fold(seq, |seq, index| {
                let path = path / format!("block_{}", index + num_down + num_mid);

                seq.add_fn(|xs| {
                    let (_, _, h, w) = xs.size4().unwrap();
                    xs.upsample_nearest2d(&[h * 2, w * 2], None, None)
                })
                .inspect(|xs| {
                    debug_assert!(xs.is_all_finite());
                })
                .add(padding_kind.build([padding, padding, padding, padding]))
                .inspect(|xs| {
                    debug_assert!(xs.is_all_finite());
                })
                .add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    inner_c,
                    ksize,
                    nn::ConvConfig {
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .inspect(|xs| {
                    debug_assert!(xs.is_all_finite());
                })
                .add(norm_kind.build(&path / "norm", inner_c))
                .inspect(|xs| {
                    debug_assert!(xs.is_all_finite());
                })
                .add_fn(|xs| xs.lrelu())
                .inspect(|xs| {
                    debug_assert!(xs.is_all_finite());
                })
            });

            // last block
            let seq = {
                let path = path / "last_block";

                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    out_c,
                    1,
                    nn::ConvConfig {
                        stride: 1,
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", out_c))
                .inspect(|xs| {
                    debug_assert!(xs.is_all_finite());
                })
            };

            Ok(Resnet { seq })
        }
    }

    #[derive(Debug)]
    pub struct Resnet {
        seq: nn::SequentialT,
    }

    impl nn::ModuleT for Resnet {
        fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            self.seq.forward_t(xs, train)
        }
    }
}

mod detection_encode {
    use super::*;

    pub trait DetectionEncode {
        fn encode_detection(&self) -> Result<(Tensor, Vec<RatioSize<R64>>)>;
        fn decode_detection(input: &Tensor, anchors: Vec<RatioSize<R64>>) -> Self;
    }

    impl DetectionEncode for DenseDetectionTensor {
        fn encode_detection(&self) -> Result<(Tensor, Vec<RatioSize<R64>>)> {
            let input = self;
            let device = input.device();
            let bsize = input.batch_size() as i64;
            let num_classes = input.num_classes() as i64;
            let in_h = input.height() as i64;

            let in_w = input.width() as i64;
            let num_anchors = input.anchors.len() as i64;

            let DetectionAdditions {
                y_offsets,
                x_offsets,
                anchor_heights,
                anchor_widths,
            } = DetectionAdditions::new(input.height(), input.width(), &input.anchors)
                .to_device(device);

            let merge = {
                // let wtf = (&input.cy * in_h as f64 - &y_offsets + 0.5) / 2.0;
                // dbg!(input.cy.max());
                // dbg!(input.cy.min());
                // dbg!(wtf.max());
                // dbg!(wtf.min());
                // debug_assert!(bool::from(input.h.ge(0.0).all()));
                // debug_assert!(bool::from(input.w.ge(0.0).all()));

                let cy_logit = ((&input.cy * in_h as f64 - &y_offsets + 0.5) / 2.0)
                    .logit(1e-5)
                    .view([bsize, 1, num_anchors, in_h, in_w]);
                let cx_logit = ((&input.cx * in_w as f64 - &x_offsets + 0.5) / 2.0)
                    .logit(1e-5)
                    .view([bsize, 1, num_anchors, in_h, in_w]);
                let h_logit = ((&input.h / anchor_heights).sqrt() / 2.0)
                    .logit(1e-5)
                    .view([bsize, 1, num_anchors, in_h, in_w]);
                let w_logit = ((&input.w / anchor_widths).sqrt() / 2.0).logit(1e-5).view([
                    bsize,
                    1,
                    num_anchors,
                    in_h,
                    in_w,
                ]);
                let obj_logit = input.obj_logit.view([bsize, 1, num_anchors, in_h, in_w]);
                let class_logit =
                    input
                        .class_logit
                        .view([bsize, num_classes as i64, num_anchors, in_h, in_w]);

                // dbg!(cy_logit.min(), cy_logit.max());
                // dbg!(cx_logit.min(), cx_logit.max());
                // dbg!(h_logit.min(), h_logit.max());
                // dbg!(w_logit.min(), w_logit.max());
                // dbg!(obj_logit.min(), obj_logit.max());
                // dbg!(class_logit.min(), class_logit.max());

                // debug_assert!(bool::from(input.h.ge(0.0).all()));
                // debug_assert!(bool::from(input.w.ge(0.0).all()));

                ensure!(cy_logit.is_all_finite());
                ensure!(cx_logit.is_all_finite());
                ensure!(h_logit.is_all_finite());
                ensure!(w_logit.is_all_finite());
                ensure!(obj_logit.is_all_finite());
                ensure!(class_logit.is_all_finite());

                Tensor::cat(
                    &[cy_logit, cx_logit, h_logit, w_logit, obj_logit, class_logit],
                    1,
                )
            };

            // dbg!(merge.min(), merge.max());
            // debug_assert!(merge.is_all_finite());

            Ok((merge, input.anchors.clone()))
        }

        fn decode_detection(input: &Tensor, anchors: Vec<RatioSize<R64>>) -> Self {
            debug_assert!(input.is_all_finite());

            let device = input.device();
            // let num_anchors = anchors.len() as i64;
            let (_, _, _, in_h, in_w) = input.size5().unwrap();

            let DetectionAdditions {
                y_offsets,
                x_offsets,
                anchor_heights,
                anchor_widths,
            } = DetectionAdditions::new(in_h as usize, in_w as usize, &anchors).to_device(device);

            let xs = input;
            let cy =
                ((xs.i((.., 0..1, .., .., ..)).sigmoid() * 2.0 - 0.5) + y_offsets) / in_h as f64;
            let cx =
                ((xs.i((.., 1..2, .., .., ..)).sigmoid() * 2.0 - 0.5) + x_offsets) / in_w as f64;
            let h = xs.i((.., 2..3, .., .., ..)).sigmoid().mul(2.0).pow(2.0) * anchor_heights;
            let w = xs.i((.., 3..4, .., .., ..)).sigmoid().mul(2.0).pow(2.0) * anchor_widths;
            let obj_logit = xs.i((.., 4..5, .., .., ..));
            let class_logit = xs.i((.., 5.., .., .., ..));

            let output = DenseDetectionTensorUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                anchors,
            }
            .build()
            .unwrap();

            debug_assert!(output.is_all_finite());

            output
        }
    }
}

mod denormed_detection {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub struct DenormedDetection {
        pub cy: Tensor,
        pub cx: Tensor,
        pub h: Tensor,
        pub w: Tensor,
        pub obj_prob: Tensor,
        pub class_prob: Tensor,
        #[tensor_like(clone)]
        pub anchors: Vec<RatioSize<R64>>,
    }

    impl DenormedDetection {
        pub fn height(&self) -> usize {
            let (_, _, _, height, _) = self.cy.size5().unwrap();
            height as usize
        }

        pub fn width(&self) -> usize {
            let (_, _, _, _, width) = self.cy.size5().unwrap();
            width as usize
        }
    }

    impl From<&DenseDetectionTensor> for DenormedDetection {
        fn from(input: &DenseDetectionTensor) -> Self {
            let bsize = input.batch_size() as i64;
            let num_classes = input.num_classes() as i64;
            let in_h = input.height() as i64;
            let in_w = input.width() as i64;
            let num_anchors = input.anchors.len() as i64;

            let cy = (&input.cy * in_h as f64).view([bsize, 1, num_anchors, in_h, in_w]);
            let cx = (&input.cx * in_w as f64).view([bsize, 1, num_anchors, in_h, in_w]);
            let h = (&input.h * in_h as f64).view([bsize, 1, num_anchors, in_h, in_w]);
            let w = (&input.w * in_w as f64).view([bsize, 1, num_anchors, in_h, in_w]);
            let obj_prob = input.obj_prob().view([bsize, 1, num_anchors, in_h, in_w]);
            let class_prob =
                input
                    .class_prob()
                    .view([bsize, num_classes as i64, num_anchors, in_h, in_w]);

            debug_assert!(cy.is_all_finite());
            debug_assert!(cx.is_all_finite());
            debug_assert!(h.is_all_finite());
            debug_assert!(w.is_all_finite());
            debug_assert!(obj_prob.is_all_finite());
            debug_assert!(class_prob.is_all_finite());

            Self {
                cy,
                cx,
                h,
                w,
                obj_prob,
                class_prob,
                anchors: input.anchors.clone(),
            }
        }
    }

    impl From<DenseDetectionTensor> for DenormedDetection {
        fn from(input: DenseDetectionTensor) -> Self {
            (&input).into()
        }
    }

    impl From<&DenormedDetection> for DenseDetectionTensor {
        fn from(input: &DenormedDetection) -> Self {
            let in_h = input.height() as i64;
            let in_w = input.width() as i64;
            let DenormedDetection {
                cy,
                cx,
                h,
                w,
                obj_prob,
                class_prob,
                anchors,
            } = input;

            let cy = cy / in_h as f64;
            let cx = cx / in_w as f64;
            let h = h / in_h as f64;
            let w = w / in_w as f64;
            let obj_logit = obj_prob.logit(1e-6);
            let class_logit = class_prob.logit(1e-6);

            debug_assert!(cy.is_all_finite());
            debug_assert!(cx.is_all_finite());
            debug_assert!(h.is_all_finite());
            debug_assert!(w.is_all_finite());
            debug_assert!(obj_logit.is_all_finite());
            debug_assert!(class_logit.is_all_finite());

            let output = DenseDetectionTensorUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                anchors: anchors.clone(),
            }
            .build()
            .unwrap();

            output
        }
    }

    impl From<DenormedDetection> for DenseDetectionTensor {
        fn from(input: DenormedDetection) -> Self {
            (&input).into()
        }
    }
}

#[derive(Debug, TensorLike)]
struct DetectionAdditions {
    pub y_offsets: Tensor,
    pub x_offsets: Tensor,
    pub anchor_heights: Tensor,
    pub anchor_widths: Tensor,
}

impl DetectionAdditions {
    pub fn new(height: usize, width: usize, anchors: &[RatioSize<R64>]) -> Self {
        let height = height as i64;
        let width = width as i64;
        let num_anchors = anchors.len() as i64;

        let y_offsets = Tensor::arange(height, FLOAT_CPU)
            .set_requires_grad(false)
            .view([1, 1, 1, height, 1]);
        let x_offsets = Tensor::arange(width, FLOAT_CPU)
            .set_requires_grad(false)
            .view([1, 1, 1, 1, width]);
        let (anchor_h_vec, anchor_w_vec) = anchors
            .iter()
            .cloned()
            .map(|anchor_size| {
                let anchor_size = anchor_size.cast::<f32>().unwrap();
                (anchor_size.h, anchor_size.w)
            })
            .unzip_n_vec();

        let anchor_heights = Tensor::of_slice(&anchor_h_vec)
            .set_requires_grad(false)
            .view([1, 1, num_anchors, 1, 1]);
        let anchor_widths = Tensor::of_slice(&anchor_w_vec)
            .set_requires_grad(false)
            .view([1, 1, num_anchors, 1, 1]);

        DetectionAdditions {
            x_offsets,
            y_offsets,
            anchor_heights,
            anchor_widths,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_encode_decode() {
        tch::no_grad(|| {
            let bsize = 4;
            let num_classes = 10;
            let in_h = 16;
            let in_w = 9;
            let anchors: Vec<_> = [(0.5, 0.1), (0.1, 0.6)]
                .iter()
                .map(|&(h, w)| RatioSize::from_hw(r64(h), r64(w)).unwrap())
                .collect();
            let num_anchors = anchors.len() as i64;
            let y_offsets = Tensor::arange(in_h, FLOAT_CPU)
                .set_requires_grad(false)
                .view([1, 1, 1, in_h, 1]);
            let x_offsets = Tensor::arange(in_w, FLOAT_CPU)
                .set_requires_grad(false)
                .view([1, 1, 1, 1, in_w]);
            let (anchor_h_vec, anchor_w_vec) = anchors
                .iter()
                .cloned()
                .map(|anchor_size| {
                    let anchor_size = anchor_size.cast::<f32>().unwrap();
                    (anchor_size.h, anchor_size.w)
                })
                .unzip_n_vec();
            let anchor_heights = Tensor::of_slice(&anchor_h_vec)
                .set_requires_grad(false)
                .to_device(Device::Cpu)
                .view([1, 1, num_anchors, 1, 1]);
            let anchor_widths = Tensor::of_slice(&anchor_w_vec)
                .set_requires_grad(false)
                .to_device(Device::Cpu)
                .view([1, 1, num_anchors, 1, 1]);

            let cy = (Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 2.0 - 0.5
                + y_offsets)
                / in_h as f64;
            let cx = (Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 2.0 - 0.5
                + x_offsets)
                / in_w as f64;
            let h = Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU)
                * 4.0
                * anchor_heights;
            let w =
                Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 4.0 * anchor_widths;
            let obj_logit = Tensor::randn(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 0.5;
            let class_logit =
                Tensor::randn(&[bsize, num_classes, num_anchors, in_h, in_w], FLOAT_CPU) * 0.5;

            let orig: DenseDetectionTensor = DenseDetectionTensorUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                anchors: anchors.clone(),
            }
            .try_into()
            .unwrap();

            let (tensor, _anchors) = orig.encode_detection().unwrap();
            let recon = DenseDetectionTensor::decode_detection(&tensor, anchors);

            let cy_diff: f64 = (&orig.cy - &recon.cy).abs().max().into();
            let cx_diff: f64 = (&orig.cx - &recon.cx).abs().max().into();
            let h_diff: f64 = (&orig.h - &recon.h).abs().max().into();
            let w_diff: f64 = (&orig.w - &recon.w).abs().max().into();
            let obj_diff: f64 = (&orig.obj_logit - &recon.obj_logit).abs().max().into();
            let class_diff: f64 = (&orig.class_logit - &recon.class_logit).abs().max().into();

            assert_abs_diff_eq!(cy_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(cx_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(h_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(w_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(obj_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(class_diff, 0.0, epsilon = 1e-6);
        });
    }

    #[test]
    fn detection_denorm() {
        tch::no_grad(|| {
            let bsize = 4;
            let num_classes = 10;
            let in_h = 16;
            let in_w = 9;
            let anchors: Vec<_> = [(0.5, 0.1), (0.1, 0.6)]
                .iter()
                .map(|&(h, w)| RatioSize::from_hw(r64(h), r64(w)).unwrap())
                .collect();
            let num_anchors = anchors.len() as i64;
            let y_offsets = Tensor::arange(in_h, FLOAT_CPU)
                .set_requires_grad(false)
                .view([1, 1, 1, in_h, 1]);
            let x_offsets = Tensor::arange(in_w, FLOAT_CPU)
                .set_requires_grad(false)
                .view([1, 1, 1, 1, in_w]);
            let (anchor_h_vec, anchor_w_vec) = anchors
                .iter()
                .cloned()
                .map(|anchor_size| {
                    let anchor_size = anchor_size.cast::<f32>().unwrap();
                    (anchor_size.h, anchor_size.w)
                })
                .unzip_n_vec();
            let anchor_heights = Tensor::of_slice(&anchor_h_vec)
                .set_requires_grad(false)
                .to_device(Device::Cpu)
                .view([1, 1, num_anchors, 1, 1]);
            let anchor_widths = Tensor::of_slice(&anchor_w_vec)
                .set_requires_grad(false)
                .to_device(Device::Cpu)
                .view([1, 1, num_anchors, 1, 1]);

            let cy = (Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 2.0 - 0.5
                + y_offsets)
                / in_h as f64;
            let cx = (Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 2.0 - 0.5
                + x_offsets)
                / in_w as f64;
            let h = Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU)
                * 4.0
                * anchor_heights;
            let w =
                Tensor::rand(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 4.0 * anchor_widths;
            let obj_logit = Tensor::randn(&[bsize, 1, num_anchors, in_h, in_w], FLOAT_CPU) * 0.5;
            let class_logit =
                Tensor::randn(&[bsize, num_classes, num_anchors, in_h, in_w], FLOAT_CPU) * 0.5;

            let orig: DenseDetectionTensor = DenseDetectionTensorUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                anchors,
            }
            .try_into()
            .unwrap();

            let denormed = DenormedDetection::from(&orig);
            let recon = DenseDetectionTensor::from(&denormed);

            let cy_diff: f64 = (&orig.cy - &recon.cy).abs().max().into();
            let cx_diff: f64 = (&orig.cx - &recon.cx).abs().max().into();
            let h_diff: f64 = (&orig.h - &recon.h).abs().max().into();
            let w_diff: f64 = (&orig.w - &recon.w).abs().max().into();
            let obj_diff: f64 = (&orig.obj_logit - &recon.obj_logit).abs().max().into();
            let class_diff: f64 = (&orig.class_logit - &recon.class_logit).abs().max().into();

            assert_abs_diff_eq!(cy_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(cx_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(h_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(w_diff, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(obj_diff, 0.0, epsilon = 1e-5);
            assert_abs_diff_eq!(class_diff, 0.0, epsilon = 1e-5);
        });
    }

    // #[test]
    // fn wtf() {
    //     let bsize = 8;
    //     let in_c = 3;
    //     let in_h = 5;
    //     let in_w = 5;

    //     for _ in 0..10 {
    //         let orig = Tensor::rand(
    //             &[bsize, in_c, in_h, in_w],
    //             (Kind::Float, Device::cuda_if_available()),
    //         ) * 100.0;

    //         let new = orig.clamp_spectrum(-1.0, 1.0);

    //         let dx = new.dx();
    //         let dy = new.dy();
    //         dbg!(dx.abs().max(), dy.abs().max());
    //     }

    //     panic!();
    // }
}
