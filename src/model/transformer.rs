use super::{
    generator::ResnetGeneratorInit,
    misc::{NormKind, PaddingKind},
};
use crate::common::*;
use tch_modules::GroupNormInit;

#[derive(Debug, Clone)]
pub struct TransformerInit {
    pub ksize: usize,
    pub norm_kind: NormKind,
    pub padding_kind: PaddingKind,
    pub num_resnet_blocks: usize,
    pub num_scaling_blocks: usize,
    pub num_down_sample: usize,
}

impl Default for TransformerInit {
    fn default() -> Self {
        Self {
            padding_kind: PaddingKind::Reflect,
            norm_kind: NormKind::InstanceNorm,
            ksize: 3,
            num_resnet_blocks: 2,
            num_scaling_blocks: 2,
            num_down_sample: 3,
        }
    }
}

impl TransformerInit {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        num_detections: usize,
        num_classes: usize,
        inner_c: usize,
    ) -> Result<Transformer> {
        const BORDER_SIZE_RATIO: f64 = 4.0 / 64.0;

        let path = path.borrow();
        let Self {
            ksize,
            norm_kind,
            padding_kind,
            num_resnet_blocks,
            num_scaling_blocks,
            num_down_sample,
        } = self;
        let device = path.device();
        let in_c = 5 + num_classes;
        let bias = norm_kind == NormKind::InstanceNorm;
        ensure!(num_detections > 0);

        let encoder = {
            let path = path / "encoder";

            let seq = {
                let path = &path / "block_0";
                nn::seq_t()
                    .add(padding_kind.build([0, 0, 0, 0]))
                    .add(nn::conv2d(
                        &path / "conv",
                        in_c as i64,
                        inner_c as i64,
                        1,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c as i64))
                    .add_fn(|xs| xs.lrelu())
            };

            (1..3).fold(seq, |seq, index| {
                let path = &path / format!("block_{}", index);

                seq.add(padding_kind.build([0, 0, 0, 0]))
                    .add(nn::conv2d(
                        &path / "conv",
                        inner_c as i64,
                        inner_c as i64,
                        1,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c as i64))
                    .add_fn(|xs| xs.lrelu())
            })
        };
        let decoder = {
            let path = path / "decoder";
            let num_blocks = 3;

            let seq = (0..(num_blocks - 1)).fold(nn::seq_t(), |seq, index| {
                let path = &path / format!("block_{}", index);

                seq.add(padding_kind.build([0, 0, 0, 0]))
                    .add(nn::conv2d(
                        &path / "conv",
                        inner_c as i64,
                        inner_c as i64,
                        1,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c as i64))
                    .add_fn(|xs| xs.lrelu())
            });

            let seq = {
                let path = &path / format!("block_{}", num_blocks - 1);
                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c as i64,
                    in_c as i64,
                    1,
                    nn::ConvConfig {
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c as i64))
                .add_fn(|xs| xs.lrelu())
            };

            seq
        };

        let branches: Vec<_> = (0..num_detections)
            .map(|index| -> Result<_> {
                let path = path / format!("branch_{}", index);

                let attention_block = TransformerBlockInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_resnet_blocks,
                    num_scaling_blocks,
                    num_down_sample,
                }
                .build(&path / "attention", inner_c, inner_c)?;

                let patch_block = ResnetGeneratorInit {
                    ksize,
                    norm_kind,
                    padding_kind,
                    num_scale_blocks: num_scaling_blocks,
                    num_blocks: num_resnet_blocks,
                    ..Default::default()
                }
                .build(&path / "patch", inner_c, inner_c, inner_c);

                Ok((attention_block, patch_block))
            })
            .try_collect()?;

        let forward_fn = Box::new(
            move |input: &[&DenseDetectionTensorList],
                  train: bool|
                  -> Result<DenseDetectionTensorList> {
                ensure!(input.len() == num_detections);
                ensure!(input.iter().all(|list| list.tensors.len() == 1));
                ensure!(
                    input
                        .iter()
                        .map(|list| &list.tensors)
                        .flatten()
                        .all(|tensor| tensor.num_anchors() == 1
                            && tensor.num_classes() == num_classes)
                );
                let bsize = input[0].batch_size() as i64;

                let tensors = input.iter().map(|list| &list.tensors).flatten();
                let detections: Vec<_> = izip!(tensors, &branches)
                    .map(
                        |(in_detection, (attention_block, patch_block))| -> Result<_> {
                            let in_h = in_detection.height() as i64;
                            let in_w = in_detection.width() as i64;

                            let y_offsets = Tensor::arange(in_h, (Kind::Float, device))
                                .div(in_h as f64)
                                .set_requires_grad(false)
                                .view([1, 1, 1, in_h, 1]);
                            let x_offsets = Tensor::arange(in_w, (Kind::Float, device))
                                .div(in_w as f64)
                                .set_requires_grad(false)
                                .view([1, 1, 1, 1, in_w]);

                            let in_tensor = {
                                let unbiased_cy =
                                    (&in_detection.cy - &y_offsets).view([bsize, 1, in_h, in_w]);
                                let unbiased_cx =
                                    (&in_detection.cx - &x_offsets).view([bsize, 1, in_h, in_w]);
                                let box_h = in_detection.h.view([bsize, 1, in_h, in_w]);
                                let box_w = in_detection.w.view([bsize, 1, in_h, in_w]);
                                let obj_logit = in_detection.obj_logit.view([bsize, 1, in_h, in_w]);
                                let class_logit = in_detection.class_logit.view([
                                    bsize,
                                    num_classes as i64,
                                    in_h,
                                    in_w,
                                ]);

                                Tensor::cat(
                                    &[
                                        unbiased_cy,
                                        unbiased_cx,
                                        box_h,
                                        box_w,
                                        obj_logit,
                                        class_logit,
                                    ],
                                    1,
                                )
                            };
                            let in_context = encoder.forward_t(&in_tensor, train);
                            let shifted = attention_block.forward_t(&in_context, train)?;
                            let patch = patch_block.forward_t(&in_context, train);
                            let patch_mask = {
                                let border_h = (in_h as f64 * BORDER_SIZE_RATIO).floor() as usize;
                                let border_w = (in_w as f64 * BORDER_SIZE_RATIO).floor() as usize;
                                Tensor::from_cv(nd::Array2::from_shape_fn(
                                    [in_h as usize, in_w as usize],
                                    |(row, col)| {
                                        let ok = row < border_h
                                            || row >= (in_h as usize - border_h)
                                            || col < border_w
                                            || col >= (in_w as usize - border_w);
                                        if ok {
                                            1.0
                                        } else {
                                            0.0
                                        }
                                    },
                                ))
                                .set_requires_grad(false)
                                .to_device(device)
                            };
                            let out_context = shifted + patch * patch_mask;
                            let out_tensor = decoder.forward_t(&out_context, train);
                            let out_detection: DenseDetectionTensor = {
                                let xs = out_tensor.view([bsize, -1, 1, in_h, in_w]);
                                let cy = xs.i((.., 0..1, .., .., ..)) + y_offsets;
                                let cx = xs.i((.., 1..2, .., .., ..)) + x_offsets;
                                let h = xs.i((.., 2..3, .., .., ..));
                                let w = xs.i((.., 3..4, .., .., ..));
                                let obj_logit = xs.i((.., 4..5, .., .., ..));
                                let class_logit = xs.i((.., 5.., .., .., ..));

                                DenseDetectionTensorUnchecked {
                                    cy,
                                    cx,
                                    h,
                                    w,
                                    obj_logit,
                                    class_logit,
                                    anchors: in_detection.anchors.clone(),
                                }
                                .try_into()
                                .unwrap()
                            };

                            Ok(out_detection)
                        },
                    )
                    .try_collect()?;

                let output: DenseDetectionTensorList = DenseDetectionTensorListUnchecked {
                    tensors: detections,
                }
                .try_into()
                .unwrap();

                Ok(output)
            },
        );

        Ok(Transformer { forward_fn })
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Transformer {
    #[derivative(Debug = "ignore")]
    forward_fn:
        Box<dyn Fn(&[&DenseDetectionTensorList], bool) -> Result<DenseDetectionTensorList> + Send>,
}

impl Transformer {
    pub fn forward_t(
        &self,
        input: &[impl Borrow<DenseDetectionTensorList>],
        train: bool,
    ) -> Result<DenseDetectionTensorList> {
        let input: Vec<_> = input.iter().map(|list| list.borrow()).collect();
        (self.forward_fn)(&input, train)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlockInit {
    pub ksize: usize,
    pub norm_kind: NormKind,
    pub padding_kind: PaddingKind,
    pub num_resnet_blocks: usize,
    pub num_scaling_blocks: usize,
    pub num_down_sample: usize,
}

impl TransformerBlockInit {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_c: usize,
        inner_c: usize,
    ) -> Result<TransformerBlock> {
        let path = path.borrow();
        let Self {
            ksize,
            norm_kind,
            padding_kind,
            num_resnet_blocks,
            num_scaling_blocks,
            num_down_sample,
        } = self;
        let bias = norm_kind == NormKind::InstanceNorm;

        let resnet_init = ResnetGeneratorInit {
            ksize,
            norm_kind,
            padding_kind,
            num_scale_blocks: num_scaling_blocks,
            num_blocks: num_resnet_blocks,
            ..Default::default()
        };

        let input_norm =
            GroupNormInit::default().build(path / "input_norm", inner_c as i64, inner_c as i64);

        let query_transform =
            resnet_init
                .clone()
                .build(path / "query_transform", in_c, inner_c, inner_c);
        let key_transform = resnet_init.build(path / "key_transform", in_c, inner_c, inner_c);

        let query_down_sample = {
            let path = path / "query_down_sample";
            let padding = ksize / 2;

            (0..num_down_sample).fold(nn::seq_t(), |seq, index| {
                let path = &path / format!("block_{}", index + 1);

                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c as i64,
                    inner_c as i64,
                    ksize as i64,
                    nn::ConvConfig {
                        stride: 2,
                        padding: padding as i64,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c as i64))
                .add_fn(|xs| xs.lrelu())
            })
        };

        let forward_fn = Box::new(move |input: &Tensor, train: bool| -> Result<Tensor> {
            let (bsize, _, in_h, in_w) = input.size4()?;
            let patch_h = in_h / 2i64.pow(num_down_sample as u32);
            let patch_w = in_w / 2i64.pow(num_down_sample as u32);

            let value = input;
            let value_norm = input_norm.forward_t(input, train);
            let key = key_transform.forward_t(&value_norm, train);
            let query = {
                let xs = query_transform.forward_t(&value_norm, train);
                query_down_sample.forward_t(&xs, train)
            };

            let attention = Tensor::einsum(
                "bcq,bck->bqk",
                &[
                    query.view([bsize, inner_c as i64, -1]),
                    key.view([bsize, inner_c as i64, -1]),
                ],
            )
            .div((inner_c as f64).sqrt())
            .softmax(1, Kind::Float)
            .view([bsize, patch_h * patch_w, in_h * in_w]);

            let patches = Tensor::einsum(
                "bqk,bck->bcqk",
                &[attention, value.view([bsize, inner_c as i64, -1])],
            );

            let output = patches
                .view([bsize, inner_c as i64 * patch_h * patch_w, in_h * in_w])
                .col2im(
                    &[in_h, in_w],
                    &[patch_h, patch_w],
                    &[1, 1], // dilation
                    &[patch_h / 2, patch_w / 2],
                    &[1, 1], // stride
                );

            Ok(output)
        });

        Ok(TransformerBlock { forward_fn })
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct TransformerBlock {
    #[derivative(Debug = "ignore")]
    forward_fn: Box<dyn Fn(&Tensor, bool) -> Result<Tensor> + Send>,
}

impl TransformerBlock {
    pub fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        (self.forward_fn)(input, train)
    }
}
