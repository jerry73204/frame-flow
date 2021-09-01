use super::{
    generator::ResnetGeneratorInit,
    misc::{NormKind, PaddingKind},
};
use crate::{
    common::*,
    utils::{DenseDetectionTensorExt, DenseDetectionTensorListExt},
};
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
        input_len: usize,
        num_classes: usize,
        inner_c: usize,
    ) -> Result<Transformer> {
        // const BORDER_SIZE_RATIO: f64 = 4.0 / 64.0;

        let path = path.borrow();
        let Self {
            ksize,
            norm_kind,
            padding_kind,
            num_resnet_blocks,
            num_scaling_blocks,
            num_down_sample,
        } = self;
        // let device = path.device();
        let in_c = 5 + num_classes;
        ensure!(input_len > 0);

        // let (ctx_encoder, ctx_decoder) =
        //     ChannelWiseAutoencoderInit { norm_kind }.build(path / "autoencoder", in_c, inner_c);

        // let det_encoder = move |det: &DenseDetectionTensor, train: bool| -> Result<_> {
        //     assert!(!det.has_nan());

        //     let (xs, anchors) = encode_detection(det)?;
        //     assert!(!xs.has_nan());

        //     let xs = ctx_encoder.forward_t(&xs, train);
        //     assert!(!xs.has_nan());

        //     Ok((xs, anchors))
        // };
        // let det_decoder = move |xs: &Tensor, anchors, train: bool| {
        //     let xs = ctx_decoder.forward_t(xs, train);
        //     decode_detection(&xs, anchors)
        // };

        let attention_block = TransformerBlockInit {
            ksize,
            norm_kind,
            padding_kind,
            num_resnet_blocks,
            num_scaling_blocks,
            num_down_sample,
        }
        .build(path / "attention_block", in_c, in_c * input_len, inner_c)?;

        // let patch_block = ResnetGeneratorInit {
        //     ksize,
        //     norm_kind,
        //     padding_kind,
        //     num_scale_blocks: num_scaling_blocks,
        //     num_blocks: num_resnet_blocks,
        //     ..Default::default()
        // }
        // .build(path / "patch_block", inner_c * input_len, inner_c, inner_c);

        let forward_fn = Box::new(
            move |input: &[&DenseDetectionTensorList],
                  train: bool,
                  with_artifacts: bool|
                  -> Result<_> {
                ensure!(input.len() == input_len);
                ensure!(input.iter().all(|list| list.tensors.len() == 1));
                ensure!(
                    input
                        .iter()
                        .map(|list| &list.tensors)
                        .flatten()
                        .all(|tensor| tensor.num_anchors() == 1
                            && tensor.num_classes() == num_classes)
                );
                assert!(!input.iter().any(|&list| list.has_nan()));

                let in_h = input[0].tensors[0].height() as i64;
                let in_w = input[0].tensors[0].width() as i64;
                let anchors = input[0].tensors[0].anchors.clone();
                ensure!(input
                    .iter()
                    .flat_map(|list| &list.tensors)
                    .all(|det| det.height() == in_h as usize
                        && det.width() == in_w as usize
                        && det.anchors == anchors));

                let in_context_vec: Vec<_> = input
                    .iter()
                    .flat_map(|list| &list.tensors)
                    .map(|det| -> Result<_> {
                        let (context, _) = encode_detection(det)?;
                        Ok(context)
                    })
                    .try_collect()?;
                // let anchors = {
                //     let mut iter = anchors_set.into_iter();
                //     let anchors = iter.next().unwrap();
                //     ensure!(iter.next().is_none());
                //     anchors
                // };

                let merge_context = Tensor::cat(&in_context_vec, 1);
                let (last_context, anchors) = unpack_detection(&input.last().unwrap().tensors[0])?;
                assert!(!merge_context.has_nan());

                let (shifted, attention_image) = attention_block.forward_t(
                    &last_context,
                    &merge_context,
                    train,
                    with_artifacts,
                )?;
                // let patch = patch_block.forward_t(&merge_context, train);
                // let patch_mask = {
                //     let border_h = (in_h as f64 * BORDER_SIZE_RATIO).floor() as usize;
                //     let border_w = (in_w as f64 * BORDER_SIZE_RATIO).floor() as usize;
                //     Tensor::from_cv(nd::Array2::from_shape_fn(
                //         [in_h as usize, in_w as usize],
                //         |(row, col)| {
                //             let ok = row < border_h
                //                 || row >= (in_h as usize - border_h)
                //                 || col < border_w
                //                 || col >= (in_w as usize - border_w);
                //             if ok {
                //                 1f32
                //             } else {
                //                 0f32
                //             }
                //         },
                //     ))
                //     .set_requires_grad(false)
                //     .to_device(device)
                // };

                // assert!(!shifted.has_nan());
                // assert!(!patch.has_nan());

                // let out_context = shifted + patch * patch_mask;
                let out_context = shifted;
                assert!(!out_context.has_nan());

                let out_detection = pack_detection(&out_context, anchors);

                let output: DenseDetectionTensorList = DenseDetectionTensorListUnchecked {
                    tensors: vec![out_detection],
                }
                .try_into()
                .unwrap();

                let artifacts = with_artifacts.then(|| {
                    // compute reconstruction for detection autoencoder
                    // let autoencoder_recon_loss = {
                    //     let recon_loss_vec: Vec<_> =
                    //         izip!(input.iter().flat_map(|list| &list.tensors), &in_context_vec)
                    //             .map(|(orig, latent)| -> Tensor {
                    //                 let recon = det_decoder(latent, anchors.clone(), train);
                    //                 super::loss::dense_detection_similarity(
                    //                     &orig.detach().copy(),
                    //                     &recon,
                    //                 )
                    //                 .unwrap()
                    //                 .total_loss()
                    //             })
                    //             .collect();
                    //     recon_loss_vec.iter().sum::<Tensor>() / recon_loss_vec.len() as f64
                    // };

                    TransformerArtifacts {
                        // autoencoder_recon_loss,
                        attention_image: attention_image.unwrap(),
                    }
                });

                Ok((output, artifacts))
            },
        );

        Ok(Transformer {
            input_len,
            forward_fn,
        })
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Transformer {
    input_len: usize,
    #[derivative(Debug = "ignore")]
    forward_fn: Box<
        dyn Fn(
                &[&DenseDetectionTensorList],
                bool,
                bool,
            ) -> Result<(DenseDetectionTensorList, Option<TransformerArtifacts>)>
            + Send,
    >,
}

impl Transformer {
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    pub fn forward_t(
        &self,
        input: &[impl Borrow<DenseDetectionTensorList>],
        train: bool,
        with_artifacts: bool,
    ) -> Result<(DenseDetectionTensorList, Option<TransformerArtifacts>)> {
        let input: Vec<_> = input.iter().map(|list| list.borrow()).collect();
        (self.forward_fn)(&input, train, with_artifacts)
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
        input_c: usize,
        ctx_c: usize,
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

        let context_norm =
            GroupNormInit::default().build(path / "context_norm", ctx_c as i64, ctx_c as i64);
        let query_transform =
            resnet_init
                .clone()
                .build(path / "query_transform", ctx_c, inner_c, inner_c);
        let key_transform = resnet_init.build(path / "key_transform", ctx_c, inner_c, inner_c);
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

        let forward_fn = Box::new(
            move |input: &Tensor,
                  context: &Tensor,
                  train: bool,
                  with_artifacts: bool|
                  -> Result<(Tensor, Option<Tensor>)> {
                // assert!(!input.has_nan());
                // assert!(!context.has_nan());

                let (bsize, _, in_h, in_w) = input.size4()?;
                ensure!(
                    matches!(context.size4()?, (bsize_, ctx_c_, ctx_h, ctx_w) if bsize == bsize_ && ctx_c == ctx_c_ as usize && in_h == ctx_h && in_w == ctx_w)
                );

                let patch_h = in_h / 2i64.pow(num_down_sample as u32);
                let patch_w = in_w / 2i64.pow(num_down_sample as u32);

                let context = context_norm.forward_t(context, train);
                // assert!(!context.has_nan());
                let key = key_transform.forward_t(&context, train);
                // assert!(!key.has_nan());
                let query = {
                    let xs = query_transform.forward_t(&context, train);
                    query_down_sample.forward_t(&xs, train)
                };
                // assert!(!query.has_nan());

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
                // assert!(!attention.has_nan());

                let attention_image = with_artifacts
                    .then(|| attention.view([bsize, 1, patch_h, patch_w, in_h, in_w]));

                let patches = Tensor::einsum(
                    "bqk,bck->bcqk",
                    &[attention, input.view([bsize, input_c as i64, -1])],
                )
                .view([bsize, input_c as i64, patch_h, patch_w, in_h, in_w]);

                // pad patch sizes to odd numbers
                let (patches, patch_h) = if patch_h & 1 == 1 {
                    (patches, patch_h)
                } else {
                    (
                        patches.constant_pad_nd(&[0, 0, 0, 0, 0, 0, 0, 1]),
                        patch_h + 1,
                    )
                };
                let (patches, patch_w) = if patch_w & 1 == 1 {
                    (patches, patch_w)
                } else {
                    (
                        patches.constant_pad_nd(&[0, 0, 0, 0, 0, 1, 0, 0]),
                        patch_w + 1,
                    )
                };

                // merge patches
                let output = patches
                    .view([bsize, input_c as i64 * patch_h * patch_w, in_h * in_w])
                    .col2im(
                        &[in_h, in_w],
                        &[patch_h, patch_w],
                        &[1, 1], // dilation
                        &[patch_h / 2, patch_w / 2],
                        &[1, 1], // stride
                    );

                Ok((output, attention_image))
            },
        );

        Ok(TransformerBlock { forward_fn })
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct TransformerBlock {
    #[derivative(Debug = "ignore")]
    forward_fn:
        Box<dyn Fn(&Tensor, &Tensor, bool, bool) -> Result<(Tensor, Option<Tensor>)> + Send>,
}

impl TransformerBlock {
    pub fn forward_t(
        &self,
        input: &Tensor,
        context: &Tensor,
        train: bool,
        with_artifacts: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        (self.forward_fn)(input, context, train, with_artifacts)
    }
}

#[derive(Debug)]
pub struct TransformerArtifacts {
    // pub autoencoder_recon_loss: Tensor,
    pub attention_image: Tensor,
}

#[derive(Debug, Clone)]
pub struct ChannelWiseAutoencoderInit {
    pub norm_kind: NormKind,
}

impl ChannelWiseAutoencoderInit {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_c: usize,
        inner_c: usize,
    ) -> (ChannelWiseEncoder, ChannelWiseDecoder) {
        let path = path.borrow();
        let Self { norm_kind } = self;
        let bias = norm_kind == NormKind::InstanceNorm;

        let encoder = {
            let path = path / "encoder";

            let seq = {
                let path = &path / "block_0";
                nn::seq_t()
                    // .inspect(|xs| {
                    //     // dbg!(xs.min(), xs.max());
                    //     assert!(!xs.has_nan());
                    // })
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
                    // .inspect(|xs| {
                    //     // dbg!(xs.min(), xs.max());
                    //     assert!(!xs.has_nan());
                    // })
                    .add(norm_kind.build(&path / "norm", inner_c as i64))
                    // .inspect(|xs| {
                    //     assert!(!xs.has_nan());
                    // })
                    .add_fn(|xs| xs.lrelu())
            };

            (1..3).fold(seq, |seq, index| {
                let path = &path / format!("block_{}", index);

                seq.add(nn::conv2d(
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

                seq.add(nn::conv2d(
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
                .add(norm_kind.build(&path / "norm", in_c as i64))
                .add_fn(|xs| xs.lrelu())
            };

            seq
        };

        (
            ChannelWiseEncoder {
                forward_fn: encoder,
            },
            ChannelWiseDecoder {
                forward_fn: decoder,
            },
        )
    }
}

#[derive(Debug)]
pub struct ChannelWiseEncoder {
    forward_fn: nn::SequentialT,
}

#[derive(Debug)]
pub struct ChannelWiseDecoder {
    forward_fn: nn::SequentialT,
}

impl nn::ModuleT for ChannelWiseEncoder {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        self.forward_fn.forward_t(input, train)
    }
}

impl nn::ModuleT for ChannelWiseDecoder {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        self.forward_fn.forward_t(input, train)
    }
}

pub fn encode_detection(input: &DenseDetectionTensor) -> Result<(Tensor, Vec<RatioSize<R64>>)> {
    let device = input.device();
    let bsize = input.batch_size() as i64;
    let num_classes = input.num_classes() as i64;
    let in_h = input.height() as i64;
    let in_w = input.width() as i64;
    let num_anchors = input.anchors.len() as i64;

    let y_offsets = Tensor::arange(in_h, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, in_h, 1]);
    let x_offsets = Tensor::arange(in_w, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, 1, in_w]);
    let (anchor_heights, anchor_widths) = {
        let (anchor_h_vec, anchor_w_vec) = input
            .anchors
            .iter()
            .cloned()
            .map(|anchor_size| {
                let anchor_size = anchor_size.cast::<f32>().unwrap();
                (anchor_size.h, anchor_size.w)
            })
            .unzip_n_vec();

        let anchor_heights = Tensor::of_slice(&anchor_h_vec)
            .set_requires_grad(false)
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);
        let anchor_widths = Tensor::of_slice(&anchor_w_vec)
            .set_requires_grad(false)
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);

        (anchor_heights, anchor_widths)
    };

    let merge = {
        // let wtf = (&input.cy * in_h as f64 - &y_offsets + 0.5) / 2.0;
        // dbg!(input.cy.max());
        // dbg!(input.cy.min());
        // dbg!(wtf.max());
        // dbg!(wtf.min());
        // assert!(bool::from(input.h.ge(0.0).all()));
        // assert!(bool::from(input.w.ge(0.0).all()));

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

        // assert!(bool::from(input.h.ge(0.0).all()));
        // assert!(bool::from(input.w.ge(0.0).all()));

        ensure!(!cy_logit.has_nan());
        ensure!(!cx_logit.has_nan());
        ensure!(!h_logit.has_nan());
        ensure!(!w_logit.has_nan());
        ensure!(!obj_logit.has_nan());
        ensure!(!class_logit.has_nan());

        Tensor::cat(
            &[cy_logit, cx_logit, h_logit, w_logit, obj_logit, class_logit],
            1,
        )
        .view([bsize, -1, in_h, in_w])
    };

    // dbg!(merge.min(), merge.max());
    // assert!(!merge.has_nan());

    Ok((merge, input.anchors.clone()))
}

pub fn decode_detection(input: &Tensor, anchors: Vec<RatioSize<R64>>) -> DenseDetectionTensor {
    assert!(!input.has_nan());

    let device = input.device();
    let num_anchors = anchors.len() as i64;
    let (bsize, _, in_h, in_w) = input.size4().unwrap();

    let y_offsets = Tensor::arange(in_h, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, in_h, 1]);
    let x_offsets = Tensor::arange(in_w, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, 1, in_w]);
    let (anchor_heights, anchor_widths) = {
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
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);
        let anchor_widths = Tensor::of_slice(&anchor_w_vec)
            .set_requires_grad(false)
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);

        (anchor_heights, anchor_widths)
    };

    let xs = input.view([bsize, -1, num_anchors, in_h, in_w]);
    let cy = ((xs.i((.., 0..1, .., .., ..)).sigmoid() * 2.0 - 0.5) + y_offsets) / in_h as f64;
    let cx = ((xs.i((.., 1..2, .., .., ..)).sigmoid() * 2.0 - 0.5) + x_offsets) / in_w as f64;
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

    assert!(!output.has_nan());

    output
}

pub fn unpack_detection(input: &DenseDetectionTensor) -> Result<(Tensor, Vec<RatioSize<R64>>)> {
    let device = input.device();
    let bsize = input.batch_size() as i64;
    let num_classes = input.num_classes() as i64;
    let in_h = input.height() as i64;
    let in_w = input.width() as i64;
    let num_anchors = input.anchors.len() as i64;

    let y_offsets = Tensor::arange(in_h, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, in_h, 1]);
    let x_offsets = Tensor::arange(in_w, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, 1, in_w]);
    let (anchor_heights, anchor_widths) = {
        let (anchor_h_vec, anchor_w_vec) = input
            .anchors
            .iter()
            .cloned()
            .map(|anchor_size| {
                let anchor_size = anchor_size.cast::<f32>().unwrap();
                (anchor_size.h, anchor_size.w)
            })
            .unzip_n_vec();

        let anchor_heights = Tensor::of_slice(&anchor_h_vec)
            .set_requires_grad(false)
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);
        let anchor_widths = Tensor::of_slice(&anchor_w_vec)
            .set_requires_grad(false)
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);

        (anchor_heights, anchor_widths)
    };

    let merge = {
        // let wtf = (&input.cy * in_h as f64 - &y_offsets + 0.5) / 2.0;
        // dbg!(input.cy.max());
        // dbg!(input.cy.min());
        // dbg!(wtf.max());
        // dbg!(wtf.min());
        // assert!(bool::from(input.h.ge(0.0).all()));
        // assert!(bool::from(input.w.ge(0.0).all()));

        let cy_unbiased = ((&input.cy * in_h as f64 - &y_offsets + 0.5) / 2.0).view([
            bsize,
            1,
            num_anchors,
            in_h,
            in_w,
        ]);
        let cx_unbiased = ((&input.cx * in_w as f64 - &x_offsets + 0.5) / 2.0).view([
            bsize,
            1,
            num_anchors,
            in_h,
            in_w,
        ]);
        let h_unbiased =
            ((&input.h / anchor_heights).sqrt() / 2.0).view([bsize, 1, num_anchors, in_h, in_w]);
        let w_unbiased =
            ((&input.w / anchor_widths).sqrt() / 2.0).view([bsize, 1, num_anchors, in_h, in_w]);
        let obj_prob = input.obj_prob().view([bsize, 1, num_anchors, in_h, in_w]);
        let class_prob =
            input
                .class_prob()
                .view([bsize, num_classes as i64, num_anchors, in_h, in_w]);

        // dbg!(cy_logit.min(), cy_logit.max());
        // dbg!(cx_logit.min(), cx_logit.max());
        // dbg!(h_logit.min(), h_logit.max());
        // dbg!(w_logit.min(), w_logit.max());
        // dbg!(obj_logit.min(), obj_logit.max());
        // dbg!(class_logit.min(), class_logit.max());

        // assert!(bool::from(input.h.ge(0.0).all()));
        // assert!(bool::from(input.w.ge(0.0).all()));

        ensure!(!cy_unbiased.has_nan());
        ensure!(!cx_unbiased.has_nan());
        ensure!(!h_unbiased.has_nan());
        ensure!(!w_unbiased.has_nan());
        ensure!(!obj_prob.has_nan());
        ensure!(!class_prob.has_nan());

        Tensor::cat(
            &[
                cy_unbiased,
                cx_unbiased,
                h_unbiased,
                w_unbiased,
                obj_prob,
                class_prob,
            ],
            1,
        )
        .view([bsize, -1, in_h, in_w])
    };

    // dbg!(merge.min(), merge.max());
    // assert!(!merge.has_nan());

    Ok((merge, input.anchors.clone()))
}

pub fn pack_detection(input: &Tensor, anchors: Vec<RatioSize<R64>>) -> DenseDetectionTensor {
    assert!(!input.has_nan());

    let device = input.device();
    let num_anchors = anchors.len() as i64;
    let (bsize, _, in_h, in_w) = input.size4().unwrap();

    let y_offsets = Tensor::arange(in_h, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, in_h, 1]);
    let x_offsets = Tensor::arange(in_w, (Kind::Float, device))
        .set_requires_grad(false)
        .view([1, 1, 1, 1, in_w]);
    let (anchor_heights, anchor_widths) = {
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
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);
        let anchor_widths = Tensor::of_slice(&anchor_w_vec)
            .set_requires_grad(false)
            .to_device(device)
            .view([1, 1, num_anchors, 1, 1]);

        (anchor_heights, anchor_widths)
    };

    let xs = input.view([bsize, -1, num_anchors, in_h, in_w]);
    let cy = ((xs.i((.., 0..1, .., .., ..)) * 2.0 - 0.5) + y_offsets) / in_h as f64;
    let cx = ((xs.i((.., 1..2, .., .., ..)) * 2.0 - 0.5) + x_offsets) / in_w as f64;
    let h = xs.i((.., 2..3, .., .., ..)).mul(2.0).pow(2.0) * anchor_heights;
    let w = xs.i((.., 3..4, .., .., ..)).mul(2.0).pow(2.0) * anchor_widths;
    let obj_logit = xs.i((.., 4..5, .., .., ..)).logit(1e-6);
    let class_logit = xs.i((.., 5.., .., .., ..)).logit(1e-6);

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

    assert!(!output.has_nan());

    output
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

            let (tensor, _anchors) = encode_detection(&orig).unwrap();
            let recon = decode_detection(&tensor, anchors);

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
    fn detection_pack_unpack() {
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

            let (tensor, _anchors) = unpack_detection(&orig).unwrap();
            let recon = pack_detection(&tensor, anchors);

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
}
