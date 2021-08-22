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

impl TransformerInit {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_c: usize,
        inner_c: usize,
    ) -> Result<Transformer> {
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
                .build(path / "query_transform", inner_c, inner_c, inner_c);
        let key_transform = resnet_init.build(path / "key_transform", inner_c, inner_c, inner_c);
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
            let path = path / "output_transform";
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

            let value = encoder.forward_t(input, train);

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

            let merge = patches
                .view([bsize, inner_c as i64 * patch_h * patch_w, in_h * in_w])
                .col2im(
                    &[in_h, in_w],
                    &[patch_h, patch_w],
                    &[1, 1], // dilation
                    &[patch_h / 2, patch_w / 2],
                    &[1, 1], // stride
                );

            let output = decoder.forward_t(&merge, train);

            Ok(output)
        });

        Ok(Transformer { forward_fn })
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Transformer {
    #[derivative(Debug = "ignore")]
    forward_fn: Box<dyn Fn(&Tensor, bool) -> Result<Tensor> + Send>,
}

impl Transformer {
    pub fn forward<'a>(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        (self.forward_fn)(input, train)
    }
}
