use super::misc::NormKind;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DetectionEmbeddingInit {
    pub in_c: [usize; 3],
    pub out_c: usize,
    pub inner_c: usize,
    pub norm_kind: NormKind,
}

impl DetectionEmbeddingInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> DetectionEmbedding {
        let Self {
            in_c,
            out_c,
            inner_c,
            norm_kind,
        } = self;
        let bias = norm_kind == NormKind::InstanceNorm;

        let path = path.borrow();
        let out_c = out_c as i64;
        let inner_c = inner_c as i64;

        let branch1 = {
            let path = path / "branch1";

            nn::seq_t()
                .add(nn::conv2d(
                    &path / "conv",
                    in_c[0] as i64,
                    inner_c,
                    3,
                    nn::ConvConfig {
                        padding: 1,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.relu())
        };

        let branch2 = {
            let path = path / "branch2";

            nn::seq_t()
                .add(nn::conv_transpose2d(
                    &path / "conv_transpose",
                    in_c[1] as i64,
                    inner_c,
                    3,
                    nn::ConvTransposeConfig {
                        stride: 2,
                        padding: 1,
                        output_padding: 1,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm1", inner_c))
                .add_fn(|xs| xs.relu())
        };

        let branch3 = {
            let path = path / "branch3";

            nn::seq_t()
                .add(nn::conv_transpose2d(
                    &path / "conv_transpose1",
                    in_c[2] as i64,
                    inner_c,
                    3,
                    nn::ConvTransposeConfig {
                        stride: 2,
                        padding: 1,
                        output_padding: 1,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm1", inner_c))
                .add_fn(|xs| xs.relu())
                .add(nn::conv_transpose2d(
                    &path / "conv_transpose2",
                    inner_c,
                    inner_c,
                    3,
                    nn::ConvTransposeConfig {
                        stride: 2,
                        padding: 1,
                        output_padding: 1,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm2", inner_c))
                .add_fn(|xs| xs.relu())
        };

        let merge = {
            let path = path / "merge";
            nn::seq_t()
                .add(nn::conv2d(
                    &path / "conv",
                    inner_c * 3,
                    inner_c,
                    3,
                    nn::ConvConfig {
                        padding: 1,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.relu())
        };

        const NUM_UP_SAMPLING: usize = 3;
        let merge = (0..NUM_UP_SAMPLING).fold(merge, |seq, index| {
            let path = path / format!("up_sample_{}", index);
            seq.add(nn::conv_transpose2d(
                &path / "conv_transpose",
                inner_c,
                inner_c,
                3,
                nn::ConvTransposeConfig {
                    stride: 2,
                    padding: 1,
                    output_padding: 1,
                    bias,
                    ..Default::default()
                },
            ))
            .add(norm_kind.build(&path / "norm", inner_c))
            .add_fn(|xs| xs.relu())
        });

        let merge = {
            let path = path / "last";
            merge
                .add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    out_c,
                    3,
                    nn::ConvConfig {
                        padding: 1,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.relu())
        };

        DetectionEmbedding {
            in_c,
            branch1,
            branch2,
            branch3,
            merge,
        }
    }
}

#[derive(Debug)]
pub struct DetectionEmbedding {
    in_c: [usize; 3],
    branch1: nn::SequentialT,
    branch2: nn::SequentialT,
    branch3: nn::SequentialT,
    merge: nn::SequentialT,
}

impl DetectionEmbedding {
    pub fn forward_t(&self, input: &DenseDetectionTensorList, train: bool) -> Result<Tensor> {
        ensure!(input.tensors.len() == 3);
        ensure!(
            input.tensors[0].height() == input.tensors[1].height() * 2
                && input.tensors[0].height() == input.tensors[2].height() * 4
        );
        ensure!(
            input.tensors[0].width() == input.tensors[1].width() * 2
                && input.tensors[0].width() == input.tensors[2].width() * 4
        );

        let tensors: Vec<_> = izip!(input.tensors.iter(), self.in_c)
            .map(|(tensor, in_c)| {
                let tensor = tensor.to_tensor();
                let (b, e, a, h, w) = tensor.size5().unwrap();
                ensure!(
                    e * a == in_c as i64,
                    "expect {} channels, get {} entries and {} anchros",
                    in_c,
                    e,
                    a
                );
                let tensor = tensor.view([b, e * a, h, w]);
                Ok(tensor)
            })
            .try_collect()?;

        let feature1 = self.branch1.forward_t(&tensors[0], train);
        let feature2 = self.branch2.forward_t(&tensors[1], train);
        let feature3 = self.branch3.forward_t(&tensors[2], train);
        let cat = Tensor::cat(&[feature1, feature2, feature3], 1);
        let merge = self.merge.forward_t(&cat, train);
        Ok(merge)
    }
}
