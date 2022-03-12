use super::misc::{NormKind, PaddingKind};
use crate::common::*;
use tch_goodies::DenseDetectionTensorList;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DetectionEmbeddingInit {
    pub inner_c: usize,
    pub ksize: usize,
    pub norm_kind: NormKind,
    pub padding_kind: PaddingKind,
}

impl Default for DetectionEmbeddingInit {
    fn default() -> Self {
        Self {
            inner_c: 64,
            ksize: 5,
            norm_kind: NormKind::BatchNorm,
            padding_kind: PaddingKind::Reflect,
        }
    }
}

impl DetectionEmbeddingInit {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_c: &[usize],
        out_c: usize,
        num_blocks: &[usize],
    ) -> Result<DetectionEmbedding> {
        let Self {
            inner_c,
            ksize,
            norm_kind,
            padding_kind,
        } = self;
        ensure!(in_c.len() == num_blocks.len());
        let bias = norm_kind == NormKind::None;

        let path = path.borrow();
        let out_c = out_c as i64;
        let inner_c = inner_c as i64;
        let padding = ksize / 2;
        let ksize = ksize as i64;

        let branches: Vec<_> = izip!(in_c, num_blocks)
            .enumerate()
            .map(|(index, (&in_c, &num_blocks))| {
                let path = path / format!("branch{}", index);
                let in_c = in_c as i64;

                let seq = nn::seq_t()
                    .add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / "conv_0",
                        in_c,
                        inner_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c))
                    .add_fn(|xs| xs.lrelu());

                (0..num_blocks).fold(seq, |seq, index| {
                    seq.add_fn(|xs| {
                        let (_, _, h, w) = xs.size4().unwrap();
                        xs.upsample_nearest2d(&[h * 2, w * 2], None, None)
                    })
                    .add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / format!("conv_{}", index + 1),
                        inner_c,
                        inner_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c))
                    .add_fn(|xs| xs.lrelu())
                })
            })
            .collect();

        let merge = {
            let path = path / "merge";
            nn::seq_t()
                .add(padding_kind.build([padding, padding, padding, padding]))
                .add(nn::conv2d(
                    &path / "conv1",
                    inner_c * num_blocks.len() as i64,
                    inner_c,
                    ksize,
                    nn::ConvConfig {
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.lrelu())
                .add(padding_kind.build([padding, padding, padding, padding]))
                .add(nn::conv2d(
                    &path / "conv2",
                    inner_c,
                    out_c,
                    ksize,
                    nn::ConvConfig {
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.lrelu())
        };

        Ok(DetectionEmbedding {
            in_c: in_c.iter().map(|&in_c| in_c as i64).collect(),
            branches,
            merge,
        })
    }
}

#[derive(Debug)]
pub struct DetectionEmbedding {
    in_c: Vec<i64>,
    branches: Vec<nn::SequentialT>,
    merge: nn::SequentialT,
}

impl DetectionEmbedding {
    pub fn forward_t(&self, input: &DenseDetectionTensorList, train: bool) -> Result<Tensor> {
        ensure!(input.tensors.len() == self.branches.len());

        let tensors: Vec<_> = izip!(input.tensors.iter(), &self.branches, &self.in_c)
            .map(|(input, branch, &in_c)| {
                let input = input.to_tensor();
                let (in_b, in_e, in_a, in_h, in_w) = input.size5().unwrap();
                ensure!(
                    in_e * in_a == in_c,
                    "expect {} channels, get {} entries and {} anchors",
                    in_c,
                    in_e,
                    in_a
                );
                let input = input.view([in_b, in_c, in_h, in_w]);
                let output = branch.forward_t(&input, train);
                Ok(output)
            })
            .try_collect()?;

        let cat = Tensor::cat(&tensors, 1);
        let merge = self.merge.forward_t(&cat, train);
        Ok(merge)
    }
}
