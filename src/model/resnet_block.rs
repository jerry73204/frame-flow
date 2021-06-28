use super::misc::{NormKind, PaddingKind};
use crate::common::*;

#[derive(Debug, Clone)]
pub struct ResnetBlockInit {
    pub padding_kind: PaddingKind,
    pub dropout: bool,
    pub norm_kind: NormKind,
    pub bias: bool,
}

impl Default for ResnetBlockInit {
    fn default() -> Self {
        Self {
            padding_kind: PaddingKind::Reflect,
            norm_kind: NormKind::BatchNorm,
            dropout: false,
            bias: true,
        }
    }
}

impl ResnetBlockInit {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>, channels: usize) -> ResnetBlock {
        let path = path.borrow();
        let Self {
            padding_kind,
            norm_kind,
            dropout,
            bias,
        } = self;
        let channels = channels as i64;

        let seq = nn::seq_t()
            .add(padding_kind.build([1, 1, 1, 1]))
            .add(nn::conv2d(
                path / "conv1",
                channels,
                channels,
                3,
                nn::ConvConfig {
                    padding: 0,
                    bias,
                    ..Default::default()
                },
            ))
            .add(norm_kind.build(path / "norm1", channels))
            .add_fn(|xs| xs.relu());

        let seq = if dropout {
            seq.add_fn_t(|xs, train| xs.dropout(0.5, train))
        } else {
            seq
        };

        let seq = seq
            .add(padding_kind.build([1, 1, 1, 1]))
            .add(nn::conv2d(
                path / "conv2",
                channels,
                channels,
                3,
                nn::ConvConfig {
                    padding: 0,
                    bias,
                    ..Default::default()
                },
            ))
            .add(norm_kind.build(path / "norm2", channels));

        ResnetBlock { seq }
    }
}

#[derive(Debug)]
pub struct ResnetBlock {
    seq: nn::SequentialT,
}

impl nn::ModuleT for ResnetBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs + self.seq.forward_t(xs, train)
    }
}
