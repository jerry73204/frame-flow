use super::misc::PaddingKind;
use crate::common::*;

#[derive(Debug, Clone)]
pub struct ResnetBlockInit {
    pub padding_kind: PaddingKind,
    pub dropout: bool,
    pub bias: bool,
}

impl Default for ResnetBlockInit {
    fn default() -> Self {
        Self {
            padding_kind: PaddingKind::Reflect,
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
            dropout,
            bias,
        } = self;
        let channels = channels as i64;

        let seq = nn::seq_t();

        let (seq, conv_pad) = match padding_kind {
            PaddingKind::Reflect => {
                let seq = seq.add_fn(|xs| xs.reflection_pad2d(&[1, 1]));
                (seq, 0)
            }
            PaddingKind::Replicate => {
                let seq = seq.add_fn(|xs| xs.replication_pad2d(&[1, 1]));
                (seq, 0)
            }
            PaddingKind::Zero => (seq, 1),
        };

        let seq = seq
            .add(nn::conv2d(
                path / "conv1",
                channels,
                channels,
                3,
                nn::ConvConfig {
                    padding: conv_pad,
                    bias,
                    ..Default::default()
                },
            ))
            .add(nn::batch_norm2d(
                path / "norm1",
                channels,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());

        let seq = if dropout {
            seq.add_fn_t(|xs, train| xs.dropout(0.5, train))
        } else {
            seq
        };

        let (seq, conv_pad) = match padding_kind {
            PaddingKind::Reflect => {
                let seq = seq.add_fn(|xs| xs.reflection_pad2d(&[1, 1]));
                (seq, 0)
            }
            PaddingKind::Replicate => {
                let seq = seq.add_fn(|xs| xs.replication_pad2d(&[1, 1]));
                (seq, 0)
            }
            PaddingKind::Zero => (seq, 1),
        };

        let seq = seq
            .add(nn::conv2d(
                path / "conv2",
                channels,
                channels,
                3,
                nn::ConvConfig {
                    padding: conv_pad,
                    bias,
                    ..Default::default()
                },
            ))
            .add(nn::batch_norm2d(
                path / "norm2",
                channels,
                Default::default(),
            ));

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
