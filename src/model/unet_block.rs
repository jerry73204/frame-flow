use super::misc::NormKind;
use crate::common::*;

#[derive(Debug, Clone)]
pub struct UnetBlockInit {
    pub norm_kind: NormKind,
    pub dropout: bool,
}

impl UnetBlockInit {
    pub fn build<'a, F>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_c: usize,
        outer_c: usize,
        inner_c: usize,
        module: UnetModule<F>,
    ) -> UnetBlock
    where
        F: 'static + Send + Fn(&Tensor, bool) -> Tensor,
    {
        let path = path.borrow();
        let Self { norm_kind, dropout } = self;
        let bias = norm_kind == NormKind::InstanceNorm;
        let in_c = in_c as i64;
        let inner_c = inner_c as i64;
        let outer_c = outer_c as i64;

        let down_conv = nn::conv2d(
            path / "down_conv",
            in_c,
            inner_c,
            4,
            nn::ConvConfig {
                stride: 2,
                padding: 1,
                bias,
                ..Default::default()
            },
        );

        let seq = match module {
            UnetModule::Standard(f) => {
                let up_conv = nn::conv_transpose2d(
                    path / "up_conv",
                    inner_c * 2,
                    outer_c,
                    4,
                    nn::ConvTransposeConfig {
                        stride: 2,
                        padding: 1,
                        bias,
                        ..Default::default()
                    },
                );
                let seq = nn::seq_t()
                    .add_fn(|xs| xs.relu())
                    .add(down_conv)
                    .add(norm_kind.build(path / "down_norm", inner_c))
                    .add_fn_t(f)
                    .add_fn(|xs| xs.relu())
                    .add(up_conv)
                    .add(norm_kind.build(path / "up_norm", outer_c));

                if dropout {
                    seq.add_fn_t(|xs, train| xs.dropout(0.5, train))
                } else {
                    seq
                }
            }
            UnetModule::OuterMost(f) => {
                let up_conv = nn::conv_transpose2d(
                    path / "up_conv",
                    inner_c * 2,
                    outer_c,
                    4,
                    nn::ConvTransposeConfig {
                        stride: 2,
                        padding: 1,
                        ..Default::default()
                    },
                );
                nn::seq_t()
                    .add(down_conv)
                    .add_fn_t(f)
                    .add_fn(|xs| xs.relu())
                    .add(up_conv)
                    .add_fn(|xs| xs.tanh())
            }
            UnetModule::InnerMost => {
                let up_conv = nn::conv_transpose2d(
                    path / "up_conv",
                    inner_c,
                    outer_c,
                    4,
                    nn::ConvTransposeConfig {
                        stride: 2,
                        padding: 1,
                        bias,
                        ..Default::default()
                    },
                );
                nn::seq_t()
                    .add_fn(|xs| xs.relu())
                    .add(down_conv)
                    .add_fn(|xs| xs.relu())
                    .add(up_conv)
                    .add(norm_kind.build(path / "up_norm", outer_c))
            }
        };

        UnetBlock { seq }
    }
}

impl Default for UnetBlockInit {
    fn default() -> Self {
        Self {
            norm_kind: NormKind::BatchNorm,
            dropout: false,
        }
    }
}

#[derive(Debug)]
pub enum UnetModule<F>
where
    F: 'static + Fn(&Tensor, bool) -> Tensor + Send,
{
    Standard(F),
    OuterMost(F),
    InnerMost,
}

impl<F> UnetModule<F>
where
    F: 'static + Fn(&Tensor, bool) -> Tensor + Send,
{
    pub fn standard(f: F) -> Self {
        Self::Standard(f)
    }

    pub fn outer_most(f: F) -> Self {
        Self::OuterMost(f)
    }
}

impl UnetModule<Box<dyn Fn(&Tensor, bool) -> Tensor + Send>> {
    pub fn inner_most() -> Self {
        Self::InnerMost
    }
}

#[derive(Debug)]
pub struct UnetBlock {
    seq: nn::SequentialT,
}

impl nn::ModuleT for UnetBlock {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        self.seq.forward_t(input, train)
    }
}
