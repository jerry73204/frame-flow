use crate::common::*;
// use crate::{common::*, config};
// use tch_modules::{BceWithLogitsLoss, BceWithLogitsLossInit, L2Loss};

// #[derive(Debug)]
// pub enum GanLoss {
//     L2(L2Loss),
//     BceWithLogits(BceWithLogitsLoss),
//     WGan(WGanLoss),
// }

// impl GanLoss {
//     pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
//         ensure!(pred.kind() == Kind::Float);
//         ensure!(target.kind() == Kind::Bool);

//         let loss = match self {
//             Self::L2(loss) => loss.forward(pred, &target.to_kind(Kind::Float)),
//             Self::BceWithLogits(loss) => loss.forward(pred, &target.to_kind(Kind::Float)),
//             Self::WGan(loss) => loss.forward(pred, target)?,
//         };

//         Ok(loss)
//     }
// }

// impl From<L2Loss> for GanLoss {
//     fn from(v: L2Loss) -> Self {
//         Self::L2(v)
//     }
// }

// impl From<BceWithLogitsLoss> for GanLoss {
//     fn from(v: BceWithLogitsLoss) -> Self {
//         Self::BceWithLogits(v)
//     }
// }

// impl From<WGanLoss> for GanLoss {
//     fn from(v: WGanLoss) -> Self {
//         Self::WGan(v)
//     }
// }

// impl GanLoss {
//     pub fn new<'a>(
//         path: impl Borrow<nn::Path<'a>>,
//         config: &config::GanLoss,
//         reduction: Reduction,
//     ) -> Self {
//         match config {
//             config::GanLoss::L2 => L2Loss::new(reduction).into(),
//             config::GanLoss::BceWithLogits => {
//                 BceWithLogitsLossInit::default(reduction).build(path).into()
//             }
//             config::GanLoss::WGan => WGanLoss::new(reduction).into(),
//         }
//     }
// }

// #[derive(Debug)]
// pub struct WGanLoss {
//     reduction: Reduction,
// }

// impl WGanLoss {
//     pub fn new(reduction: Reduction) -> Self {
//         Self { reduction }
//     }

//     pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
//         ensure!(pred.kind() == Kind::Float);
//         ensure!(target.kind() == Kind::Bool);

//         let loss = pred * target.to_kind(Kind::Float) * -2.0 + 1.0;
//         let loss = match self.reduction {
//             Reduction::None => loss,
//             Reduction::Mean => loss.mean(Kind::Float),
//             Reduction::Sum => loss.sum(Kind::Float),
//             Reduction::Other(_) => unimplemented!(),
//         };

//         Ok(loss)
//     }
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WGanGpKind {
    Real,
    Fake,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct WGanGpInit {
    pub kind: WGanGpKind,
    pub constant: f64,
    pub lambda: f64,
}

impl WGanGpInit {
    pub fn build(self) -> Result<WGanGp> {
        let Self {
            kind,
            constant,
            lambda,
        } = self;

        ensure!(lambda > 0.0);

        Ok(WGanGp {
            kind,
            c: constant,
            λ: lambda,
        })
    }
}

impl Default for WGanGpInit {
    fn default() -> Self {
        Self {
            kind: WGanGpKind::Mixed,
            constant: 1.0,
            lambda: 10.0,
        }
    }
}

#[derive(Debug)]
pub struct WGanGp {
    kind: WGanGpKind,
    c: f64,
    λ: f64,
}

pub fn dense_detectino_similarity(
    lhs: &DenseDetectionTensorList,
    rhs: &DenseDetectionTensorList,
) -> Result<Tensor> {
    ensure!(lhs.tensors.len() == rhs.tensors.len());

    izip!(&lhs.tensors, &rhs.tensors).try_for_each(|(lhs, rhs)| {
        ensure!(lhs.cy.size() == rhs.cy.size());
        ensure!(lhs.cx.size() == rhs.cx.size());
        ensure!(lhs.h.size() == rhs.h.size());
        ensure!(lhs.w.size() == rhs.w.size());
        ensure!(lhs.cy.size() == rhs.cy.size());
        ensure!(lhs.cx.size() == rhs.cx.size());
        Ok(())
    })?;

    let (cy_loss, cx_loss, h_loss, w_loss, obj_loss, class_loss) =
        izip!(&lhs.tensors, &rhs.tensors)
            .map(|(lhs, rhs)| {
                (
                    lhs.cy.mse_loss(&rhs.cy, Reduction::None).view([-1]),
                    lhs.cx.mse_loss(&rhs.cx, Reduction::None).view([-1]),
                    lhs.h.mse_loss(&rhs.h, Reduction::None).view([-1]),
                    lhs.w.mse_loss(&rhs.w, Reduction::None).view([-1]),
                    lhs.obj_logit
                        .binary_cross_entropy_with_logits::<Tensor>(
                            &rhs.obj_logit.sigmoid(),
                            None,
                            None,
                            Reduction::None,
                        )
                        .view([-1]),
                    lhs.class_logit
                        .binary_cross_entropy_with_logits::<Tensor>(
                            &rhs.class_logit.sigmoid(),
                            None,
                            None,
                            Reduction::None,
                        )
                        .view([-1]),
                )
            })
            .unzip_n_vec();

    let loss = Tensor::cat(&cy_loss, 0).mean(Kind::Float)
        + Tensor::cat(&cx_loss, 0).mean(Kind::Float)
        + Tensor::cat(&h_loss, 0).mean(Kind::Float)
        + Tensor::cat(&w_loss, 0).mean(Kind::Float)
        + Tensor::cat(&obj_loss, 0).mean(Kind::Float)
        + Tensor::cat(&class_loss, 0).mean(Kind::Float);

    Ok(loss)
}

impl WGanGp {
    pub fn forward(
        &self,
        real: &Tensor,
        fake: &Tensor,
        discriminator: impl FnOnce(&Tensor, bool) -> Tensor,
        train: bool,
    ) -> Result<Tensor> {
        // ensure!(!fake.requires_grad() && !real.requires_grad());
        ensure!(fake.size() == real.size());
        ensure!(fake.kind() == real.kind());
        ensure!(fake.device() == real.device());
        ensure!(fake.dim() > 0);

        let Self { kind, c, λ } = *self;
        let batch_size = fake.size()[0];

        let mix = match kind {
            WGanGpKind::Real => real.detach(),
            WGanGpKind::Fake => fake.detach(),
            WGanGpKind::Mixed => {
                let ratio = Tensor::rand(&[batch_size, 1], (fake.kind(), fake.device()))
                    .expand(&[batch_size, fake.numel() as i64 / batch_size], false)
                    .contiguous()
                    .view(&*fake.size());

                &ratio * real.detach() + (-&ratio + 1.0) * fake.detach()
            }
        }
        .set_requires_grad(true);

        let score = discriminator(&mix, train);
        let grad = &Tensor::run_backward(
            &[&score], // outputs
            &[&mix],   // inputs
            true,      // keep_graph
            true,      // create_graph
        )[0];
        let penalty = (Tensor::norm_except_dim(&(grad + 1e-16), 2, 1) - c)
            .pow(2)
            .mean(Kind::Float)
            * λ;
        debug_assert!(!penalty.has_nan());

        Ok(penalty)
    }
}
