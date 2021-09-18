use crate::common::*;

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

pub fn dense_detection_list_similarity(
    lhs: &DenseDetectionTensorList,
    rhs: &DenseDetectionTensorList,
) -> Result<DetectionSimilarity> {
    tch::no_grad(|| {
        ensure!(lhs.tensors.len() == rhs.tensors.len());

        let (count, cy_loss, cx_loss, h_loss, w_loss, obj_loss, class_loss): (
            Count<_>,
            AddVal<_>,
            AddVal<_>,
            AddVal<_>,
            AddVal<_>,
            AddVal<_>,
            AddVal<_>,
        ) = izip!(&lhs.tensors, &rhs.tensors)
            .map(|(lhs, rhs)| -> Result<_> {
                let DetectionSimilarity {
                    cy_loss,
                    cx_loss,
                    h_loss,
                    w_loss,
                    obj_loss,
                    class_loss,
                } = dense_detection_similarity(lhs, rhs)?;

                Ok(((), cy_loss, cx_loss, h_loss, w_loss, obj_loss, class_loss))
            })
            .try_collect::<_, Vec<_>, _>()?
            .into_iter()
            .unzip_n();

        let count = count.get() as f64;

        Ok(DetectionSimilarity {
            cy_loss: cy_loss.unwrap() / count,
            cx_loss: cx_loss.unwrap() / count,
            h_loss: h_loss.unwrap() / count,
            w_loss: w_loss.unwrap() / count,
            obj_loss: obj_loss.unwrap() / count,
            class_loss: class_loss.unwrap() / count,
        })
    })
}

pub fn dense_detection_similarity(
    lhs: &DenseDetectionTensor,
    rhs: &DenseDetectionTensor,
) -> Result<DetectionSimilarity> {
    tch::no_grad(|| {
        let diff1 = dense_detection_difference(lhs, rhs)?;
        let diff2 = dense_detection_difference(rhs, lhs)?;

        Ok(DetectionSimilarity {
            cy_loss: (diff1.cy_loss + diff2.cy_loss) / 2.0,
            cx_loss: (diff1.cx_loss + diff2.cx_loss) / 2.0,
            h_loss: (diff1.h_loss + diff2.h_loss) / 2.0,
            w_loss: (diff1.w_loss + diff2.w_loss) / 2.0,
            obj_loss: (diff1.obj_loss + diff2.obj_loss) / 2.0,
            class_loss: (diff1.class_loss + diff2.class_loss) / 2.0,
        })
    })
}

fn dense_detection_difference(
    src: &DenseDetectionTensor,
    dst: &DenseDetectionTensor,
) -> Result<DetectionSimilarity> {
    const CONFIDENCE_THRESH: f64 = 0.4;

    tch::no_grad(|| {
        ensure!(src.cy.size() == dst.cy.size());
        ensure!(src.cx.size() == dst.cx.size());
        ensure!(src.h.size() == dst.h.size());
        ensure!(src.w.size() == dst.w.size());
        ensure!(src.cy.size() == dst.cy.size());
        ensure!(src.cx.size() == dst.cx.size());

        let (confidence, _) = dst.confidence().max_dim(1, true);
        let indexes: Vec<_> = confidence.ge(CONFIDENCE_THRESH).nonzero_numpy();

        if indexes[0].is_empty() {
            let device = src.device();
            return Ok(DetectionSimilarity {
                cy_loss: Tensor::from(0f32).to_device(device),
                cx_loss: Tensor::from(0f32).to_device(device),
                h_loss: Tensor::from(0f32).to_device(device),
                w_loss: Tensor::from(0f32).to_device(device),
                obj_loss: Tensor::from(0f32).to_device(device),
                class_loss: Tensor::from(0f32).to_device(device),
            });
        }

        let indexes: Vec<_> = indexes.into_iter().map(Some).collect();
        let reduction = Reduction::Mean;

        let cy_loss = src
            .cy
            .index(&indexes)
            .mse_loss(&dst.cy.index(&indexes), reduction);
        let cx_loss = src
            .cx
            .index(&indexes)
            .mse_loss(&dst.cx.index(&indexes), reduction);
        let h_loss = src
            .h
            .index(&indexes)
            .mse_loss(&dst.h.index(&indexes), reduction);
        let w_loss = src
            .w
            .index(&indexes)
            .mse_loss(&dst.w.index(&indexes), reduction);
        let obj_loss = src.obj_logit.binary_cross_entropy_with_logits::<Tensor>(
            &dst.obj_logit.sigmoid(),
            None,
            None,
            reduction,
        );
        let class_loss = {
            let batch_indexes = indexes[0].as_ref();
            let anchor_indexes = indexes[2].as_ref();
            let row_indexes = indexes[3].as_ref();
            let col_indexes = indexes[4].as_ref();
            let indexes = &[
                batch_indexes,
                None,
                anchor_indexes,
                row_indexes,
                col_indexes,
            ];
            src.class_logit
                .index(indexes)
                .binary_cross_entropy_with_logits::<Tensor>(
                    &dst.class_logit.index(indexes).sigmoid(),
                    None,
                    None,
                    reduction,
                )
        };

        Ok(DetectionSimilarity {
            cy_loss,
            cx_loss,
            h_loss,
            w_loss,
            obj_loss,
            class_loss,
        })
    })
}

#[derive(Debug)]
pub struct DetectionSimilarity {
    pub cy_loss: Tensor,
    pub cx_loss: Tensor,
    pub h_loss: Tensor,
    pub w_loss: Tensor,
    pub obj_loss: Tensor,
    pub class_loss: Tensor,
}

impl DetectionSimilarity {
    pub fn position_loss(&self) -> Tensor {
        let Self {
            cy_loss, cx_loss, ..
        } = self;
        cy_loss + cx_loss
    }

    pub fn size_loss(&self) -> Tensor {
        let Self { h_loss, w_loss, .. } = self;
        h_loss + w_loss
    }

    pub fn total_loss(&self) -> Tensor {
        let Self {
            cy_loss,
            cx_loss,
            h_loss,
            w_loss,
            obj_loss,
            class_loss,
        } = self;
        cy_loss + cx_loss + h_loss + w_loss + obj_loss + class_loss
    }
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
        debug_assert!(penalty.is_all_finite());

        Ok(penalty)
    }
}
