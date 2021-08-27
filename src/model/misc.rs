use crate::common::*;
use pathfinding::kuhn_munkres::Weights;
use tch_modules::{DarkBatchNorm, DarkBatchNormInit, InstanceNorm, InstanceNormInit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingKind {
    Reflect,
    Replicate,
    Zero,
}

impl PaddingKind {
    pub fn build(self, lrtb: [usize; 4]) -> Pad2D {
        let [l, r, t, b] = lrtb;
        Pad2D {
            kind: self,
            lrtb: [l as i64, r as i64, t as i64, b as i64],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormKind {
    BatchNorm,
    InstanceNorm,
    None,
}

impl NormKind {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>, out_dim: i64) -> Norm {
        match self {
            Self::BatchNorm => {
                let norm = DarkBatchNormInit {
                    var_min: Some(1e-3),
                    var_max: Some(1e3),
                    ..Default::default()
                }
                .build(path, out_dim);
                Norm::BatchNorm(norm)
            }
            Self::InstanceNorm => {
                let norm = InstanceNormInit {
                    var_min: Some(1e-3),
                    var_max: Some(1e3),
                    ..Default::default()
                }
                .build(path, out_dim);
                Norm::InstanceNorm(norm)
            }
            Self::None => Norm::None,
        }
    }
}

#[derive(Debug)]
pub enum Norm {
    BatchNorm(DarkBatchNorm),
    InstanceNorm(InstanceNorm),
    None,
}

impl nn::ModuleT for Norm {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        match self {
            Self::BatchNorm(norm) => norm.forward_t(input, train),
            Self::InstanceNorm(norm) => norm.forward_t(input, train),
            Self::None => input.shallow_clone(),
        }
    }
}

#[derive(Debug)]
pub struct Pad2D {
    kind: PaddingKind,
    lrtb: [i64; 4],
}

impl nn::Module for Pad2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self.kind {
            PaddingKind::Reflect => xs.reflection_pad2d(&self.lrtb),
            PaddingKind::Replicate => xs.replication_pad1d(&self.lrtb),
            PaddingKind::Zero => {
                let [l, r, t, b] = self.lrtb;
                xs.zero_pad2d(l, r, t, b)
            }
        }
    }
}

pub trait DenseDetectionTensorExt
where
    Self: Sized,
{
    fn from_labels<'a>(
        labels: impl IntoIterator<Item = impl Borrow<RatioRectLabel<f64>>>,
        anchors: impl IntoIterator<Item = impl Into<Cow<'a, RatioSize<R64>>>>,
        height: usize,
        width: usize,
        num_classes: usize,
    ) -> Result<Self>;
}

impl DenseDetectionTensorExt for DenseDetectionTensor {
    fn from_labels<'a>(
        labels: impl IntoIterator<Item = impl Borrow<RatioRectLabel<f64>>>,
        anchors: impl IntoIterator<Item = impl Into<Cow<'a, RatioSize<R64>>>>,
        height: usize,
        width: usize,
        num_classes: usize,
    ) -> Result<Self> {
        tch::no_grad(move || -> Result<_> {
            const MAX_ANCHOR_SCALE: f64 = 4.0;
            let anchor_scale_range = MAX_ANCHOR_SCALE.recip()..=MAX_ANCHOR_SCALE;

            let anchors: Vec<_> = anchors
                .into_iter()
                .map(|anchor| anchor.into().into_owned())
                .collect();
            let num_anchors = anchors.len();

            // filter out out-of-bound labels
            let labels: Vec<_> = {
                let image_boundary =
                    PixelTLBR::from_tlbr(0.0, 0.0, height as f64, width as f64).unwrap();
                let image_size = PixelSize::from_hw(height as f64, width as f64).unwrap();

                labels
                    .into_iter()
                    .filter_map(|ratio_label| {
                        const MIN_SIZE: f64 = 1.0;
                        const MAX_HW_RATIO: f64 = 4.0;

                        let pixel_label = ratio_label.borrow().to_pixel_label(&image_size);

                        let bbox = pixel_label.intersect_with(&image_boundary)?;

                        if bbox.h() < MIN_SIZE && bbox.w() < MIN_SIZE {
                            return None;
                        }

                        if !(MAX_HW_RATIO.recip()..=MAX_HW_RATIO).contains(&(bbox.h() / bbox.w())) {
                            return None;
                        }

                        let pixel_label = PixelRectLabel {
                            rect: bbox.into(),
                            class: pixel_label.class,
                        };
                        let ratio_label = pixel_label.to_ratio_label(&image_size);

                        let cy = ratio_label.cy();
                        let cx = ratio_label.cx();
                        let row = cy.floor() as usize;
                        let col = cx.floor() as usize;

                        if !(0..height).contains(&row) || !(0..width).contains(&col) {
                            return None;
                        }

                        Some((ratio_label, row, col))
                    })
                    .map(|(ratio_label, row, col)| -> Result<_> {
                        ensure!((0..num_classes).contains(&ratio_label.class));
                        Ok((ratio_label, row, col))
                    })
                    .try_collect()?
            };
            let num_labels = labels.len();

            // list candidates of position to label relations
            let pos_label_relations: IndexSet<_> =
                iproduct!(labels.iter().enumerate(), anchors.iter().enumerate())
                    .filter_map(
                        |((label_index, &(ref label, row, col)), (anchor_index, anchor))| {
                            let h_scale = label.h() / anchor.h.raw();
                            let w_scale = label.w() / anchor.w.raw();
                            let position = (row, col, anchor_index);

                            let ok = anchor_scale_range.contains(&h_scale)
                                && anchor_scale_range.contains(&w_scale);
                            ok.then(|| (position, label_index))
                        },
                    )
                    .collect();

            // list occurred positions
            let positions: IndexSet<_> = pos_label_relations
                .iter()
                .map(|&(position_index, _)| position_index)
                .collect();

            struct WeightTable {
                is_neg: bool,
                max_rows: usize,
                max_cols: usize,
                relations: HashSet<(usize, usize)>,
            }

            impl Weights<isize> for WeightTable {
                fn rows(&self) -> usize {
                    self.max_rows
                }

                fn columns(&self) -> usize {
                    self.max_cols
                }

                fn at(&self, row: usize, col: usize) -> isize {
                    if self.relations.contains(&(row, col)) {
                        if self.is_neg {
                            1
                        } else {
                            -1
                        }
                    } else {
                        0
                    }
                }

                fn neg(&self) -> Self
                where
                    Self: Sized,
                {
                    let Self {
                        is_neg,
                        max_rows,
                        max_cols,
                        ref relations,
                    } = *self;

                    Self {
                        is_neg: !is_neg,
                        max_rows,
                        max_cols,
                        relations: relations.clone(),
                    }
                }
            }

            let table = WeightTable {
                is_neg: false,
                max_rows: positions.len(),
                max_cols: labels.len(),
                relations: pos_label_relations
                    .iter()
                    .map(|&(position, label_index)| {
                        let row = positions.get_index_of(&position).unwrap();
                        let col = label_index;
                        (row, col)
                    })
                    .collect(),
            };

            let (_, assignments) = pathfinding::kuhn_munkres::kuhn_munkres(&table);

            let (row_vec, col_vec, anchor_vec, cy_vec, cx_vec, h_vec, w_vec, class_vec) =
                assignments
                    .iter()
                    .enumerate()
                    .map(|(table_row, &table_col)| {
                        let (image_row, image_col, anchor_index) =
                            *positions.get_index(table_row).unwrap();
                        let label_index = table_col;
                        let (label, _, _) = &labels[label_index];
                        (image_row, image_col, anchor_index, label)
                    })
                    .map(|(row, col, anchor, label)| {
                        let [cy, cx, h, w] = label.rect.cycxhw();
                        let class = label.class;
                        (
                            row as i64,
                            col as i64,
                            anchor as i64,
                            cy,
                            cx,
                            h,
                            w,
                            class as i64,
                        )
                    })
                    .unzip_n_vec();

            let batch_tensor = Tensor::zeros(&[num_labels as i64], INT64_CPU);
            let cy_entry_tensor = Tensor::full(&[num_labels as i64], 0, INT64_CPU);
            let cx_entry_tensor = Tensor::full(&[num_labels as i64], 1, INT64_CPU);
            let h_entry_tensor = Tensor::full(&[num_labels as i64], 2, INT64_CPU);
            let w_entry_tensor = Tensor::full(&[num_labels as i64], 3, INT64_CPU);
            let obj_entry_tensor = Tensor::full(&[num_labels as i64], 4, INT64_CPU);
            let class_entry_tensor = {
                let vec: Vec<_> = class_vec.iter().cloned().map(|index| index + 5).collect();
                Tensor::of_slice(&vec)
            };
            let anchor_tensor = Tensor::of_slice(&anchor_vec);
            let row_tensor = Tensor::of_slice(&row_vec);
            let col_tensor = Tensor::of_slice(&col_vec);
            let cy_value_tensor = Tensor::of_slice(&cy_vec);
            let cx_value_tensor = Tensor::of_slice(&cx_vec);
            let h_value_tensor = Tensor::of_slice(&h_vec);
            let w_value_tensor = Tensor::of_slice(&w_vec);

            let mut cy_tensor = Tensor::zeros(
                &[1, 1, num_anchors as i64, height as i64, width as i64],
                FLOAT_CPU,
            );
            let mut cx_tensor = Tensor::zeros(
                &[1, 1, num_anchors as i64, height as i64, width as i64],
                FLOAT_CPU,
            );
            let mut h_tensor = Tensor::zeros(
                &[1, 1, num_anchors as i64, height as i64, width as i64],
                FLOAT_CPU,
            );
            let mut w_tensor = Tensor::zeros(
                &[1, 1, num_anchors as i64, height as i64, width as i64],
                FLOAT_CPU,
            );
            let obj_logit_tensor = Tensor::rand(
                &[1, 1, num_anchors as i64, height as i64, width as i64],
                FLOAT_CPU,
            );
            let class_logit_tensor = Tensor::zeros(
                &[
                    1,
                    num_classes as i64,
                    num_anchors as i64,
                    height as i64,
                    width as i64,
                ],
                FLOAT_CPU,
            );

            let _ = cy_tensor.index_put_(
                &[
                    Some(&batch_tensor),
                    Some(&cy_entry_tensor),
                    Some(&anchor_tensor),
                    Some(&row_tensor),
                    Some(&col_tensor),
                ],
                &cy_value_tensor,
                false,
            );
            let _ = cx_tensor.index_put_(
                &[
                    Some(&batch_tensor),
                    Some(&cx_entry_tensor),
                    Some(&anchor_tensor),
                    Some(&row_tensor),
                    Some(&col_tensor),
                ],
                &cx_value_tensor,
                false,
            );
            let _ = h_tensor.index_put_(
                &[
                    Some(&batch_tensor),
                    Some(&h_entry_tensor),
                    Some(&anchor_tensor),
                    Some(&row_tensor),
                    Some(&col_tensor),
                ],
                &h_value_tensor,
                false,
            );
            let _ = w_tensor.index_put_(
                &[
                    Some(&batch_tensor),
                    Some(&w_entry_tensor),
                    Some(&anchor_tensor),
                    Some(&row_tensor),
                    Some(&col_tensor),
                ],
                &w_value_tensor,
                false,
            );
            let _ = obj_logit_tensor
                .index(&[
                    Some(&batch_tensor),
                    Some(&obj_entry_tensor),
                    Some(&anchor_tensor),
                    Some(&row_tensor),
                    Some(&col_tensor),
                ])
                .fill_(5.0); // logit of 0.993
            let _ = class_logit_tensor
                .index(&[
                    Some(&batch_tensor),
                    Some(&class_entry_tensor),
                    Some(&anchor_tensor),
                    Some(&row_tensor),
                    Some(&col_tensor),
                ])
                .fill_(5.0); // logit of 0.993

            let output = DenseDetectionTensorUnchecked {
                cy: cy_tensor,
                cx: cx_tensor,
                h: h_tensor,
                w: w_tensor,
                obj_logit: obj_logit_tensor,
                class_logit: class_logit_tensor,
                anchors,
            }
            .build()
            .unwrap();

            Ok(output)
        })
    }
}
