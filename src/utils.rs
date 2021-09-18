use crate::common::*;
use num_traits::{NumCast, ToPrimitive};

pub trait SequentialExt {
    fn inspect<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) + Send;
}

impl SequentialExt for nn::Sequential {
    fn inspect<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) + Send,
    {
        self.add_fn(move |xs| {
            f(xs);
            xs.shallow_clone()
        })
    }
}

impl SequentialExt for nn::SequentialT {
    fn inspect<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) + Send,
    {
        self.add_fn(move |xs| {
            f(xs);
            xs.shallow_clone()
        })
    }
}

pub trait SequentialTExt {
    fn inspect_t<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) + Send;
}

impl SequentialTExt for nn::SequentialT {
    fn inspect_t<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) + Send,
    {
        self.add_fn_t(move |xs, train| {
            f(xs, train);
            xs.shallow_clone()
        })
    }
}

pub trait NumFrom<T> {
    fn num_from(from: T) -> Self;
}

impl<T, U> NumFrom<T> for U
where
    T: ToPrimitive,
    U: NumCast,
{
    fn num_from(from: T) -> Self {
        <Self as NumCast>::from(from).unwrap()
    }
}

pub trait NumInto<T> {
    fn num_into(self) -> T;
}

impl<T, U> NumInto<T> for U
where
    T: NumFrom<U>,
{
    fn num_into(self) -> T {
        T::num_from(self)
    }
}

pub trait DenseDetectionTensorListExt
where
    Self: Sized,
{
    fn from_labels<R>(
        labels: impl IntoIterator<Item = impl Borrow<RatioRectLabel<f64>>>,
        anchors: &[impl Borrow<[R]>],
        heights: &[usize],
        widths: &[usize],
        num_classes: usize,
    ) -> Result<Self>
    where
        R: Borrow<RatioSize<R64>>;

    fn is_all_finite(&self) -> bool;
}

impl DenseDetectionTensorListExt for DenseDetectionTensorList {
    fn from_labels<R>(
        labels: impl IntoIterator<Item = impl Borrow<RatioRectLabel<f64>>>,
        anchors: &[impl Borrow<[R]>],
        heights: &[usize],
        widths: &[usize],
        num_classes: usize,
    ) -> Result<Self>
    where
        R: Borrow<RatioSize<R64>>,
    {
        let list_len = anchors.len();
        ensure!(list_len == heights.len() && list_len == widths.len());
        ensure!(list_len == 1, "only list size == 1 is supported");

        let det = DenseDetectionTensor::from_labels(
            labels,
            anchors[0].borrow(),
            heights[0],
            widths[0],
            num_classes,
        )?;

        let list = DenseDetectionTensorListUnchecked { tensors: vec![det] }
            .build()
            .unwrap();

        Ok(list)
    }

    fn is_all_finite(&self) -> bool {
        self.tensors.iter().any(|tensor| tensor.is_all_finite())
    }
}

pub trait DenseDetectionTensorExt
where
    Self: Sized,
{
    fn from_labels<R>(
        labels: impl IntoIterator<Item = impl Borrow<RatioRectLabel<f64>>>,
        anchors: impl Borrow<[R]>,
        height: usize,
        width: usize,
        num_classes: usize,
    ) -> Result<Self>
    where
        R: Borrow<RatioSize<R64>>;

    fn detach(&self) -> Self;
    fn copy(&self) -> Self;
    fn is_all_finite(&self) -> bool;
}

impl DenseDetectionTensorExt for DenseDetectionTensor {
    fn from_labels<R>(
        labels: impl IntoIterator<Item = impl Borrow<RatioRectLabel<f64>>>,
        anchors: impl Borrow<[R]>,
        height: usize,
        width: usize,
        num_classes: usize,
    ) -> Result<Self>
    where
        R: Borrow<RatioSize<R64>>,
    {
        tch::no_grad(move || -> Result<_> {
            const MAX_ANCHOR_SCALE: f64 = 4.0;
            const SNAP_THRESH: f64 = 0.5;
            let anchor_scale_range = MAX_ANCHOR_SCALE.recip()..=MAX_ANCHOR_SCALE;

            let anchors: Vec<_> = anchors
                .borrow()
                .iter()
                .map(|anchor| anchor.borrow().to_owned())
                .collect();
            let num_anchors = anchors.len() as i64;
            let image_boundary =
                PixelTLBR::from_tlbr(0.0, 0.0, height as f64, width as f64).unwrap();
            let image_size = PixelSize::from_hw(height as f64, width as f64).unwrap();

            // crop and filter out bad labels
            let mut used_positions = HashSet::new();

            let assignments: Vec<_> = labels
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

                    Some((ratio_label, pixel_label))
                })
                .flat_map(|(ratio_label, pixel_label)| {
                    let cy = pixel_label.cy();
                    let cx = pixel_label.cx();
                    let cy_fract = cy.fract();
                    let cx_fract = cx.fract();
                    let row = cy.floor() as isize;
                    let col = cx.floor() as isize;

                    let pos_c = iter::once((row, col));
                    let pos_t = (cy_fract < SNAP_THRESH).then(|| (row - 1, col));
                    let pos_b = (cy_fract > 1.0 - SNAP_THRESH).then(|| (row + 1, col));
                    let pos_l = (cx_fract < SNAP_THRESH).then(|| (row, col - 1));
                    let pos_r = (cx_fract > 1.0 - SNAP_THRESH).then(|| (row, col + 1));

                    let ratio_label = Arc::new(ratio_label);
                    let pixel_label = Arc::new(pixel_label);

                    chain!(pos_c, pos_t, pos_b, pos_l, pos_r)
                        .filter(|(row, col)| {
                            (0..height as isize).contains(row) && (0..width as isize).contains(col)
                        })
                        .map(move |(row, col)| {
                            (
                                ratio_label.clone(),
                                pixel_label.clone(),
                                row as usize,
                                col as usize,
                            )
                        })
                })
                // filter by position constraint
                .filter(|&(_, ref pixel_label, row, col)| {
                    let [cy, cx, _, _] = pixel_label.cycxhw();
                    (0.0..=1.0).contains(&((cy - row as f64 + 0.5) / 2.0))
                        && (0.0..=1.0).contains(&((cx - col as f64 + 0.5) / 2.0))
                })
                .filter_map(|(ratio_label, _, row, col)| {
                    let position =
                        anchors
                            .iter()
                            .enumerate()
                            .find_map(|(anchor_index, anchor_size)| {
                                let h_scale = ratio_label.h() / anchor_size.h.raw();
                                let w_scale = ratio_label.w() / anchor_size.w.raw();
                                let position = (row, col, anchor_index);
                                let ok = !used_positions.contains(&position)
                                    && anchor_scale_range.contains(&h_scale)
                                    && anchor_scale_range.contains(&w_scale)
                                    && (0.0..=2.0).contains(&h_scale.sqrt())
                                    && (0.0..=2.0).contains(&w_scale.sqrt());
                                ok.then(|| position)
                            })?;

                    used_positions.insert(position);
                    Some((ratio_label, position))
                })
                .map(|(ratio_label, position)| -> Result<_> {
                    ensure!((0..num_classes).contains(&ratio_label.class));
                    Ok((ratio_label, position))
                })
                .try_collect()?;

            let (row_vec, col_vec, anchor_vec, cy_vec, cx_vec, h_vec, w_vec, class_vec) =
                assignments
                    .into_iter()
                    .map(|(ratio_label, (row, col, anchor_index))| {
                        let [cy, cx, h, w] = ratio_label.rect.cycxhw();
                        // dbg!(cy, cx, cy * height as f64, cx * width as f64, row, col);
                        // assert!(
                        //     (0.0..=1.0).contains(&((cy * height as f64 - row as f64 + 0.5) / 2.0))
                        //         && (0.0..=1.0)
                        //             .contains(&((cx * width as f64 - col as f64 + 0.5) / 2.0))
                        // );

                        let class = ratio_label.class;
                        (
                            row as i64,
                            col as i64,
                            anchor_index as i64,
                            cy as f32,
                            cx as f32,
                            h as f32,
                            w as f32,
                            class as i64,
                        )
                    })
                    .unzip_n_vec();

            let height = height as i64;
            let width = width as i64;
            let num_classes = num_classes as i64;
            let num_assignments = row_vec.len() as i64;
            let batch_tensor = Tensor::zeros(&[num_assignments], INT64_CPU);
            let zero_entry_tensor = Tensor::full(&[num_assignments], 0, INT64_CPU);
            let cy_entry_tensor = &zero_entry_tensor;
            let cx_entry_tensor = &zero_entry_tensor;
            let h_entry_tensor = &zero_entry_tensor;
            let w_entry_tensor = &zero_entry_tensor;
            let obj_entry_tensor = &zero_entry_tensor;
            let class_entry_tensor = Tensor::of_slice(&class_vec);
            let anchor_tensor = Tensor::of_slice(&anchor_vec);
            let row_tensor = Tensor::of_slice(&row_vec);
            let col_tensor = Tensor::of_slice(&col_vec);
            let cy_value_tensor = Tensor::of_slice(&cy_vec);
            let cx_value_tensor = Tensor::of_slice(&cx_vec);
            let h_value_tensor = Tensor::of_slice(&h_vec);
            let w_value_tensor = Tensor::of_slice(&w_vec);

            let mut cy_tensor = Tensor::arange(height, FLOAT_CPU)
                .to_kind(Kind::Float)
                .div(height as f64)
                .view([1, 1, 1, height, 1])
                .expand(&[1, 1, num_anchors, height, width], false)
                .copy();
            let mut cx_tensor = Tensor::arange(width, FLOAT_CPU)
                .to_kind(Kind::Float)
                .div(width as f64)
                .view([1, 1, 1, 1, width])
                .expand(&[1, 1, num_anchors, height, width], false)
                .copy();
            let mut h_tensor = Tensor::full(&[1, 1, num_anchors, height, width], 1e-4, FLOAT_CPU);
            let mut w_tensor = Tensor::full(&[1, 1, num_anchors, height, width], 1e-4, FLOAT_CPU);
            let obj_logit_tensor = Tensor::full(
                &[1, 1, num_anchors, height, width],
                -10.0, // logit of approx. 0.0001
                FLOAT_CPU,
            );
            let class_logit_tensor = Tensor::full(
                &[1, num_classes, num_anchors, height, width],
                -10.0, // logit of approx. 0.0001
                FLOAT_CPU,
            );

            let _ = cy_tensor.index_put_(
                &[
                    Some(&batch_tensor),
                    Some(cy_entry_tensor),
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
                    Some(cx_entry_tensor),
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
                    Some(h_entry_tensor),
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
                    Some(w_entry_tensor),
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
                    Some(obj_entry_tensor),
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
                cy: cy_tensor.set_requires_grad(false),
                cx: cx_tensor.set_requires_grad(false),
                h: h_tensor.set_requires_grad(false),
                w: w_tensor.set_requires_grad(false),
                obj_logit: obj_logit_tensor.set_requires_grad(false),
                class_logit: class_logit_tensor.set_requires_grad(false),
                anchors,
            }
            .build()
            .unwrap();

            debug_assert!(output.is_all_finite());

            Ok(output)
        })
    }

    fn detach(&self) -> Self {
        let DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            anchors,
        } = &**self;

        DenseDetectionTensorUnchecked {
            cy: cy.detach(),
            cx: cx.detach(),
            h: h.detach(),
            w: w.detach(),
            obj_logit: obj_logit.detach(),
            class_logit: class_logit.detach(),
            anchors: anchors.clone(),
        }
        .build()
        .unwrap()
    }

    fn copy(&self) -> Self {
        let DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            anchors,
        } = &**self;

        DenseDetectionTensorUnchecked {
            cy: cy.copy(),
            cx: cx.copy(),
            h: h.copy(),
            w: w.copy(),
            obj_logit: obj_logit.copy(),
            class_logit: class_logit.copy(),
            anchors: anchors.clone(),
        }
        .build()
        .unwrap()
    }

    fn is_all_finite(&self) -> bool {
        let DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            ..
        } = &**self;

        cy.is_all_finite()
            || cx.is_all_finite()
            || h.is_all_finite()
            || w.is_all_finite()
            || obj_logit.is_all_finite()
            || class_logit.is_all_finite()
    }
}
