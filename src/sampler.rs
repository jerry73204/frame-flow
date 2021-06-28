use crate::common::*;
use num_traits::ToPrimitive;

#[derive(Debug, Clone)]
pub struct DetectionSamplerInit {
    pub features: Vec<Feature>,
    pub min_bbox_size: Option<R64>,
    pub min_bbox_cropping_ratio: Option<R64>,
    pub anchor_scale_thresh: R64,
}

impl DetectionSamplerInit {
    pub fn build(self) -> Result<DetectionSampler> {
        ensure!(self.anchor_scale_thresh >= 1.0);

        if let Some(min_bbox_size) = self.min_bbox_size {
            ensure!((0.0..=1.0).contains(&min_bbox_size.raw()));
        }

        if let Some(min_bbox_cropping_ratio) = self.min_bbox_cropping_ratio {
            ensure!((0.0..=1.0).contains(&min_bbox_cropping_ratio.raw()));
        }

        Ok(DetectionSampler { init: self })
    }
}

#[derive(Debug)]
pub struct DetectionSampler {
    init: DetectionSamplerInit,
}

impl DetectionSampler {
    pub fn sample<T>(&self, boxes: impl IntoIterator<Item = impl Borrow<RatioRectLabel<T>>>)
    where
        T: Num + Copy + ToPrimitive,
    {
        let mut rng = rand::thread_rng();
        let init = &self.init;

        let rect_pos_relations: HashMap<_, Vec<_>> = boxes
            .into_iter()
            .filter_map(|rect| {
                let rect = {
                    let rect = rect.borrow();
                    let rect = RatioRectLabel {
                        rect: rect.to_cycxhw().cast::<R64>().unwrap(),
                        class: rect.class,
                    };
                    ByAddress(Arc::new(rect))
                };

                if let Some(min_bbox_size) = init.min_bbox_size {
                    if rect.h() < min_bbox_size || rect.w() < min_bbox_size {
                        return None;
                    }
                }

                Some(rect)
            })
            .flat_map(|rect| {
                let anchor_scale_range = {
                    let max = init.anchor_scale_thresh;
                    let min = init.anchor_scale_thresh.recip();
                    min..=max
                };

                let matchings: Vec<_> = init
                    .features
                    .iter()
                    .enumerate()
                    .filter_map(|(feature_index, feature)| {
                        let cy_grid = rect.cy().raw() / feature.size.h as f64;
                        let cx_grid = rect.cx().raw() / feature.size.w as f64;
                        let ok = (0.0..=(feature.size.h as f64)).contains(&cy_grid)
                            && (0.0..=(feature.size.w as f64)).contains(&cx_grid);
                        ok.then(|| (feature_index, feature, cy_grid, cx_grid))
                    })
                    .flat_map(|(feature_index, feature, cy_grid, cx_grid)| {
                        let snap_thresh = 0.5;

                        let row = cy_grid.floor() as usize;
                        let col = cx_grid.floor() as usize;
                        let cy_fract = cy_grid.fract();
                        let cx_fract = cx_grid.fract();

                        let positions = {
                            let orig = iter::once((row, col));
                            let top = (cy_fract < snap_thresh && row > 0).then(|| (row - 1, col));
                            let bottom = (cy_fract > 1.0 - snap_thresh && row < feature.size.h - 1)
                                .then(|| (row + 1, col));
                            let left = (cx_fract < snap_thresh && col > 0).then(|| (row, col - 1));
                            let right = (cx_fract > 1.0 - snap_thresh && col < feature.size.w - 1)
                                .then(|| (row, col + 1));
                            orig.chain(top).chain(bottom).chain(left).chain(right)
                        };

                        let anchors = feature.anchors.iter().enumerate().filter({
                            |(anchor_index, anchor)| {
                                anchor_scale_range.contains(&(rect.h() / anchor.h))
                                    && anchor_scale_range.contains(&(rect.w() / anchor.w))
                            }
                        });

                        let candidates: Vec<_> = iproduct!(positions, anchors)
                            .map(|args| {
                                let ((row, col), (anchor_index, _anchor)) = args;
                                (rect.clone(), (feature_index, anchor_index, row, col))
                            })
                            .collect();

                        candidates
                    })
                    .collect();

                matchings
            })
            .into_group_map();

        let pos_rect_relations: HashMap<_, Vec<_>> = rect_pos_relations
            .into_iter()
            .map(|(rect, positions)| {
                let position = *positions.choose(&mut rng).unwrap();
                (position, rect)
            })
            .into_group_map();

        let pos_rect_matchings = pos_rect_relations.into_iter().map(|(position, rects)| {
            let rect = rects.choose(&mut rng).unwrap().clone();
            (position, rect)
        });

        init.features.iter().map(|feature| {
            let y_offsets = (Tensor::arange(feature.size.h as i64, FLOAT_CPU)
                / feature.size.h as f64)
                .set_requires_grad(false);
            let x_offsets = (Tensor::arange(feature.size.w as i64, FLOAT_CPU)
                / feature.size.w as f64)
                .set_requires_grad(false);

            let (anchor_h_vec, anchor_w_vec) = feature
                .anchors
                .iter()
                .map(|anchor| {
                    let anchor = anchor.cast::<f32>().unwrap();
                    (anchor.h, anchor.w)
                })
                .unzip_n_vec();

            let anchor_h = Tensor::of_slice(&anchor_h_vec).set_requires_grad(false);
            let anchor_w = Tensor::of_slice(&anchor_w_vec).set_requires_grad(false);
        });
    }
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub size: GridSize<usize>,
    pub anchors: Vec<RatioSize<R64>>,
}
