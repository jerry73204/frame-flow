use crate::{common::*, message as msg, model::DetectionSimilarity, FILE_STRFTIME};

pub async fn logging_worker(
    log_dir: impl AsRef<Path>,
    mut log_rx: mpsc::Receiver<msg::LogMessage>,
    save_motion_field_image: bool,
    save_files: bool,
) -> Result<()> {
    let log_dir = log_dir.as_ref();
    let event_dir = log_dir.join("events");
    let image_dir = log_dir.join("image");
    tokio::fs::create_dir_all(&event_dir).await?;

    let mut event_writer = {
        let event_path_prefix = event_dir
            .join("frame-flow")
            .into_os_string()
            .into_string()
            .unwrap();

        EventWriterInit::default()
            .from_prefix_async(event_path_prefix, None)
            .await?
    };

    loop {
        let msg = match log_rx.recv().await {
            Some(msg) => msg,
            None => break,
        };

        match msg {
            msg::LogMessage::Loss(msg::Loss {
                step,
                learning_rate,

                detector_loss,
                discriminator_loss,
                generator_loss,
                retraction_identity_loss,
                retraction_identity_similarity,
                triangular_identity_loss,
                triangular_identity_similarity,
                forward_consistency_loss,
                forward_consistency_similarity_seq,
                backward_consistency_gen_loss,
                backward_consistency_disc_loss,

                detector_weights,
                generator_weights,
                discriminator_weights,
                transformer_weights,
                image_seq_discriminator_weights,

                gt_det_seq,
                ground_truth_image_seq,
                detector_det_seq,
                generator_image_seq,
                transformer_det_seq,
                transformer_image_seq,

                motion_potential_pixel_seq,
                motion_field_pixel_seq,
                attention_image_seq,
            }) => {
                let step = step as i64;

                let sub_image_dir = {
                    let dir = image_dir.join(format!(
                        "{:08}_{}",
                        step,
                        Local::now().format(FILE_STRFTIME)
                    ));
                    fs::create_dir_all(&dir)?;
                    Arc::new(dir)
                };

                event_writer
                    .write_scalar_async("params/learning_rate", step, learning_rate as f32)
                    .await?;

                if let Some(loss) = detector_loss {
                    event_writer
                        .write_scalar_async("loss/detector_loss", step, loss as f32)
                        .await?;
                }

                if let Some(loss) = generator_loss {
                    event_writer
                        .write_scalar_async("loss/generator_loss", step, loss as f32)
                        .await?;
                }

                if let Some(loss) = discriminator_loss {
                    event_writer
                        .write_scalar_async("loss/discriminator_loss", step, loss as f32)
                        .await?;
                }

                if let Some(loss) = retraction_identity_loss {
                    event_writer
                        .write_scalar_async("loss/retraction_identity_loss", step, loss as f32)
                        .await?;
                }

                if let Some(loss) = triangular_identity_loss {
                    event_writer
                        .write_scalar_async("loss/triangular_identity_loss", step, loss as f32)
                        .await?;
                }

                if let Some(loss) = forward_consistency_loss {
                    event_writer
                        .write_scalar_async("loss/forward_consistency_loss", step, loss as f32)
                        .await?;
                }

                if let Some(loss) = backward_consistency_gen_loss {
                    event_writer
                        .write_scalar_async("loss/backward_consistency_gen_loss", step, loss as f32)
                        .await?;
                }

                if let Some(loss) = backward_consistency_disc_loss {
                    event_writer
                        .write_scalar_async(
                            "loss/backward_consistency_disc_loss",
                            step,
                            loss as f32,
                        )
                        .await?;
                }

                if let Some(similarity) = retraction_identity_similarity {
                    let DetectionSimilarity {
                        cy_loss,
                        cx_loss,
                        h_loss,
                        w_loss,
                        obj_loss,
                        class_loss,
                    } = similarity;

                    let position_loss = cy_loss + cx_loss;
                    let size_loss = h_loss + w_loss;

                    let position_loss_vec: Vec<f32> = position_loss.try_into().unwrap();
                    let size_loss_vec: Vec<f32> = size_loss.try_into().unwrap();
                    let obj_loss_vec: Vec<f32> = obj_loss.try_into().unwrap();
                    let class_loss_vec: Vec<f32> = class_loss.try_into().unwrap();

                    for (bidx, (position_loss, size_loss, obj_loss, class_loss)) in izip!(
                        position_loss_vec,
                        size_loss_vec,
                        obj_loss_vec,
                        class_loss_vec
                    )
                    .enumerate()
                    {
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "retraction_identity_similarity/position_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                position_loss,
                            )
                            .await?;
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "retraction_identity_similarity/size_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                size_loss,
                            )
                            .await?;
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "retraction_identity_similarity/obj_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                obj_loss,
                            )
                            .await?;
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "retraction_identity_similarity/class_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                class_loss,
                            )
                            .await?;
                    }
                }

                if let Some(similarity) = triangular_identity_similarity {
                    let DetectionSimilarity {
                        cy_loss,
                        cx_loss,
                        h_loss,
                        w_loss,
                        obj_loss,
                        class_loss,
                    } = similarity;

                    let position_loss = cy_loss + cx_loss;
                    let size_loss = h_loss + w_loss;

                    let position_loss_vec: Vec<f32> = position_loss.try_into().unwrap();
                    let size_loss_vec: Vec<f32> = size_loss.try_into().unwrap();
                    let obj_loss_vec: Vec<f32> = obj_loss.try_into().unwrap();
                    let class_loss_vec: Vec<f32> = class_loss.try_into().unwrap();

                    for (bidx, (position_loss, size_loss, obj_loss, class_loss)) in izip!(
                        position_loss_vec,
                        size_loss_vec,
                        obj_loss_vec,
                        class_loss_vec
                    )
                    .enumerate()
                    {
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "triangular_identity_similarity/position_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                position_loss,
                            )
                            .await?;
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "triangular_identity_similarity/size_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                size_loss,
                            )
                            .await?;
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "triangular_identity_similarity/obj_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                obj_loss,
                            )
                            .await?;
                        event_writer
                            .write_scalar_async(
                                format!(
                                    "triangular_identity_similarity/class_loss/batch_{:04}",
                                    bidx
                                ),
                                step,
                                class_loss,
                            )
                            .await?;
                    }
                }

                if let Some(seq) = forward_consistency_similarity_seq {
                    let (position_loss_seq, size_loss_seq, obj_loss_seq, class_loss_seq) = seq
                        .into_iter()
                        .map(|similarity| {
                            let position_loss = similarity.position_loss();
                            let size_loss = similarity.size_loss();
                            let DetectionSimilarity {
                                obj_loss,
                                class_loss,
                                ..
                            } = similarity;

                            (position_loss, size_loss, obj_loss, class_loss)
                        })
                        .unzip_n_vec();

                    save_scalar_seq("position_loss", &*sub_image_dir, &position_loss_seq)?;
                    save_scalar_seq("size_loss", &*sub_image_dir, &size_loss_seq)?;
                    save_scalar_seq("obj_loss", &*sub_image_dir, &obj_loss_seq)?;
                    save_scalar_seq("class_loss", &*sub_image_dir, &class_loss_seq)?;

                    save_scalar_seq_to_tfrecord(
                        &mut event_writer,
                        "forward_consistency_similarity/position_loss",
                        step,
                        position_loss_seq,
                    )
                    .await?;
                    save_scalar_seq_to_tfrecord(
                        &mut event_writer,
                        "forward_consistency_similarity/size_loss",
                        step,
                        size_loss_seq,
                    )
                    .await?;
                    save_scalar_seq_to_tfrecord(
                        &mut event_writer,
                        "forward_consistency_similarity/obj_loss",
                        step,
                        obj_loss_seq,
                    )
                    .await?;
                    save_scalar_seq_to_tfrecord(
                        &mut event_writer,
                        "forward_consistency_similarity/class_loss",
                        step,
                        class_loss_seq,
                    )
                    .await?;

                    // for (seq_index, similarity) in seq.into_iter().enumerate() {
                    //     let DetectionSimilarity {
                    //         cy_loss,
                    //         cx_loss,
                    //         h_loss,
                    //         w_loss,
                    //         obj_loss,
                    //         class_loss,
                    //     } = similarity;

                    //     let position_loss = cy_loss + cx_loss;
                    //     let size_loss = h_loss + w_loss;

                    //     event_writer
                    //         .write_scalar_async(
                    //             format!(
                    //                 "forward_consistency_similarity/position_loss/seq_{}",
                    //                 seq_index
                    //             ),
                    //             step,
                    //             f32::from(position_loss),
                    //         )
                    //         .await?;
                    //     event_writer
                    //         .write_scalar_async(
                    //             format!(
                    //                 "forward_consistency_similarity/size_loss/seq_{}",
                    //                 seq_index
                    //             ),
                    //             step,
                    //             f32::from(size_loss),
                    //         )
                    //         .await?;
                    //     event_writer
                    //         .write_scalar_async(
                    //             format!(
                    //                 "forward_consistency_similarity/obj_loss/seq_{}",
                    //                 seq_index
                    //             ),
                    //             step,
                    //             f32::from(obj_loss),
                    //         )
                    //         .await?;
                    //     event_writer
                    //         .write_scalar_async(
                    //             format!(
                    //                 "forward_consistency_similarity/class_loss/seq_{}",
                    //                 seq_index
                    //             ),
                    //             step,
                    //             f32::from(class_loss),
                    //         )
                    //         .await?;
                    // }
                }

                // log weights and gradients
                if let Some(msg::WeightsAndGrads { weights, grads }) = detector_weights {
                    for (name, weight) in weights {
                        event_writer
                            .write_scalar_async(
                                format!("detector_weights/{}", name),
                                step,
                                weight as f32,
                            )
                            .await?;
                    }

                    for (name, grad) in grads {
                        event_writer
                            .write_scalar_async(
                                format!("detector_gradients/{}", name),
                                step,
                                grad as f32,
                            )
                            .await?;
                    }
                }

                if let Some(msg::WeightsAndGrads { weights, grads }) = discriminator_weights {
                    for (name, weight) in weights {
                        event_writer
                            .write_scalar_async(
                                format!("discriminator_weights/{}", name),
                                step,
                                weight as f32,
                            )
                            .await?;
                    }

                    for (name, grad) in grads {
                        event_writer
                            .write_scalar_async(
                                format!("discriminator_gradients/{}", name),
                                step,
                                grad as f32,
                            )
                            .await?;
                    }
                }

                if let Some(msg::WeightsAndGrads { weights, grads }) = generator_weights {
                    for (name, weight) in weights {
                        event_writer
                            .write_scalar_async(
                                format!("generator_weights/{}", name),
                                step,
                                weight as f32,
                            )
                            .await?;
                    }

                    for (name, grad) in grads {
                        event_writer
                            .write_scalar_async(
                                format!("generator_gradients/{}", name),
                                step,
                                grad as f32,
                            )
                            .await?;
                    }
                }

                if let Some(msg::WeightsAndGrads { weights, grads }) = transformer_weights {
                    for (name, weight) in weights {
                        event_writer
                            .write_scalar_async(
                                format!("transformer_weights/{}", name),
                                step,
                                weight as f32,
                            )
                            .await?;
                    }

                    for (name, grad) in grads {
                        event_writer
                            .write_scalar_async(
                                format!("transformer_gradients/{}", name),
                                step,
                                grad as f32,
                            )
                            .await?;
                    }
                }

                if let Some(msg::WeightsAndGrads { weights, grads }) =
                    image_seq_discriminator_weights
                {
                    for (name, weight) in weights {
                        event_writer
                            .write_scalar_async(
                                format!("image_seq_discriminator_weights/{}", name),
                                step,
                                weight as f32,
                            )
                            .await?;
                    }

                    for (name, grad) in grads {
                        event_writer
                            .write_scalar_async(
                                format!("image_seq_discriminator_gradients/{}", name),
                                step,
                                grad as f32,
                            )
                            .await?;
                    }
                }

                let save_image_seq_async =
                    |name: &'static str, dir: Arc<PathBuf>, seq: Vec<Tensor>| async move {
                        let seq = tokio::task::spawn_blocking(move || -> Result<_> {
                            if save_files {
                                save_image_seq(name, &**dir, &seq)?;
                            }
                            Ok(seq)
                        })
                        .await??;

                        Fallible::Ok(seq)
                    };

                if let Some(seq) = ground_truth_image_seq {
                    let seq =
                        save_image_seq_async("ground_truth", sub_image_dir.clone(), seq).await?;
                    save_image_seq_to_tfrecord(&mut event_writer, "ground_truth", step, seq)
                        .await?;
                }

                if let Some(seq) = generator_image_seq {
                    let seq =
                        save_image_seq_async("generator_image", sub_image_dir.clone(), seq).await?;
                    save_image_seq_to_tfrecord(&mut event_writer, "generator_image", step, seq)
                        .await?;
                }

                if let Some(seq) = transformer_image_seq {
                    let seq = save_image_seq_async("transformer_image", sub_image_dir.clone(), seq)
                        .await?;

                    save_image_seq_to_tfrecord(&mut event_writer, "transformer_image", step, seq)
                        .await?;
                }

                if let Some(seq) = gt_det_seq {
                    let objectness_seq: Vec<_> = seq
                        .into_iter()
                        .map(|det| {
                            assert!(det.tensors.len() == 1);
                            let (max, _argmax) = det.tensors[0].obj_prob().max_dim(2, false);
                            max
                        })
                        .collect();

                    let objectness_seq = save_image_seq_async(
                        "ground_truth_objectness",
                        sub_image_dir.clone(),
                        objectness_seq,
                    )
                    .await?;

                    save_image_seq_to_tfrecord(
                        &mut event_writer,
                        "ground_truth_objectness",
                        step,
                        objectness_seq,
                    )
                    .await?;
                }

                if let Some(seq) = detector_det_seq {
                    let objectness_seq: Vec<_> = seq
                        .into_iter()
                        .map(|det| {
                            assert!(det.tensors.len() == 1);
                            let (max, _argmax) = det.tensors[0].obj_prob().max_dim(2, false);
                            max
                        })
                        .collect();

                    let objectness_seq = save_image_seq_async(
                        "detector_objectness",
                        sub_image_dir.clone(),
                        objectness_seq,
                    )
                    .await?;

                    save_image_seq_to_tfrecord(
                        &mut event_writer,
                        "detector_objectness",
                        step,
                        objectness_seq,
                    )
                    .await?;
                }

                if let Some(seq) = transformer_det_seq {
                    let objectness_seq: Vec<_> = seq
                        .into_iter()
                        .map(|det| {
                            assert!(det.tensors.len() == 1);
                            let (max, _argmax) = det.tensors[0].obj_prob().max_dim(2, false);
                            max
                        })
                        .collect();

                    let objectness_seq = save_image_seq_async(
                        "transformer_objectness",
                        sub_image_dir.clone(),
                        objectness_seq,
                    )
                    .await?;

                    save_image_seq_to_tfrecord(
                        &mut event_writer,
                        "transformer_objectness",
                        step,
                        objectness_seq,
                    )
                    .await?;
                }

                if let Some(seq) = motion_potential_pixel_seq {
                    let seq = save_image_seq_async("motion_potential", sub_image_dir.clone(), seq)
                        .await?;
                    save_image_seq_to_tfrecord(&mut event_writer, "motion_potential", step, seq)
                        .await?;
                }

                if let Some(motion_field_pixel_seq) = motion_field_pixel_seq {
                    let (bsize, _, hsize, wsize) = motion_field_pixel_seq[0].size4().unwrap();

                    if save_motion_field_image {
                        let ident_grid = {
                            let theta = Tensor::from_cv([[[1f32, 0.0, 0.0], [0.0, 1.0, 0.0]]])
                                .expand(&[bsize, 2, 3], false);
                            Tensor::affine_grid_generator(&theta, &[bsize, 1, hsize, wsize], false)
                        };

                        let field_image_seq: Vec<_> = motion_field_pixel_seq
                            .iter()
                            .map(|field_pixel| {
                                // dbg!(field.abs().max());

                                let dx_pixel = field_pixel.i((.., 0..1, .., ..));
                                let dy_pixel = field_pixel.i((.., 1..2, .., ..));

                                let dx_grid = dx_pixel * 2.0 / wsize as f64;
                                let dy_grid = dy_pixel * 2.0 / hsize as f64;
                                let field_grid = Tensor::cat(&[dx_grid, dy_grid], 1);

                                let src_grid = &ident_grid - field_grid.permute(&[0, 2, 3, 1]);
                                let src_x_grid = src_grid.i((.., .., .., 0..1));
                                let src_y_grid = src_grid.i((.., .., .., 1..2));

                                let src_x_pixel = (src_x_grid + 1.0) * wsize as f64 * 0.5 - 0.5;
                                let src_y_pixel = (src_y_grid + 1.0) * hsize as f64 * 0.5 - 0.5;

                                const SCALE: f64 = 8.0;

                                let images: Vec<_> = (0..bsize)
                                    .map(|bindex| {
                                        let mut image = core_cv::Mat::zeros(
                                            (hsize as f64 * SCALE) as i32,
                                            (wsize as f64 * SCALE) as i32,
                                            core_cv::CV_32FC3,
                                        )
                                        .unwrap()
                                        .to_mat()
                                        .unwrap();

                                        iproduct!(0..hsize, 0..wsize).for_each(|(row, col)| {
                                            let tgt_y = row as f64;
                                            let tgt_x = col as f64;
                                            let src_y =
                                                f64::from(src_y_pixel.i((bindex, row, col, 0)));
                                            let src_x =
                                                f64::from(src_x_pixel.i((bindex, row, col, 0)));

                                            let src_pt = core_cv::Point {
                                                x: (src_x * SCALE) as i32,
                                                y: (src_y * SCALE) as i32,
                                            };
                                            let dst_pt = core_cv::Point {
                                                x: (tgt_x * SCALE) as i32,
                                                y: (tgt_y * SCALE) as i32,
                                            };
                                            let color = {
                                                let dx = tgt_x - src_x;
                                                let dy = tgt_y - src_y;

                                                let angle = dy.atan2(dx);
                                                let magnitude = (dx.powi(2) + dy.powi(2)).sqrt();

                                                let color: Srgb<_> = Hsv::from_components((
                                                    RgbHue::from_radians(angle),
                                                    1.0,
                                                    (magnitude / 8.0).max(1.0),
                                                ))
                                                .into_color();
                                                let (r, g, b) = color.into_components();
                                                core_cv::Scalar::new(r, g, b, 0.0)
                                            };

                                            imgproc::arrowed_line(
                                                &mut image, src_pt, dst_pt, color,
                                                1,   // thickness
                                                8,   // line type
                                                0,   // shift
                                                0.1, // tip_length
                                            )
                                            .unwrap();
                                        });

                                        let image = Tensor::try_from_cv(image)
                                            .unwrap()
                                            .permute(&[2, 0, 1])
                                            .unsqueeze(0);
                                        image
                                    })
                                    .collect();
                                let field_image = Tensor::cat(&images, 0);

                                field_image
                            })
                            .collect();

                        let field_image_seq = save_image_seq_async(
                            "motion_field",
                            sub_image_dir.clone(),
                            field_image_seq,
                        )
                        .await?;

                        save_image_seq_to_tfrecord(
                            &mut event_writer,
                            "motion_field",
                            step,
                            field_image_seq,
                        )
                        .await?;
                    }

                    let max_motion_field_magnitude: R64 = motion_field_pixel_seq
                        .iter()
                        .map(|field| {
                            let max: f64 = field
                                .pow(2)
                                .sum_dim_intlist(&[1], true, Kind::Float)
                                .sqrt()
                                .max()
                                .into();
                            r64(max)
                        })
                        .max()
                        .unwrap();

                    event_writer
                        .write_scalar_async(
                            "max_motion_field_magnitude",
                            step,
                            max_motion_field_magnitude.raw() as f32,
                        )
                        .await?;

                    // let (motion_dx_seq, motion_dy_seq) = seq
                    //     .into_iter()
                    //     .map(|motion_field| {
                    //         let motion_dx = motion_field.i((.., 0..1, .., ..));
                    //         let motion_dy = motion_field.i((.., 1..2, .., ..));
                    //         (motion_dx, motion_dy)
                    //     })
                    //     .unzip_n_vec();

                    // let motion_dx_seq =
                    //     save_image_seq_async("motion_dx", sub_image_dir.clone(), motion_dx_seq)
                    //         .await?;
                    // let motion_dy_seq =
                    //     save_image_seq_async("motion_dy", sub_image_dir.clone(), motion_dy_seq)
                    //         .await?;

                    // save_image_seq_to_tfrecord(&mut event_writer, "motion_dx", step, motion_dx_seq)
                    //     .await?;
                    // save_image_seq_to_tfrecord(&mut event_writer, "motion_dy", step, motion_dy_seq)
                    //     .await?;
                }

                if let Some(seq) = attention_image_seq {
                    let seq: Vec<_> = seq
                        .into_iter()
                        .map(|attention_image| {
                            let (bsize, _one, attention_h, attention_w, image_h, image_w) =
                                attention_image.size6().unwrap();
                            let attention_image = attention_image
                                .constant_pad_nd(&[0, 0, 0, 0, 0, 1, 0, 0])
                                .constant_pad_nd(&[0, 0, 0, 0, 0, 0, 0, 1])
                                .permute(&[0, 1, 4, 2, 5, 3])
                                .reshape(&[
                                    bsize,
                                    1,
                                    image_h * (attention_h + 1),
                                    image_w * (attention_w + 1),
                                ]);

                            // event_writer
                            //     .write_image_list_async(
                            //         format!("attention_image/seq_{}", seq_index),
                            //         step,
                            //         attention_image,
                            //     )
                            //     .await?;
                            attention_image
                        })
                        .collect();

                    let seq =
                        save_image_seq_async("attention_image", sub_image_dir.clone(), seq).await?;

                    save_image_seq_to_tfrecord(&mut event_writer, "attention_image", step, seq)
                        .await?;
                }
            }
            msg::LogMessage::Image { step, sequence } => {
                let step = step as i64;

                for (seq_index, log_step) in sequence.into_iter().enumerate() {
                    let msg::ImageLog {
                        true_image,
                        fake_image,
                    } = log_step;

                    event_writer
                        .write_image_list_async(
                            format!("true_image/{}", seq_index),
                            step,
                            true_image,
                        )
                        .await?;

                    if let Some(fake_image) = fake_image {
                        event_writer
                            .write_image_list_async(
                                format!("fake_image/{}", seq_index),
                                step,
                                fake_image,
                            )
                            .await?;
                    }
                }
            }
        }
    }

    Ok(())
}

fn save_image_seq(
    name: &str,
    base_dir: impl AsRef<Path>,
    seq: &[impl Borrow<Tensor>],
) -> Result<()> {
    let base_dir = base_dir.as_ref();
    let batch_size = seq[0].borrow().size()[0];

    for (seq_index, batch) in seq.iter().enumerate() {
        let batch = batch.borrow();

        for batch_index in 0..batch_size {
            let image = batch.select(0, batch_index).mul(255.0).to_kind(Kind::Uint8);

            let dir = base_dir
                .join(format!("batch_{:03}", batch_index))
                .join(name);
            let path = dir.join(format!("seq_{:03}.jpg", seq_index));
            fs::create_dir_all(&dir)?;

            vision::image::save(&image, path)?;
        }
    }

    Ok(())
}

async fn save_image_seq_to_tfrecord<W>(
    event_writer: &mut EventWriter<W>,
    name: &'static str,
    step: i64,
    seq: Vec<Tensor>,
) -> Result<()>
where
    W: Unpin + futures::AsyncWriteExt,
{
    let batch_size = seq[0].size()[0];

    for (seq_index, image) in seq.into_iter().enumerate() {
        for batch_index in 0..batch_size {
            let image = image.select(0, batch_index);

            let name = format!("{}/batch_{:04}/seq_{:04}", name, batch_index, seq_index);

            let result = event_writer.write_image_async(&name, step, image).await;

            if let Err(err) = result {
                warn!(
                    "unable to write to TensorBoard, name = '{}': {:?}",
                    name, err
                )
            }
        }
    }

    Ok(())
}

async fn save_scalar_seq_to_tfrecord<W>(
    event_writer: &mut EventWriter<W>,
    name: &'static str,
    step: i64,
    seq: Vec<Tensor>,
) -> Result<()>
where
    W: Unpin + futures::AsyncWriteExt,
{
    let batch_size = seq[0].size()[0];

    for (seq_index, image) in seq.into_iter().enumerate() {
        for batch_index in 0..batch_size {
            let scalar = image.select(0, batch_index);

            let name = format!("{}/batch_{:04}/seq_{:04}", name, batch_index, seq_index);

            let result = event_writer
                .write_scalar_async(&name, step, f32::from(scalar))
                .await;

            if let Err(err) = result {
                warn!(
                    "unable to write to TensorBoard, name = '{}': {:?}",
                    name, err
                )
            }
        }
    }

    Ok(())
}

fn save_scalar_seq(
    name: &str,
    base_dir: impl AsRef<Path>,
    seq: &[impl Borrow<Tensor>],
) -> Result<()> {
    let base_dir = base_dir.as_ref();
    let batch_size = seq[0].borrow().size()[0];

    for batch_index in 0..batch_size {
        let dir = base_dir.join(format!("batch_{:03}", batch_index));
        let file = dir.join(format!("{}.txt", name));

        fs::create_dir_all(&dir)?;
        let mut writer = fs::File::create(file)?;

        for batch in seq {
            let batch = batch.borrow();
            let scalar = batch.select(0, batch_index);
            writeln!(writer, "{:+e}", f64::from(scalar))?;
        }
    }

    Ok(())
}
