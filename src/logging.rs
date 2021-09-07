use crate::{common::*, message as msg, model::DetectionSimilarity, FILE_STRFTIME};

pub async fn logging_worker(
    log_dir: impl AsRef<Path>,
    mut log_rx: mpsc::Receiver<msg::LogMessage>,
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
                forward_consistency_similarity,
                backward_consistency_gen_loss,
                backward_consistency_disc_loss,

                detector_weights,
                generator_weights,
                discriminator_weights,
                transformer_weights,
                image_seq_discriminator_weights,

                ground_truth_image_seq,
                generator_generated_image_seq,
                transformer_generated_image_seq,
                transformer_generated_det_seq,
                transformer_attention_image_seq,
            }) => {
                let step = step as i64;

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

                    event_writer
                        .write_scalar_async(
                            "retraction_identity_similarity/position_loss",
                            step,
                            f32::from(position_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "retraction_identity_similarity/size_loss",
                            step,
                            f32::from(size_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "retraction_identity_similarity/obj_loss",
                            step,
                            f32::from(obj_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "retraction_identity_similarity/class_loss",
                            step,
                            f32::from(class_loss),
                        )
                        .await?;
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

                    event_writer
                        .write_scalar_async(
                            "triangular_identity_similarity/position_loss",
                            step,
                            f32::from(position_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "triangular_identity_similarity/size_loss",
                            step,
                            f32::from(size_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "triangular_identity_similarity/obj_loss",
                            step,
                            f32::from(obj_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "triangular_identity_similarity/class_loss",
                            step,
                            f32::from(class_loss),
                        )
                        .await?;
                }

                if let Some(similarity) = forward_consistency_similarity {
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

                    event_writer
                        .write_scalar_async(
                            "forward_consistency_similarity/position_loss",
                            step,
                            f32::from(position_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "forward_consistency_similarity/size_loss",
                            step,
                            f32::from(size_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "forward_consistency_similarity/obj_loss",
                            step,
                            f32::from(obj_loss),
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            "forward_consistency_similarity/class_loss",
                            step,
                            f32::from(class_loss),
                        )
                        .await?;
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

                let sub_image_dir = {
                    let dir = image_dir.join(format!(
                        "{:08}_{}",
                        step,
                        Local::now().format(FILE_STRFTIME)
                    ));
                    Arc::new(dir)
                };

                let save_image_seq_async =
                    |name: &'static str, dir: Arc<PathBuf>, seq: Vec<Tensor>| async move {
                        let seq = tokio::task::spawn_blocking(move || -> Result<_> {
                            save_image_seq(name, &**dir, &seq)?;
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

                if let Some(seq) = generator_generated_image_seq {
                    let seq = save_image_seq_async(
                        "generator_generated_image",
                        sub_image_dir.clone(),
                        seq,
                    )
                    .await?;
                    save_image_seq_to_tfrecord(
                        &mut event_writer,
                        "generator_generated_image",
                        step,
                        seq,
                    )
                    .await?;
                }

                if let Some(seq) = transformer_generated_image_seq {
                    let seq = save_image_seq_async(
                        "transformer_generated_image",
                        sub_image_dir.clone(),
                        seq,
                    )
                    .await?;

                    save_image_seq_to_tfrecord(
                        &mut event_writer,
                        "transformer_generated_image",
                        step,
                        seq,
                    )
                    .await?;
                }

                if let Some(seq) = transformer_generated_det_seq {
                    let objectness_seq: Vec<_> = seq
                        .into_iter()
                        .map(|det| {
                            assert!(det.tensors.len() == 1);
                            let (max, _argmax) = det.tensors[0].obj_prob().max_dim(2, false);
                            max
                        })
                        .collect();

                    let objectness_seq = save_image_seq_async(
                        "transformer_generated_objectness",
                        sub_image_dir.clone(),
                        objectness_seq,
                    )
                    .await?;

                    save_image_seq_to_tfrecord(
                        &mut event_writer,
                        "transformer_detection_objectness",
                        step,
                        objectness_seq,
                    )
                    .await?;
                }

                if let Some(seq) = transformer_attention_image_seq {
                    let seq: Vec<_> = seq
                        .into_iter()
                        .map(|image| {
                            let (bsize, _one, field_h, field_w, image_h, image_w) =
                                image.size6().unwrap();

                            // add border
                            let image = image
                                .constant_pad_nd(&[0, 0, 0, 0, 0, 1])
                                .constant_pad_nd(&[0, 0, 0, 0, 0, 0, 0, 1])
                                .expand(
                                    &[bsize, 3, field_h + 1, field_w + 1, image_h, image_w],
                                    false,
                                );

                            // fill border with green color
                            // let _ = image
                            //     .i((.., 1..2, field_h..(field_h + 1), .., .., ..))
                            //     .fill_(0.1);
                            // let _ = image
                            //     .i((.., 1..2, .., field_w..(field_w + 1), .., ..))
                            //     .fill_(0.1);

                            image.permute(&[0, 1, 4, 2, 5, 3]).reshape(&[
                                bsize,
                                3,
                                (field_h + 1) * image_h,
                                (field_w + 1) * image_w,
                            ])
                        })
                        .collect();

                    let seq =
                        save_image_seq_async("transformer_attention_image", sub_image_dir, seq)
                            .await?;

                    save_image_seq_to_tfrecord(
                        &mut event_writer,
                        "transformer_attention_image",
                        step,
                        seq,
                    )
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

            event_writer
                .write_image_async(
                    format!("{}/batch_{:04}/seq_{:04}", name, batch_index, seq_index),
                    step,
                    image,
                )
                .await?;
        }
    }

    Ok(())
}
