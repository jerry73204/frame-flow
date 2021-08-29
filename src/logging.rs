use crate::{common::*, message as msg};

pub async fn logging_worker(
    log_dir: impl AsRef<Path>,
    mut log_rx: mpsc::Receiver<msg::LogMessage>,
) -> Result<()> {
    let event_dir = log_dir.as_ref().join("events");
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
                triangular_identity_loss,
                forward_consistency_loss,
                backward_consistency_gen_loss,
                backward_consistency_disc_loss,

                detector_weights,
                generator_weights,
                discriminator_weights,
                transformer_weights,
                image_seq_discriminator_weights,

                ground_truth_image_seq,
                generated_image_seq,
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

                if let Some(seq) = ground_truth_image_seq {
                    for (seq_index, image) in seq.into_iter().enumerate() {
                        event_writer
                            .write_image_list_async(
                                format!("ground_truth_image/seq_{}", seq_index),
                                step,
                                image,
                            )
                            .await?;
                    }
                }

                if let Some(seq) = generated_image_seq {
                    for (seq_index, image) in seq.into_iter().enumerate() {
                        event_writer
                            .write_image_list_async(
                                format!("generated_image/seq_{}", seq_index),
                                step,
                                image,
                            )
                            .await?;
                    }
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
