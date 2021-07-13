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
            msg::LogMessage::Loss {
                step,
                learning_rate,
                sequence,
            } => {
                let step = step as i64;

                for (seq_index, log_step) in sequence.into_iter().enumerate() {
                    let msg::LossLog {
                        real_det_loss,
                        fake_det_loss,
                        discriminator_loss,
                        generator_loss,
                        detector_grads,
                        discriminator_grads,
                        generator_grads,
                        detector_weights,
                        discriminator_weights,
                        generator_weights,
                    } = log_step;

                    if let Some(real_det_loss) = real_det_loss {
                        event_writer
                            .write_scalar_async(
                                format!("real_det_loss/{}", seq_index),
                                step,
                                real_det_loss as f32,
                            )
                            .await?;
                    }

                    if let Some(fake_det_loss) = fake_det_loss {
                        event_writer
                            .write_scalar_async(
                                format!("fake_det_loss/{}", seq_index),
                                step,
                                fake_det_loss as f32,
                            )
                            .await?;
                    }

                    if let Some(discriminator_loss) = discriminator_loss {
                        event_writer
                            .write_scalar_async(
                                format!("discriminator_loss/{}", seq_index),
                                step,
                                discriminator_loss as f32,
                            )
                            .await?;
                    }

                    if let Some(generator_loss) = generator_loss {
                        event_writer
                            .write_scalar_async(
                                format!("generator_loss/{}", seq_index),
                                step,
                                generator_loss as f32,
                            )
                            .await?;
                    }

                    event_writer
                        .write_scalar_async("params/learning_rate", step, learning_rate as f32)
                        .await?;

                    // log weights
                    if let Some(detector_weights) = detector_weights {
                        for (name, weight) in detector_weights {
                            event_writer
                                .write_scalar_async(
                                    format!("detector_weights/{}", name),
                                    step,
                                    weight as f32,
                                )
                                .await?;
                        }
                    }
                    if let Some(discriminator_weights) = discriminator_weights {
                        for (name, weight) in discriminator_weights {
                            event_writer
                                .write_scalar_async(
                                    format!("discriminator_weights/{}", name),
                                    step,
                                    weight as f32,
                                )
                                .await?;
                        }
                    }
                    if let Some(generator_weights) = generator_weights {
                        for (name, weight) in generator_weights {
                            event_writer
                                .write_scalar_async(
                                    format!("generator_weights/{}", name),
                                    step,
                                    weight as f32,
                                )
                                .await?;
                        }
                    }

                    // log gradients
                    if let Some(detector_grads) = detector_grads {
                        for (name, grad) in detector_grads {
                            event_writer
                                .write_scalar_async(
                                    format!("detector_gradients/{}", name),
                                    step,
                                    grad as f32,
                                )
                                .await?;
                        }
                    }
                    if let Some(discriminator_grads) = discriminator_grads {
                        for (name, grad) in discriminator_grads {
                            event_writer
                                .write_scalar_async(
                                    format!("discriminator_gradients/{}", name),
                                    step,
                                    grad as f32,
                                )
                                .await?;
                        }
                    }
                    if let Some(generator_grads) = generator_grads {
                        for (name, grad) in generator_grads {
                            event_writer
                                .write_scalar_async(
                                    format!("generator_gradients/{}", name),
                                    step,
                                    grad as f32,
                                )
                                .await?;
                        }
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
