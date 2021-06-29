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
                        det_loss,
                        det_recon_loss,
                        discriminator_loss,
                        generator_loss,
                    } = log_step;

                    event_writer
                        .write_scalar_async(
                            format!("det_loss/{}", seq_index),
                            step,
                            det_loss as f32,
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            format!("det_recon_loss/{}", seq_index),
                            step,
                            det_recon_loss as f32,
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            format!("discriminator_loss/{}", seq_index),
                            step,
                            discriminator_loss as f32,
                        )
                        .await?;
                    event_writer
                        .write_scalar_async(
                            format!("generator_loss/{}", seq_index),
                            step,
                            generator_loss as f32,
                        )
                        .await?;
                    event_writer
                        .write_scalar_async("params/learning_rate", step, learning_rate as f32)
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

    Ok(())
}
