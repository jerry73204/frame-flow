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

        let msg::LogMessage {
            step,
            dis_loss,
            gen_loss,
            learning_rate,
            true_image,
            fake_image,
        } = msg;
        let step = step as i64;

        event_writer
            .write_scalar_async("loss/discriminator_loss", step, dis_loss as f32)
            .await?;
        event_writer
            .write_scalar_async("loss/generator_loss", step, gen_loss as f32)
            .await?;
        event_writer
            .write_scalar_async("params/learning_rate", step, learning_rate as f32)
            .await?;

        if let Some(true_image) = true_image {
            for (index, image) in true_image.into_iter().enumerate() {
                event_writer
                    .write_image_list_async(format!("image/true_image/{}", index), step, image)
                    .await?;
            }
        }

        if let Some(fake_image) = fake_image {
            for (index, image) in fake_image.into_iter().enumerate() {
                event_writer
                    .write_image_list_async(format!("image/fake_image/{}", index), step, image)
                    .await?;
            }
        }
    }

    Ok(())
}
