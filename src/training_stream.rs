use bbox::{Transform, HW};
use tch_goodies::{Pixel, Ratio};

use crate::{common::*, config, dataset::Dataset, message as msg};

pub async fn training_stream(
    dataset: Dataset,
    train_cfg: &config::Training,
) -> Result<impl Stream<Item = Result<msg::TrainingMessage>>> {
    let config::Training {
        // ref cache_dir,
        latent_dim,
        batch_size,
        // device,
        image_size,
        seq_len,
        ..
    } = *train_cfg;
    let seq_len = seq_len.get();
    let image_size = image_size.get() as i64;
    let batch_size = batch_size.get();
    let latent_dim = latent_dim.get() as i64;

    let dataset = Arc::new(dataset);

    // load sequence samples
    let stream = {
        stream::repeat(()).par_then_unordered(None, move |()| {
            let dataset = dataset.clone();

            async move {
                let samples = dataset.sample(seq_len)?;
                let pairs: Vec<_> = samples
                    .iter()
                    .map(|sample| -> Result<_> {
                        let image = sample.image()?;

                        let orig_pixel_size = {
                            let (_, h, w) = image.size3().unwrap();
                            HW::from_hw([h, w]).cast::<R64>()
                        };
                        let new_pixel_size = HW::from_hw([image_size, image_size]).cast::<R64>();
                        let pixel_transform = Transform::from_sizes_letterbox(
                            orig_pixel_size,
                            new_pixel_size.clone(),
                        );
                        let ratio_transform =
                            Transform::from_sizes_exact(new_pixel_size, HW::unit());
                        let transform = &ratio_transform * &pixel_transform;

                        // resize and scale image
                        let image = image
                            .resize2d_letterbox(image_size, image_size)?
                            .mul(2.0)
                            .sub(1.0)
                            .set_requires_grad(false);

                        // transform boxes
                        let boxes: Vec<_> = sample
                            .boxes()
                            .iter()
                            .map(|label| {
                                Ratio(RectLabel {
                                    rect: &transform * &label.rect,
                                    class: label.class,
                                })
                            })
                            .collect();

                        Ok((image, boxes))
                    })
                    .try_collect()?;
                let (image_seq, boxes_seq) = pairs.into_iter().unzip_n_vec();

                anyhow::Ok((image_seq, boxes_seq))
            }
        })
    };

    // group into chunks
    let stream =
        stream
            .chunks(batch_size)
            .enumerate()
            .par_map_unordered(None, |(batch_index, results)| {
                move || {
                    let chunk: Vec<_> = results.into_iter().try_collect()?;
                    anyhow::Ok((batch_index, chunk))
                }
            });

    // convert to batched type
    let stream = stream.try_par_map_unordered(None, move |(batch_index, chunk)| {
        move || {
            let (
                image_seq_batch, // sample index -> sequence index -> image
                boxes_seq_batch, // sample index -> sequence index -> boxes set
            ) = chunk.into_iter().unzip_n_vec();

            // transpose to
            // sequence index -> sample index -> X
            let image_batch_seq: Vec<Tensor> = image_seq_batch
                .transpose()
                .unwrap()
                .into_iter()
                .map(|image_batch| Tensor::stack(&image_batch, 0))
                .collect();
            let boxes_batch_seq: Vec<Vec<Vec<Ratio<RectLabel>>>> =
                boxes_seq_batch.transpose().unwrap();

            let seq_len = image_batch_seq.len();
            let noise_seq: Vec<Tensor> = (0..seq_len)
                .map(|_| Tensor::randn(&[batch_size as i64, latent_dim], FLOAT_CPU))
                .collect();

            anyhow::Ok((batch_index, image_batch_seq, boxes_batch_seq, noise_seq))
        }
    });

    // convert to output type
    let stream = stream.try_par_map_unordered(
        None,
        |(batch_index, image_batch_seq, boxes_batch_seq, noise_seq)| {
            move || {
                let record = msg::TrainingMessage {
                    batch_index,
                    image_batch_seq,
                    boxes_batch_seq,
                    noise_seq,
                };
                Ok(record)
            }
        },
    );

    Ok(stream)
}
