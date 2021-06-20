use crate::{
    common::*,
    config,
    dataset::{Dataset, IiiDataset, SimpleDataset},
    message as msg,
};

pub async fn training_stream(
    dataset_cfg: &config::Dataset,
    train_cfg: &config::Training,
) -> Result<impl Stream<Item = Result<msg::TrainingMessage>>> {
    let config::Training {
        ref cache_dir,
        latent_dim,
        batch_size,
        device,
        image_size,
        peek_len,
        pred_len,
        ..
    } = *train_cfg;
    let seq_len = peek_len.get() + pred_len.get();
    let image_size = image_size.get() as i64;
    let image_dim = dataset_cfg.image_dim() as i64;
    let batch_size = batch_size.get();
    let latent_dim = latent_dim.get() as i64;

    let dataset: Dataset = match *dataset_cfg {
        config::Dataset::Iii(config::IiiDataset {
            ref dataset_dir,
            ref classes_file,
            ref class_whitelist,
            ref blacklist_files,
            min_seq_len,
            ..
        }) => IiiDataset::load(
            dataset_dir,
            classes_file,
            class_whitelist.clone(),
            min_seq_len,
            blacklist_files.clone().unwrap_or_else(HashSet::new),
        )
        .await?
        .into(),
        config::Dataset::Simple(config::SimpleDataset {
            ref dataset_dir,
            file_name_digits,
            ..
        }) => SimpleDataset::load(dataset_dir, file_name_digits.get())
            .await?
            .into(),
    };
    let dataset = Arc::new(dataset);

    // load subsequence samples
    let stream = {
        stream::repeat(())
            .wrapping_enumerate()
            .map(|(record_index, _)| Result::Ok(record_index))
            .try_par_then_unordered(None, move |record_index| {
                let dataset = dataset.clone();

                async move {
                    let samples = dataset.sample(seq_len)?;
                    let images: Vec<_> = samples
                        .iter()
                        .map(|sample| -> Result<_> {
                            let image = vision::image::load_and_resize(
                                &sample.image_file,
                                image_size,
                                image_size,
                            )?
                            .to_device(device);
                            let image = image / 255.0;
                            let image = image.to_kind(Kind::Float).set_requires_grad(false);
                            Ok(image)
                        })
                        .try_collect()?;

                    Fallible::Ok((record_index, images))
                }
            })
    };

    // group into chunks
    let stream = stream
        .chunks(batch_size)
        .wrapping_enumerate()
        .par_map_unordered(None, |(index, results)| {
            move || {
                let chunk: Vec<_> = results.into_iter().try_collect()?;
                Fallible::Ok((index, chunk))
            }
        });

    // convert to batched type
    let stream = stream.try_par_map_unordered(None, move |(batch_index, chunk)| {
        move || {
            let (max_record_index, sequence_vec): (MaxVal<_>, Vec<_>) = chunk.into_iter().unzip_n();
            let record_index = max_record_index.unwrap();

            let mut sequence_iter_vec: Vec<_> = sequence_vec
                .into_iter()
                .map(|seq| seq.into_iter())
                .collect();

            let sequence: Vec<_> = (0..seq_len)
                .map(|_seq_index| {
                    let samples_vec: Vec<_> = (0..batch_size)
                        .map(|batch_index| sequence_iter_vec[batch_index].next().unwrap())
                        .collect();

                    let samples = Tensor::stack(&samples_vec, 0);
                    samples
                })
                .collect();

            Fallible::Ok((batch_index, record_index, sequence))
        }
    });

    // generate noise for each batch
    let stream =
        stream.try_par_map_unordered(None, move |(batch_index, record_index, sequence)| {
            move || {
                let noise = Tensor::rand(&[batch_size as i64, latent_dim], (Kind::Float, device));
                Ok((batch_index, record_index, sequence, noise))
            }
        });

    // convert to output type
    let stream =
        stream.try_par_map_unordered(None, |(batch_index, record_index, sequence, noise)| {
            move || {
                let record = msg::TrainingMessage {
                    batch_index,
                    record_index,
                    sequence,
                    noise,
                };
                Ok(record)
            }
        });

    Ok(stream)
}
