use crate::{
    common::*,
    dataset::{Dataset, DatasetInit},
};

#[derive(Debug, TensorLike)]
pub struct TrainingRecord {
    pub sequence: Vec<Tensor>,
    pub noise: Tensor,
    pub batch_index: usize,
    pub record_index: usize,
}

#[derive(Debug, Clone)]
pub struct TrainingStreamInit<P1, P2>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    pub dataset_dir: P1,
    pub cache_dir: P2,
    pub file_name_digits: usize,
    pub latent_dim: usize,
    pub batch_size: usize,
    pub device: Device,
    pub seq_len: usize,
    pub image_size: usize,
    pub image_dim: usize,
}

impl<P1, P2> TrainingStreamInit<P1, P2>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    pub async fn build(self) -> Result<TrainingStream> {
        let Self {
            dataset_dir,
            file_name_digits,
            latent_dim,
            batch_size,
            device,
            seq_len,
            cache_dir,
            image_size,
            image_dim,
        } = self;

        let dataset = DatasetInit {
            dir: dataset_dir,
            file_name_digits,
        }
        .load()
        .await?;

        Ok(TrainingStream {
            dataset: Arc::new(dataset),
            latent_dim,
            batch_size,
            device,
            seq_len,
            cache_dir: cache_dir.as_ref().to_owned(),
            image_size,
            image_dim,
        })
    }
}

#[derive(Debug)]
pub struct TrainingStream {
    dataset: Arc<Dataset>,
    latent_dim: usize,
    batch_size: usize,
    device: Device,
    seq_len: usize,
    cache_dir: PathBuf,
    image_size: usize,
    image_dim: usize,
}

impl TrainingStream {
    pub fn stream(&self) -> impl Stream<Item = Result<TrainingRecord>> {
        let Self {
            ref dataset,
            latent_dim,
            batch_size,
            device,
            seq_len,
            ref cache_dir,
            image_size,
            image_dim,
        } = *self;

        // load subsequence samples
        let stream = {
            let dataset = dataset.clone();

            stream::repeat(())
                .wrapping_enumerate()
                .map(|(record_index, _)| Result::Ok(record_index))
                .try_par_then_unordered(None, move |record_index| {
                    let dataset = dataset.clone();

                    async move {
                        let paths = dataset.sample(seq_len)?;
                        let images: Vec<_> = paths
                            .into_iter()
                            .map(|path| -> Result<_> {
                                let image = vision::image::load_and_resize(
                                    path,
                                    image_size as i64,
                                    image_size as i64,
                                )?
                                .to_device(device);
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
                let (max_record_index, sequence_vec): (MaxCollector<_>, Vec<_>) =
                    chunk.into_iter().unzip_n();
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
                    let noise = Tensor::rand(
                        &[batch_size as i64, latent_dim as i64],
                        (Kind::Float, device),
                    );
                    Ok((batch_index, record_index, sequence, noise))
                }
            });

        // convert to output type
        let stream =
            stream.try_par_map_unordered(None, |(batch_index, record_index, sequence, noise)| {
                move || {
                    let record = TrainingRecord {
                        batch_index,
                        record_index,
                        sequence,
                        noise,
                    };
                    Ok(record)
                }
            });

        stream
    }
}
