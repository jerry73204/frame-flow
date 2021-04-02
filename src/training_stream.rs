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
pub struct TrainingStreamInit<P>
where
    P: AsRef<Path>,
{
    pub dir: P,
    pub file_name_digits: usize,
    pub height: usize,
    pub width: usize,
    pub latent_dim: usize,
    pub batch_size: usize,
    pub device: Device,
    pub seq_len: usize,
}

impl<P> TrainingStreamInit<P>
where
    P: AsRef<Path>,
{
    pub fn build(self) -> Result<TrainingStream> {
        let Self {
            dir,
            file_name_digits,
            height,
            width,
            latent_dim,
            batch_size,
            device,
            seq_len,
        } = self;

        let dataset = DatasetInit {
            dir,
            file_name_digits,
            height,
            width,
        }
        .load()?;

        Ok(TrainingStream {
            dataset: Arc::new(dataset),
            latent_dim,
            batch_size,
            device,
            seq_len,
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
}

impl TrainingStream {
    pub fn stream(&self) -> impl Stream<Item = Result<TrainingRecord>> {
        let Self {
            ref dataset,
            latent_dim,
            batch_size,
            device,
            seq_len,
        } = *self;

        // load subsequence samples
        let stream = stream::repeat(())
            .wrapping_enumerate()
            .map(|(record_index, _)| Result::Ok(record_index))
            .try_par_map_init_unordered(
                None,
                || dataset.clone(),
                move |dataset, record_index| {
                    let dataset = dataset.clone();
                    move || -> Result<_> {
                        let sample = dataset.sample(seq_len)?.to_device(device);
                        Ok((record_index, sample))
                    }
                },
            );

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
