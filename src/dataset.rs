use crate::common::*;

#[derive(Debug, Clone)]
pub struct DatasetInit<P>
where
    P: AsRef<Path>,
{
    pub dir: P,
    pub file_name_digits: usize,
    pub height: usize,
    pub width: usize,
}

impl<P> DatasetInit<P>
where
    P: AsRef<Path>,
{
    pub fn load(self) -> Result<Dataset> {
        let Self {
            dir: parent_dir,
            file_name_digits,
            height,
            width,
        } = self;
        let parent_dir = parent_dir.as_ref();
        let dataset_file = parent_dir.join("dataset.csv");

        let entries: Vec<DatasetEntry> = csv::Reader::from_path(&dataset_file)?
            .deserialize()
            .try_collect()?;

        let orig_num_entries = entries.len();
        ensure!(orig_num_entries > 0, "empty dataset is not allowed");

        let entries: IndexMap<_, _> = entries
            .into_iter()
            .map(|entry| -> Result<_> {
                let DatasetEntry {
                    ref name, count, ..
                } = entry;
                let segment_dir = parent_dir.join(name);

                (1..=count).try_for_each(move |index| {
                    let file_name = format!("{:0width$}.png", index, width = file_name_digits);
                    let path = segment_dir.join(file_name);

                    ensure!(path.is_file(), "'{}' is not a file", path.display());
                    Ok(())
                })?;

                Ok((name.clone(), entry))
            })
            .try_collect()?;

        let min_length = entries.values().map(|entry| entry.count).min().unwrap();

        ensure!(
            entries.len() == orig_num_entries,
            "duplicated dataset name found"
        );

        Ok(Dataset {
            min_length,
            entries,
            dir: parent_dir.to_owned(),
            file_name_digits,
            height,
            width,
        })
    }
}

#[derive(Debug)]
pub struct Dataset {
    min_length: usize,
    entries: IndexMap<String, DatasetEntry>,
    dir: PathBuf,
    file_name_digits: usize,
    height: usize,
    width: usize,
}

impl Dataset {
    pub fn sample(&self, length: usize) -> Result<Vec<Tensor>> {
        let Self {
            min_length,
            file_name_digits,
            height,
            width,
            ref entries,
            ref dir,
        } = *self;

        ensure!(
            length <= min_length,
            "the sampling length {} is greater the minimum sequence length {}",
            length,
            min_length
        );

        let mut rng = rand::thread_rng();
        let entry_index = rng.gen_range(0..entries.len());
        let DatasetEntry {
            ref name, count, ..
        } = entries[entry_index];
        let segment_dir = dir.join(name);

        let start_index = rng.gen_range(0..=(count - length)) + 1;
        let end_index = start_index + length;

        let output: Vec<_> = (start_index..end_index)
            .map(|frame_index| -> Result<_> {
                let file_name = format!("{:0width$}.png", frame_index, width = file_name_digits);
                let path = segment_dir.join(file_name);
                let image =
                    vision::image::load(path)?.resize2d_letterbox(height as i64, width as i64)?;
                Ok(image)
            })
            .try_collect()?;

        Ok(output)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct DatasetEntry {
    pub name: String,
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_dataset_test() -> Result<()> {
        let hx = 256;
        let wx = 256;

        let dataset = DatasetInit {
            dir: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test")
                .join("demo-dataset"),
            file_name_digits: 8,
            height: hx,
            width: wx,
        }
        .load()?;

        (1..=5).try_for_each(|len| -> Result<_> {
            let samples = dataset.sample(len)?;

            ensure!(samples.len() == len);

            samples.into_iter().try_for_each(|sample| {
                ensure!(sample.size() == vec![3, hx as i64, wx as i64]);
                Ok(())
            })?;

            Ok(())
        })?;

        Ok(())
    }
}
