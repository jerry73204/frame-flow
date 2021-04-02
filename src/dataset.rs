use crate::common::*;

#[derive(Debug, Clone)]
pub struct DatasetInit<P>
where
    P: AsRef<Path>,
{
    pub dir: P,
    pub file_name_digits: usize,
}

impl<P> DatasetInit<P>
where
    P: AsRef<Path>,
{
    pub async fn load(self) -> Result<Dataset> {
        let Self {
            dir: parent_dir,
            file_name_digits,
        } = self;
        let parent_dir = Arc::new(parent_dir.as_ref().to_owned());
        let dataset_file = parent_dir.join("dataset.csv");

        let entries: Vec<DatasetEntry> = csv::Reader::from_path(&dataset_file)?
            .deserialize()
            .try_collect()?;

        let orig_num_entries = entries.len();
        ensure!(orig_num_entries > 0, "empty dataset is not allowed");
        info!("{} sequence entries", orig_num_entries);

        let entries: IndexMap<_, _> = entries
            .into_iter()
            .map(|entry| (entry.name.clone(), Arc::new(entry)))
            .collect();

        ensure!(
            entries.len() == orig_num_entries,
            "duplicated dataset name found"
        );

        {
            let parent_dir = parent_dir.clone();
            let entries_vec: Vec<_> = entries.values().cloned().collect();

            stream::iter(entries_vec)
                .flat_map(move |entry| {
                    let segment_dir = Arc::new(parent_dir.join(&entry.name));

                    stream::iter(1..=entry.count).map(move |index| Ok((segment_dir.clone(), index)))
                })
                .try_par_for_each(None, move |(segment_dir, index)| async move {
                    let file_name = format!("{:0width$}.png", index, width = file_name_digits);
                    let path = segment_dir.join(file_name);
                    ensure!(path.is_file(), "'{}' is not a file", path.display());
                    Ok(())
                })
                .await?;
        }

        let min_length = entries.values().map(|entry| entry.count).min().unwrap();

        let entries: IndexMap<_, _> = entries
            .into_iter()
            .map(|(name, entry)| (name, Arc::try_unwrap(entry).unwrap()))
            .collect();

        Ok(Dataset {
            min_length,
            entries,
            dir: (*parent_dir).clone(),
            file_name_digits,
        })
    }
}

#[derive(Debug)]
pub struct Dataset {
    min_length: usize,
    entries: IndexMap<String, DatasetEntry>,
    dir: PathBuf,
    file_name_digits: usize,
}

impl Dataset {
    pub fn sample(&self, length: usize) -> Result<Vec<PathBuf>> {
        let Self {
            min_length,
            file_name_digits,
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
                Ok(path)
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

    #[tokio::test]
    async fn simple_dataset_test() -> Result<()> {
        let hx = 256;
        let wx = 256;

        let dataset = DatasetInit {
            dir: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test")
                .join("demo-dataset"),
            file_name_digits: 8,
        }
        .load()
        .await?;

        (1..=5).try_for_each(|len| -> Result<_> {
            let samples = dataset.sample(len)?;
            ensure!(samples.len() == len);

            Ok(())
        })?;

        Ok(())
    }
}
