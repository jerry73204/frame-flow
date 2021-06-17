use crate::common::*;

pub use dataset::Dataset;
pub use iii_dataset::*;
pub use simple_dataset::*;

mod dataset {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Sample {
        pub image_file: PathBuf,
        pub size: PixelSize<usize>,
        pub boxes: Vec<RatioLabel>,
    }

    #[derive(Debug)]
    pub enum Dataset {
        Simple(simple_dataset::SimpleDataset),
        Iii(iii_dataset::IiiDataset),
    }

    impl Dataset {
        pub fn sample(&self, length: usize) -> Result<&[Sample]> {
            match self {
                Self::Simple(dataset) => dataset.sample(length),
                Self::Iii(dataset) => dataset.sample(length),
            }
        }
    }

    impl From<simple_dataset::SimpleDataset> for Dataset {
        fn from(from: simple_dataset::SimpleDataset) -> Self {
            Self::Simple(from)
        }
    }

    impl From<iii_dataset::IiiDataset> for Dataset {
        fn from(from: iii_dataset::IiiDataset) -> Self {
            Self::Iii(from)
        }
    }
}

mod iii_dataset {
    use super::{dataset::Sample, *};
    use iii_formosa_dataset as iii;

    const III_IMAGE_CHANNELS: usize = 3;

    #[derive(Debug, Clone)]
    pub struct IiiDataset {
        min_series_len: usize,
        series: Vec<TimeSeries>,
        weights: Vec<usize>,
        classes: IndexSet<String>,
    }

    #[derive(Debug, Clone)]
    struct TimeSeries {
        dir: PathBuf,
        samples: Vec<Sample>,
    }

    impl IiiDataset {
        pub async fn load(
            dataset_dir: impl AsRef<Path>,
            classes_file: impl AsRef<Path>,
            class_whitelist: Option<HashSet<String>>,
            blacklist_files: HashSet<PathBuf>,
        ) -> Result<Self> {
            let dataset_dir = dataset_dir.as_ref();
            let classes_file = classes_file.as_ref();

            // load classes file
            let classes = super::load_classes_file(&classes_file).await?;
            let classes = Arc::new(classes);
            let class_whitelist = Arc::new(class_whitelist);

            // list xml files
            let xml_files = {
                let dataset_dir = dataset_dir.to_owned();
                tokio::task::spawn_blocking(move || {
                    list_iii_xml_files(dataset_dir, blacklist_files)
                })
                .await??
            };

            // parse xml files
            let samples: GroupHashMap<_, _> = {
                let classes = classes.clone();

                let iter = xml_files.into_iter().flat_map(|(dir, files)| {
                    let dir = Arc::new(dir);
                    files.into_iter().map(move |file| (dir.clone(), file))
                });

                stream::iter(iter)
                    .par_map_unordered(None, move |(dir, annotation_file)| {
                        let classes = classes.clone();
                        let class_whitelist = class_whitelist.clone();

                        move || {
                            let xml_content =
                                fs::read_to_string(&*annotation_file).with_context(|| {
                                    format!(
                                        "failed to read annotation file {}",
                                        annotation_file.display()
                                    )
                                })?;
                            let annotation: iii::Annotation = serde_xml_rs::from_str(&xml_content)
                                .with_context(|| {
                                    format!(
                                        "failed to parse annotation file {}",
                                        annotation_file.display()
                                    )
                                })?;
                            let image_file = {
                                let file_name = format!(
                                    "{}.jpg",
                                    annotation_file.file_stem().unwrap().to_str().unwrap()
                                );
                                let image_file = annotation_file.parent().unwrap().join(file_name);
                                image_file
                            };

                            // check # of channel
                            ensure!(
                                annotation.size.depth == III_IMAGE_CHANNELS,
                                "expect depth to be {}, but found {}",
                                III_IMAGE_CHANNELS,
                                annotation.size.depth
                            );

                            let size = {
                                let iii::Size { width, height, .. } = annotation.size;
                                PixelSize::new(height, width).unwrap()
                            };

                            let boxes: Vec<_> = annotation
                                .object
                                .iter()
                                .filter_map(|obj| {
                                    // filter by class list and whitelist
                                    let class_name = &obj.name;
                                    let class_index = classes.get_index_of(class_name)?;
                                    if let Some(whitelist) = &*class_whitelist {
                                        whitelist.get(class_name)?;
                                    }
                                    Some((obj, class_index))
                                })
                                .map(|(obj, class_index)| -> Result<_> {
                                    let iii::BndBox {
                                        xmin: l_pixel,
                                        ymin: t_pixel,
                                        xmax: r_pixel,
                                        ymax: b_pixel,
                                    } = obj.bndbox;

                                    let t_ratio = t_pixel / size.h() as f64;
                                    let b_ratio = b_pixel / size.h() as f64;
                                    let l_ratio = l_pixel / size.w() as f64;
                                    let r_ratio = r_pixel / size.w() as f64;
                                    let bbox =
                                        RatioCyCxHW::from_tlbr(t_ratio, l_ratio, b_ratio, r_ratio)?;

                                    let labeled_bbox = RatioLabel {
                                        cycxhw: bbox,
                                        class: class_index,
                                    };
                                    Ok(labeled_bbox)
                                })
                                .try_collect()?;

                            let sample = Sample {
                                image_file,
                                size,
                                boxes,
                            };

                            Fallible::Ok((dir, sample))
                        }
                    })
                    .try_collect()
                    .await?
            };

            let series: Vec<_> = samples
                .into_inner()
                .into_iter()
                .map(|(dir, samples)| TimeSeries {
                    dir: Arc::try_unwrap(dir).unwrap(),
                    samples,
                })
                .collect();

            let classes = Arc::try_unwrap(classes).unwrap();
            let weights: Vec<_> = series.iter().map(|series| series.samples.len()).collect();
            let min_series_len = series
                .iter()
                .map(|series| series.samples.len())
                .min()
                .unwrap();

            Ok(Self {
                series,
                classes,
                weights,
                min_series_len,
            })
        }

        pub fn sample(&self, length: usize) -> Result<&[Sample]> {
            ensure!(length > 0, "zero length is not allowed");
            ensure!(
                self.min_series_len >= length,
                "length {} is too large, it exceeds min_series_len {}",
                length,
                self.min_series_len
            );

            // select series set
            let mut rng = rand::thread_rng();
            let dist = dists::WeightedIndex::new(&self.weights).unwrap();
            let series_index = dist.sample(&mut rng);
            let series = &self.series[series_index];

            // sample images
            let end = series.samples.len() - length;
            let start_index = rng.gen_range(0..=end);
            let end_index = start_index + length;
            let samples = &series.samples[start_index..end_index];

            Ok(samples)
        }
    }

    fn list_iii_xml_files(
        dataset_dir: impl AsRef<Path>,
        blacklist_files: HashSet<PathBuf>,
    ) -> Result<HashMap<PathBuf, Vec<PathBuf>>> {
        let dataset_dir = dataset_dir.as_ref();

        let xml_files: HashMap<_, Vec<_>> = glob::glob(&format!("{}/*/*", dataset_dir.display()))?
            .map_err(Error::from)
            .try_filter(|path| Ok(path.is_dir()))
            .and_then(|dir| {
                let xml_files: Vec<_> = glob::glob(&format!("{}/*.xml", dir.display()))?
                    .try_filter(|file| {
                        let suffix = file.strip_prefix(dataset_dir).unwrap();
                        let ok = !blacklist_files.contains(suffix);
                        if !ok {
                            warn!("ignore blacklisted file '{}'", file.display());
                        }
                        Ok(ok)
                    })
                    .try_collect()?;

                Ok((dir, xml_files))
            })
            .try_collect()?;

        Ok(xml_files)
    }

    // #[cfg(test)]
    // mod tests {
    //     use super::*;

    //     #[test]
    //     fn time_series_dataset() -> Result<()> {
    //         Ok(())
    //     }
    // }
}

mod simple_dataset {
    use super::{dataset::Sample, *};

    #[derive(Debug)]
    pub struct SimpleDataset {
        min_length: usize,
        series: IndexMap<String, TimeSeries>,
        weights: Vec<usize>,
    }

    #[derive(Debug)]
    struct TimeSeries {
        name: String,
        samples: Vec<Sample>,
    }

    impl SimpleDataset {
        pub async fn load(dir: impl AsRef<Path>, file_name_digits: usize) -> Result<Self> {
            let parent_dir = Arc::new(dir.as_ref().to_owned());
            let dataset_file = parent_dir.join("dataset.csv");
            let label_file = parent_dir.join("label.csv");
            let classes_file = parent_dir.join("class.txt");

            let classes = super::load_classes_file(&classes_file).await?;

            // load dataset entries
            let series_entries = {
                let entries: Vec<SeriesEntry> = csv::Reader::from_path(&dataset_file)?
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

                entries
            };

            let mut label_entries = {
                let entries: GroupHashMap<_, _> = csv::Reader::from_path(&label_file)?
                    .deserialize()
                    .map_err(Error::from)
                    .and_then(|entry: LabelEntry| {
                        let LabelEntry {
                            series: ref series_name,
                            index,
                            ..
                        } = entry;

                        let series = series_entries.get(series_name).ok_or_else(|| {
                            format_err!(
                                "the series name '{}' is not found in '{}'",
                                series_name,
                                dataset_file.display()
                            )
                        })?;
                        ensure!(
                            index < series.count,
                            "the index {} exceeds the number of images {} for series '{}'",
                            index,
                            series.count,
                            series_name
                        );

                        Ok((series_name.to_owned(), entry))
                    })
                    .try_collect()?;
                entries.into_inner()
            };

            let min_length = series_entries
                .values()
                .map(|entry| entry.count)
                .min()
                .unwrap();

            let series: IndexMap<_, _> = series_entries
                .into_iter()
                .map(|(series_name, series_entry)| -> Result<_> {
                    let image_size = PixelSize::new(series_entry.height, series_entry.width)?;

                    let mut labels = {
                        let labels: GroupHashMap<_, _> = label_entries
                            .remove(&series_name)
                            .into_iter()
                            .flatten()
                            .map(|label_entry: LabelEntry| -> Result<_> {
                                let LabelEntry {
                                    index,
                                    top: t_pixel,
                                    left: l_pixel,
                                    height: h_pixel,
                                    width: w_pixel,
                                    class: class_name,
                                    ..
                                } = label_entry;
                                let cy_pixel = t_pixel + h_pixel / 2.0;
                                let cx_pixel = l_pixel + w_pixel / 2.0;
                                let class = classes.get_index_of(&class_name).ok_or_else(|| {
                                    format_err!(
                                        "the class name '{}' is not foudn in class file '{}'",
                                        class_name,
                                        classes_file.display()
                                    )
                                })?;

                                let cy_ratio = cy_pixel / image_size.h() as f64;
                                let cx_ratio = cx_pixel / image_size.w() as f64;
                                let h_ratio = h_pixel / image_size.h() as f64;
                                let w_ratio = w_pixel / image_size.w() as f64;

                                let label = RatioLabel {
                                    cycxhw: RatioCyCxHW::from_cycxhw(
                                        cy_ratio, cx_ratio, h_ratio, w_ratio,
                                    )?
                                    .cast()
                                    .unwrap(),
                                    class,
                                };

                                Ok((index, label))
                            })
                            .try_collect()?;

                        labels.into_inner()
                    };

                    let samples: Vec<_> = (0..series_entry.count)
                        .map(|sample_index| {
                            let image_file = parent_dir.join(&series_name).join(format!(
                                "{:0width$}.png",
                                sample_index,
                                width = file_name_digits
                            ));
                            ensure!(
                                image_file.is_file(),
                                "the image file '{}' does not exist or is not a file",
                                image_file.display()
                            );
                            let boxes = labels.remove(&sample_index).unwrap_or_else(|| vec![]);
                            let sample = Sample {
                                image_file,
                                boxes,
                                size: image_size.clone(),
                            };
                            Ok(sample)
                        })
                        .try_collect()?;

                    let series = TimeSeries {
                        name: series_name.clone(),
                        samples,
                    };

                    Ok((series_name, series))
                })
                .try_collect()?;

            let weights: Vec<_> = series.values().map(|series| series.samples.len()).collect();

            Ok(Self {
                min_length,
                series,
                weights,
            })
        }

        pub fn sample(&self, length: usize) -> Result<&[Sample]> {
            let Self {
                min_length,
                ref series,
                ref weights,
            } = *self;

            ensure!(
                length <= min_length,
                "the sampling length {} is greater the minimum sequence length {}",
                length,
                min_length
            );

            // select series
            let mut rng = rand::thread_rng();
            let dist = dists::WeightedIndex::new(weights).unwrap();
            let series_index = dist.sample(&mut rng);
            let series = &self.series[series_index];

            // sample images
            let end = series.samples.len() - length;
            let start_index = rng.gen_range(0..=end);
            let end_index = start_index + length;
            let samples = &series.samples[start_index..end_index];

            Ok(samples)
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    struct SeriesEntry {
        pub name: String,
        pub count: usize,
        pub height: usize,
        pub width: usize,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct LabelEntry {
        pub series: String,
        pub index: usize,
        pub class: String,
        pub top: f64,
        pub left: f64,
        pub height: f64,
        pub width: f64,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn simple_dataset_test() -> Result<()> {
            // let hx = 256;
            // let wx = 256;

            let dataset = SimpleDataset::load(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("test")
                    .join("demo-dataset"),
                8,
            )
            .await?;

            (1..=5).try_for_each(|len| -> Result<_> {
                let samples = dataset.sample(len)?;
                ensure!(samples.len() == len);

                Ok(())
            })?;

            Ok(())
        }
    }
}

async fn load_classes_file(path: impl AsRef<Path>) -> Result<IndexSet<String>> {
    let path = path.as_ref();
    let content = tokio::fs::read_to_string(path).await?;
    let lines: Vec<_> = content.lines().collect();
    let classes: IndexSet<_> = lines.iter().cloned().map(ToOwned::to_owned).collect();
    ensure!(
        lines.len() == classes.len(),
        "duplicated class names found in '{}'",
        path.display()
    );
    ensure!(
        !classes.is_empty(),
        "no classes found in '{}'",
        path.display()
    );
    Ok(classes)
}
