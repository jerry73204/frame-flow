use crate::common::*;
use iii_formosa_dataset as iii;

const III_IMAGE_CHANNELS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sample {
    pub image_file: PathBuf,
    pub annotation_file: PathBuf,
    pub annotation: iii::Annotation,
    pub size: PixelSize<usize>,
    pub boxes: Vec<PixelLabel>,
}

#[derive(Debug, Clone)]
pub struct TimeSeriesDataset {
    pub classes: IndexSet<String>,
    pub series: Vec<TimeSeries>,
}

#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub dir: PathBuf,
    pub samples: Vec<Sample>,
}

impl TimeSeriesDataset {
    pub async fn load(
        dataset_dir: impl AsRef<Path>,
        classes_file: impl AsRef<Path>,
        class_whitelist: Option<HashSet<String>>,
        blacklist_files: HashSet<PathBuf>,
    ) -> Result<Self> {
        let dataset_dir = dataset_dir.as_ref();
        let classes_file = classes_file.as_ref();

        // load classes file
        let classes = load_classes_file(&classes_file).await?;
        let classes = Arc::new(classes);
        let class_whitelist = Arc::new(class_whitelist);

        // list xml files
        let xml_files = {
            let dataset_dir = dataset_dir.to_owned();
            tokio::task::spawn_blocking(move || list_iii_xml_files(dataset_dir, blacklist_files))
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
                            .filter_map(|(obj, class_index)| {
                                let iii::BndBox {
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                } = obj.bndbox;
                                let bbox = match PixelCyCxHW::from_tlbr(ymin, xmin, ymax, xmax) {
                                    Ok(bbox) => bbox,
                                    Err(_err) => {
                                        warn!(
                                            "failed to parse file '{}': invalid bbox {:?}",
                                            annotation_file.display(),
                                            [ymin, xmin, ymax, xmax]
                                        );
                                        return None;
                                    }
                                };

                                let labeled_bbox = PixelLabel {
                                    cycxhw: bbox,
                                    class: class_index,
                                };
                                Some(labeled_bbox)
                            })
                            .collect();

                        let sample = Sample {
                            annotation,
                            annotation_file,
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

        Ok(Self { series, classes })
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
