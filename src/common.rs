pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use approx::assert_abs_diff_eq;
pub use by_address::ByAddress;
pub use chrono::{DateTime, Local};
pub use collected::{GroupHashMap, MaxVal};
pub use futures::{
    future::FutureExt,
    stream::{self, Stream, StreamExt, TryStreamExt},
};
pub use indexmap::{IndexMap, IndexSet};
pub use iterator_ext::IteratorExt;
pub use itertools::{iproduct, izip, Itertools};
pub use mona::prelude::*;
pub use noisy_float::types::{r64, R64};
pub use num_traits::{Float, Num};
pub use owning_ref::ArcRef;
pub use par_stream::{ParStreamExt, TryParStreamExt};
pub use rand::{distributions as dists, prelude::*};
pub use serde::{Deserialize, Serialize};
pub use std::{
    array,
    borrow::{Borrow, Cow},
    collections::{self, HashMap, HashSet},
    convert::TryInto,
    fs,
    iter::{self, FromIterator, Sum},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Once},
};
pub use structopt::StructOpt;
pub use tch::{
    kind::FLOAT_CPU,
    nn::{self, Module as _, ModuleT as _, OptimizerConfig},
    vision, Device, IndexOp, Kind, Reduction, Tensor,
};
pub use tch_goodies::{
    Activation, DenseDetectionTensorList, GridSize, OptionalTensorList, PixelCyCxHW,
    PixelRectLabel, PixelRectTransform, PixelSize, RatioCyCxHW, RatioRectLabel, RatioSize,
    RatioUnit, Rect, TensorExt, TensorList, NONE_TENSORS,
};
pub use tch_tensor_like::TensorLike;
pub use tfrecord::{EventWriter, EventWriterInit};
pub use tokio::sync::mpsc;
pub use tokio_stream::wrappers::ReadDirStream;
pub use tracing::{error, info, info_span, instrument, trace, trace_span, warn, Instrument};
pub use unzip_n::unzip_n;

pub type Fallible<T> = Result<T>;

unzip_n!(pub 2);
unzip_n!(pub 6);
unzip_n!(pub 8);
