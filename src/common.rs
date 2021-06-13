pub use anyhow::{ensure, format_err, Context, Error, Result};
pub use approx::assert_abs_diff_eq;
pub use collected::MaxCollector;
pub use futures::{
    future::FutureExt,
    stream::{self, Stream, StreamExt, TryStreamExt},
};
pub use indexmap::IndexMap;
pub use itertools::{izip, Itertools};
pub use noisy_float::types::{r64, R64};
pub use par_stream::{ParStreamExt, TryParStreamExt};
pub use rand::prelude::*;
pub use serde::{Deserialize, Serialize};
pub use std::{
    array,
    borrow::{Borrow, Cow},
    collections::{self, HashMap},
    fs,
    iter::{self, FromIterator, Sum},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Once},
};
pub use structopt::StructOpt;
pub use tch::{
    kind::FLOAT_CPU,
    nn::{self, Module, ModuleT, OptimizerConfig},
    vision, Device, IndexOp, Kind, Tensor,
};
pub use tch_goodies::{Activation, OptionalTensorList, TensorExt, TensorList, NONE_TENSORS};
pub use tch_tensor_like::TensorLike;
pub use tfrecord::{EventWriter, EventWriterInit};
pub use tokio_stream::wrappers::ReadDirStream;
pub use tracing::{error, info, info_span, instrument, trace, trace_span, warn, Instrument};
pub use unzip_n::unzip_n;

pub type Fallible<T> = Result<T>;

unzip_n!(pub 2);
unzip_n!(pub 8);
