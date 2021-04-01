pub use anyhow::{ensure, format_err, Context, Error, Result};
pub use approx::assert_abs_diff_eq;
pub use futures::stream::{self, StreamExt, TryStreamExt};
pub use gstreamer::{self as gst, prelude::*};
pub use gstreamer_app as gst_app;
pub use indexmap::IndexMap;
pub use itertools::{izip, Itertools};
pub use log::warn;
pub use opencv::{
    highgui,
    prelude::*,
    videoio::{self, VideoCapture},
};
pub use par_stream::TryParStreamExt;
pub use rand::prelude::*;
pub use regex::Regex;
pub use serde::{Deserialize, Serialize};
pub use std::{
    array,
    borrow::{Borrow, Cow},
    collections::{self, HashMap},
    fs,
    iter::{self, FromIterator},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Once},
};
pub use structopt::StructOpt;
pub use tch::{
    kind::FLOAT_CPU,
    nn::{self, OptimizerConfig},
    vision, Device, Kind, Tensor,
};
pub use tch_goodies::{Activation, TensorExt};
pub use tch_tensor_like::TensorLike;
pub use tokio_stream::wrappers::ReadDirStream;
pub use unzip_n::unzip_n;

pub type Fallible<T> = Result<T>;

unzip_n!(pub 2);
unzip_n!(pub 8);
