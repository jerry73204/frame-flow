pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use approx::{abs_diff_eq, assert_abs_diff_eq};
pub use by_address::ByAddress;
pub use chrono::{DateTime, Local};
pub use collected::{AddVal, Count, First, GroupHashMap, Last, MaxVal};
pub use cv_convert::{FromCv as _, TryFromCv as _};
pub use derivative::Derivative;
pub use futures::{
    future::FutureExt as _,
    stream::{self, Stream, StreamExt as _, TryStreamExt as _},
};
pub use indexmap::{IndexMap, IndexSet};
pub use iterator_ext::IteratorExt as _;
pub use itertools::{chain, iproduct, izip, Itertools};
pub use mona::prelude::*;
pub use ndarray as nd;
pub use noisy_float::types::{r64, R64};
pub use num_integer::Integer as _;
pub use num_traits::{Float, Num};
pub use opencv::{core as core_cv, imgproc, prelude::*};
pub use owning_ref::{ArcRef, VecRef};
pub use palette::{convert::IntoColor, Hsv, RgbHue, Srgb};
pub use par_stream::{ParStreamExt, TryParStreamExt};
pub use rand::{distributions as dists, prelude::*};
pub use serde::{Deserialize, Serialize};
pub use std::{
    array,
    borrow::{Borrow, Cow},
    collections::{self, hash_map, HashMap, HashSet},
    convert::TryInto,
    f64, fs,
    iter::{self, FromIterator, Sum},
    num::NonZeroUsize,
    ops::{Add, Div, Mul, Sub},
    path::{Path, PathBuf},
    sync::{Arc, Once},
    time::{Duration, Instant},
};
pub use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    nn::{self, Module as _, ModuleT as _, OptimizerConfig},
    vision, Device, IndexOp, Kind, Reduction, Tensor,
};
pub use bbox::prelude::*;
pub use log::{info, warn};
pub use tch_act::TensorActivationExt as _;
pub use tch_goodies::TensorExt as _;
pub use tch_tensor_like::TensorLike;
pub use tokio::sync::mpsc;
pub use tokio_stream::wrappers::ReadDirStream;

pub type RectLabel = label::Label<bbox::CyCxHW<R64>, usize>;

use unzip_n::unzip_n;
unzip_n!(pub 2);
unzip_n!(pub 3);
unzip_n!(pub 6);
unzip_n!(pub 7);
unzip_n!(pub 8);
unzip_n!(pub 9);
unzip_n!(pub 11);
