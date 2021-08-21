use crate::common::*;
use num_traits::{NumCast, ToPrimitive};

pub trait SequentialExt {
    fn inspect<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) + Send;
}

impl SequentialExt for nn::Sequential {
    fn inspect<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) + Send,
    {
        self.add_fn(move |xs| {
            f(xs);
            xs.shallow_clone()
        })
    }
}

impl SequentialExt for nn::SequentialT {
    fn inspect<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) + Send,
    {
        self.add_fn(move |xs| {
            f(xs);
            xs.shallow_clone()
        })
    }
}

pub trait SequentialTExt {
    fn inspect_t<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) + Send;
}

impl SequentialTExt for nn::SequentialT {
    fn inspect_t<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) + Send,
    {
        self.add_fn_t(move |xs, train| {
            f(xs, train);
            xs.shallow_clone()
        })
    }
}

pub trait NumFrom<T> {
    fn num_from(from: T) -> Self;
}

impl<T, U> NumFrom<T> for U
where
    T: ToPrimitive,
    U: NumCast,
{
    fn num_from(from: T) -> Self {
        <Self as NumCast>::from(from).unwrap()
    }
}

pub trait NumInto<T> {
    fn num_into(self) -> T;
}

impl<T, U> NumInto<T> for U
where
    T: NumFrom<U>,
{
    fn num_into(self) -> T {
        T::num_from(self)
    }
}
