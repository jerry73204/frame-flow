use num_traits::{NumCast, ToPrimitive};

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
