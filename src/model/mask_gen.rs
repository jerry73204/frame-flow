use crate::common::*;

#[derive(Debug, Clone)]
pub struct TimeMask {
    length: usize,
    input_shape: Arc<Vec<i64>>,
    context_shape: Arc<Vec<i64>>,
}

impl TimeMask {
    pub fn new(length: usize, input_shape: &[i64], context_shape: &[i64]) -> Self {
        Self {
            length,
            input_shape: Arc::new(input_shape.to_owned()),
            context_shape: Arc::new(context_shape.to_owned()),
        }
    }

    pub fn make_iter(&self) -> TimeMaskIter {
        let Self {
            length,
            ref input_shape,
            ref context_shape,
        } = *self;

        TimeMaskIter {
            length,
            step: 0,
            input_shape: input_shape.clone(),
            context_shape: context_shape.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TimeMaskIter {
    length: usize,
    step: usize,
    input_shape: Arc<Vec<i64>>,
    context_shape: Arc<Vec<i64>>,
}

impl Iterator for TimeMaskIter {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            length,
            ref mut step,
            ref input_shape,
            ref context_shape,
        } = *self;

        if *step < length {
            *step += 1;
            todo!()
        } else {
            None
        }
    }
}
