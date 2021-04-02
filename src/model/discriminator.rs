use crate::common::*;
use nn::Module;
use tch_goodies::module::{ConvBn, ConvBnInitDyn, ConvND, ConvNDInitDyn};

#[derive(Debug, Clone)]
pub struct DiscriminatorInit<const DEPTH: usize> {
    pub ndims: usize,
    pub ksize: usize,
    pub input_channels: usize,
    pub channels: [usize; DEPTH],
    pub strides: [usize; DEPTH],
}

impl<const DEPTH: usize> DiscriminatorInit<DEPTH> {
    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<Discriminator> {
        ensure!(DEPTH > 0, "zero depth is not allowed");

        let path = path.borrow();
        let Self {
            ndims,
            ksize,
            input_channels,
            channels,
            strides,
        } = self;
        ensure!(
            (1..=3).contains(&ndims),
            "ndims must be one of 1, 2, 3, but get ndims = {}",
            ndims
        );

        let last_channels = *channels.last().unwrap();

        let tuples: Vec<_> = izip!(
            iter::once(input_channels).chain(array::IntoIter::new(channels)),
            array::IntoIter::new(channels),
            array::IntoIter::new(strides),
        )
        .enumerate()
        .map(|(index, (in_c, out_c, stride))| -> Result<_> {
            let conv = ConvBnInitDyn::new(ndims, ksize).build(
                path / format!("conv_{}", index),
                in_c,
                out_c,
            )?;

            let down_sample = ConvNDInitDyn {
                stride: vec![stride; ndims],
                ..ConvNDInitDyn::new(ndims, 1)
            }
            .build(path / format!("down_sample_{}", index), out_c, out_c)?;

            Ok((conv, down_sample))
        })
        .try_collect()?;

        let (convs, down_samples) = tuples.into_iter().unzip_n_vec();

        let linear = nn::linear(path / "linear", last_channels as i64, 1, Default::default());

        Ok(Discriminator {
            ndims,
            convs,
            down_samples,
            linear,
        })
    }
}

#[derive(Debug)]
pub struct Discriminator {
    ndims: usize,
    convs: Vec<ConvBn>,
    down_samples: Vec<ConvND>,
    linear: nn::Linear,
}

impl Discriminator {
    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            ndims,
            ref mut convs,
            ref down_samples,
            ref linear,
        } = *self;

        let xs = izip!(convs, down_samples).try_fold(
            input.shallow_clone(),
            |xs, (conv, down_sample)| -> Result<_> {
                let xs = conv.forward_t(&xs, train)?;
                let xs = down_sample.forward(&xs);
                Ok(xs)
            },
        )?;

        let xs = match ndims {
            1 => xs.adaptive_avg_pool1d(&[1]),
            2 => xs.adaptive_avg_pool2d(&[1, 1]),
            3 => xs.adaptive_avg_pool3d(&[1, 1, 1]),
            _ => unreachable!(),
        };
        let xs = xs.view(&xs.size()[0..2]);
        let xs = linear.forward(&xs);

        Ok(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminator_test() -> Result<()> {
        let bs = 2;
        let cx = 3;
        let hx = 11;
        let wx = 13;

        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();

        let mut discriminator = DiscriminatorInit {
            ndims: 2,
            ksize: 3,
            input_channels: cx,
            channels: [16, 32],
            strides: [2, 2],
        }
        .build(&root)?;

        let input = Tensor::rand(&[bs, cx as i64, hx, wx], FLOAT_CPU);
        let output = discriminator.forward_t(&input, true)?;

        ensure!(output.size() == vec![bs, 1], "incorrect output shape");

        Ok(())
    }
}
