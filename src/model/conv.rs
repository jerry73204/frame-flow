use crate::common::*;

pub trait ConvParam
where
    Self: Clone,
{
    fn dim(&self) -> usize;
    fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>>;
}

impl ConvParam for usize {
    fn dim(&self) -> usize {
        1
    }

    fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
        Box::new(iter::once(*self as i64))
    }
}

impl<const DIM: usize> ConvParam for [usize; DIM] {
    fn dim(&self) -> usize {
        DIM
    }

    fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
        Box::new(Vec::from(*self).into_iter().map(|val| val as i64))
    }
}

impl ConvParam for Vec<usize> {
    fn dim(&self) -> usize {
        self.len()
    }

    fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
        Box::new(self.clone().into_iter().map(|val| val as i64))
    }
}

impl ConvParam for &[usize] {
    fn dim(&self) -> usize {
        self.len()
    }

    fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
        Box::new(Vec::<usize>::from(*self).into_iter().map(|val| val as i64))
    }
}

#[derive(Debug, Clone)]
pub struct ConvNDInit<Param: ConvParam> {
    pub ksize: Param,
    pub stride: Param,
    pub padding: Param,
    pub dilation: Param,
    pub groups: usize,
    pub bias: bool,
    pub ws_init: nn::Init,
    pub bs_init: nn::Init,
}

pub type ConvNDInit1D = ConvNDInit<usize>;
pub type ConvNDInit2D = ConvNDInit<[usize; 2]>;
pub type ConvNDInit3D = ConvNDInit<[usize; 3]>;
pub type ConvNDInit4D = ConvNDInit<[usize; 4]>;
pub type ConvNDInitDyn = ConvNDInit<Vec<usize>>;

impl<const DIM: usize> ConvNDInit<[usize; DIM]> {
    pub fn new(ksize: usize) -> Self {
        Self {
            ksize: [ksize; DIM],
            stride: [1; DIM],
            padding: [ksize / 2; DIM],
            dilation: [1; DIM],
            groups: 1,
            bias: true,
            ws_init: nn::Init::KaimingUniform,
            bs_init: nn::Init::Const(0.0),
        }
    }

    pub fn into_dyn(self) -> ConvNDInitDyn {
        let Self {
            ksize,
            stride,
            padding,
            dilation,
            groups,
            bias,
            ws_init,
            bs_init,
        } = self;

        ConvNDInitDyn {
            ksize: Vec::from(ksize),
            stride: Vec::from(stride),
            padding: Vec::from(padding),
            dilation: Vec::from(dilation),
            groups,
            bias,
            ws_init,
            bs_init,
        }
    }
}

impl ConvNDInit<Vec<usize>> {
    pub fn new(ndims: usize, ksize: usize) -> Self {
        Self {
            ksize: vec![ksize; ndims],
            stride: vec![1; ndims],
            padding: vec![ksize / 2; ndims],
            dilation: vec![1; ndims],
            groups: 1,
            bias: true,
            ws_init: nn::Init::KaimingUniform,
            bs_init: nn::Init::Const(0.0),
        }
    }

    // pub fn cat(params: impl IntoIterator<Item = Self>) -> Self {
    //     params
    //         .into_iter()
    //         .map(|conv| {
    //             let Self {
    //                 ksize,
    //                 stride,
    //                 padding,
    //                 dilation,
    //                 groups,
    //                 bias,
    //                 ws_init,
    //                 bs_init,
    //             } = conv;

    //             (
    //                 ksize, stride, padding, dilation, groups, bias, ws_init, bs_init,
    //             )
    //         })
    //         .unzip_n();
    // }
}

impl<Param: ConvParam> ConvNDInit<Param> {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<ConvND> {
        let Self {
            ksize,
            stride,
            padding,
            dilation,
            groups,
            bias,
            ws_init,
            bs_init,
        } = self;

        ensure!(
            groups > 0 && in_dim % groups == 0,
            "in_dim must be multiple of group"
        );
        ensure!(
            ksize.dim() == stride.dim()
                && ksize.dim() == padding.dim()
                && ksize.dim() == dilation.dim(),
            "parameter dimension mismatch"
        );

        let path = path.borrow();
        let conv_dim = ksize.dim();
        let in_dim = in_dim as i64;
        let out_dim = out_dim as i64;
        let ksize: Vec<i64> = ksize.i64_iter().collect();
        let stride: Vec<i64> = stride.i64_iter().collect();
        let padding: Vec<i64> = padding.i64_iter().collect();
        let dilation: Vec<i64> = dilation.i64_iter().collect();
        let groups = groups as i64;

        let bs = bias.then(|| path.var("bias", &[out_dim], bs_init));
        let ws = {
            let weight_size: Vec<i64> = vec![out_dim, in_dim / groups]
                .into_iter()
                .chain(ksize)
                .collect();
            path.var("weight", weight_size.as_slice(), ws_init)
        };

        Ok(ConvND {
            stride,
            padding,
            dilation,
            output_padding: vec![0; conv_dim],
            groups,
            weight: ws,
            bias: bs,
        })
    }
}

pub struct ConvND {
    stride: Vec<i64>,
    padding: Vec<i64>,
    dilation: Vec<i64>,
    output_padding: Vec<i64>,
    groups: i64,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl ConvND {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let Self {
            ref stride,
            ref padding,
            ref dilation,
            groups,
            ref output_padding,
            ref weight,
            ref bias,
        } = *self;

        input.convolution(
            weight,
            bias.as_ref(),
            &stride,
            &padding,
            &dilation,
            false, // transposed,
            &output_padding,
            groups,
        )
    }
}
