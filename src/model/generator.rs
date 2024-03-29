use super::{
    attention::SelfAttentionInit,
    misc::{NormKind, PaddingKind},
    unet_block::{UnetBlock, UnetBlockInit, UnetModule},
};
use crate::common::*;

pub use generic::*;
mod generic {
    use super::*;

    #[derive(Debug)]
    pub enum Generator {
        Unet(UnetGenerator),
        Resnet(ResnetGenerator),
        Custom(CustomGenerator),
    }

    impl nn::ModuleT for Generator {
        fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
            match self {
                Generator::Unet(model) => model.forward_t(input, train),
                Generator::Resnet(model) => model.forward_t(input, train),
                Generator::Custom(model) => model.forward_t(input, train),
            }
        }
    }

    impl From<ResnetGenerator> for Generator {
        fn from(v: ResnetGenerator) -> Self {
            Self::Resnet(v)
        }
    }

    impl From<UnetGenerator> for Generator {
        fn from(v: UnetGenerator) -> Self {
            Self::Unet(v)
        }
    }

    impl From<CustomGenerator> for Generator {
        fn from(v: CustomGenerator) -> Self {
            Self::Custom(v)
        }
    }
}

pub use unet::*;
mod unet {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct UnetGeneratorInit<const NUM_BLOCKS: usize> {
        pub norm_kind: NormKind,
        pub dropout: bool,
    }

    impl<const NUM_BLOCKS: usize> Default for UnetGeneratorInit<NUM_BLOCKS> {
        fn default() -> Self {
            Self {
                norm_kind: NormKind::BatchNorm,
                dropout: false,
            }
        }
    }

    impl<const NUM_BLOCKS: usize> UnetGeneratorInit<NUM_BLOCKS> {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            inner_c: usize,
        ) -> UnetGenerator {
            let path = path.borrow();
            let Self { norm_kind, dropout } = self;

            let block = UnetBlockInit {
                norm_kind,
                ..Default::default()
            }
            .build(
                path / "block_0",
                inner_c * 8,
                inner_c * 8,
                inner_c * 8,
                UnetModule::inner_most(),
            );

            let block = (1..=(NUM_BLOCKS - 5)).fold(block, |prev_block, index| {
                UnetBlockInit { norm_kind, dropout }.build(
                    path / format!("block_{}", index),
                    inner_c * 8,
                    inner_c * 8,
                    inner_c * 8,
                    UnetModule::standard(move |xs, train| prev_block.forward_t(xs, train)),
                )
            });

            let block = UnetBlockInit {
                norm_kind,
                ..Default::default()
            }
            .build(
                path / format!("block_{}", NUM_BLOCKS - 4),
                inner_c * 4,
                inner_c * 4,
                inner_c * 8,
                UnetModule::standard(move |xs, train| block.forward_t(xs, train)),
            );

            let block = UnetBlockInit {
                norm_kind,
                ..Default::default()
            }
            .build(
                path / format!("block_{}", NUM_BLOCKS - 3),
                inner_c * 2,
                inner_c * 2,
                inner_c * 4,
                UnetModule::standard(move |xs, train| block.forward_t(xs, train)),
            );

            let block = UnetBlockInit {
                norm_kind,
                ..Default::default()
            }
            .build(
                path / format!("block_{}", NUM_BLOCKS - 2),
                inner_c,
                inner_c,
                inner_c * 2,
                UnetModule::standard(move |xs, train| block.forward_t(xs, train)),
            );

            let block = UnetBlockInit {
                norm_kind,
                ..Default::default()
            }
            .build(
                path / format!("block_{}", NUM_BLOCKS - 1),
                in_c,
                out_c,
                inner_c,
                UnetModule::outer_most(move |xs, train| block.forward_t(xs, train)),
            );

            UnetGenerator { block }
        }
    }

    #[derive(Debug)]
    pub struct UnetGenerator {
        block: UnetBlock,
    }

    impl nn::ModuleT for UnetGenerator {
        fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            self.block.forward_t(xs, train)
        }
    }
}

pub use resnet::*;
mod resnet {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ResnetGeneratorInit {
        pub padding_kind: PaddingKind,
        pub norm_kind: NormKind,
        pub dropout: bool,
        pub ksize: usize,
        pub num_scale_blocks: usize,
        pub num_blocks: usize,
    }

    impl Default for ResnetGeneratorInit {
        fn default() -> Self {
            Self {
                padding_kind: PaddingKind::Reflect,
                norm_kind: NormKind::InstanceNorm,
                dropout: false,
                ksize: 5,
                num_scale_blocks: 2,
                num_blocks: 2,
            }
        }
    }

    impl ResnetGeneratorInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            inner_c: usize,
        ) -> ResnetGenerator {
            let path = path.borrow();
            let Self {
                padding_kind,
                norm_kind,
                dropout,
                ksize,
                num_scale_blocks,
                num_blocks,
            } = self;
            let in_c = in_c as i64;
            let out_c = out_c as i64;
            let inner_c = inner_c as i64;
            let bias = norm_kind == NormKind::InstanceNorm;
            let padding = ksize / 2;
            let ksize = ksize as i64;

            // first block
            let seq = {
                let path = path / "block_0";
                nn::seq_t()
                    .add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / "conv",
                        in_c,
                        inner_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c))
                    .add_fn(|xs| xs.lrelu())
            };

            // down sampling
            let seq = (0..num_scale_blocks).fold(seq, |seq, index| {
                let path = path / format!("block_{}", index + 1);
                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    inner_c,
                    ksize,
                    nn::ConvConfig {
                        stride: 2,
                        padding: padding as i64,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.lrelu())
            });

            // resnet blocks
            let seq = {
                (0..num_blocks).fold(seq, |seq, index| {
                    let path = path / format!("block_{}", index + num_scale_blocks + 1);

                    let branch = {
                        let branch = nn::seq_t()
                            .add(padding_kind.build([padding, padding, padding, padding]))
                            .add(nn::conv2d(
                                &path / "conv1",
                                inner_c,
                                inner_c,
                                ksize,
                                nn::ConvConfig {
                                    padding: 0,
                                    bias,
                                    ..Default::default()
                                },
                            ))
                            .add(norm_kind.build(&path / "norm1", inner_c))
                            .add_fn(|xs| xs.lrelu());

                        let branch = if dropout {
                            branch.add_fn_t(|xs, train| xs.dropout(0.5, train))
                        } else {
                            branch
                        };

                        branch
                            .add(padding_kind.build([padding, padding, padding, padding]))
                            .add(nn::conv2d(
                                &path / "conv2",
                                inner_c,
                                inner_c,
                                ksize,
                                nn::ConvConfig {
                                    padding: 0,
                                    bias,
                                    ..Default::default()
                                },
                            ))
                            .add(norm_kind.build(&path / "norm2", inner_c))
                    };

                    seq.add_fn_t(move |xs, train| xs + branch.forward_t(xs, train))
                })
            };

            // up sampling
            let seq = (0..num_scale_blocks).fold(seq, |seq, index| {
                let path = path / format!("block_{}", index + num_scale_blocks + num_blocks + 1);
                seq.add_fn(|xs| {
                    let (_, _, h, w) = xs.size4().unwrap();
                    xs.upsample_nearest2d(&[h * 2, w * 2], None, None)
                })
                .add(padding_kind.build([padding, padding, padding, padding]))
                .add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    inner_c,
                    ksize,
                    nn::ConvConfig {
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.lrelu())
            });

            // last block
            let seq = {
                let path = path / format!("block_{}", num_scale_blocks * 2 + num_blocks + 1);
                seq.add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / "conv",
                        inner_c,
                        out_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            ..Default::default()
                        },
                    ))
                    .add_fn(|xs| xs.tanh())
            };

            ResnetGenerator { seq }
        }
    }

    #[derive(Debug)]
    pub struct ResnetGenerator {
        seq: nn::SequentialT,
    }

    impl nn::ModuleT for ResnetGenerator {
        fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            self.seq.forward_t(xs, train)
        }
    }
}

pub use custom::*;
mod custom {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct CustomGeneratorInit {
        pub padding_kind: PaddingKind,
        pub norm_kind: NormKind,
        pub dropout: bool,
        pub ksize: usize,
        pub num_scale_blocks: usize,
        pub num_blocks: usize,
    }

    impl Default for CustomGeneratorInit {
        fn default() -> Self {
            Self {
                padding_kind: PaddingKind::Reflect,
                norm_kind: NormKind::InstanceNorm,
                dropout: false,
                ksize: 5,
                num_scale_blocks: 2,
                num_blocks: 2,
            }
        }
    }

    impl CustomGeneratorInit {
        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_c: usize,
            out_c: usize,
            inner_c: usize,
        ) -> CustomGenerator {
            let path = path.borrow();
            let Self {
                padding_kind,
                norm_kind,
                dropout,
                ksize,
                num_scale_blocks,
                num_blocks,
            } = self;
            let in_c = in_c as i64;
            let out_c = out_c as i64;
            let inner_c = inner_c as i64;
            let bias = norm_kind == NormKind::InstanceNorm;
            let padding = ksize / 2;
            let ksize = ksize as i64;

            // first block
            let seq = {
                let path = path / "block_0";
                nn::seq_t()
                    .add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / "conv",
                        in_c,
                        inner_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            bias,
                            ..Default::default()
                        },
                    ))
                    .add(norm_kind.build(&path / "norm", inner_c))
                    .add_fn(|xs| xs.lrelu())
            };

            // down sampling
            let seq = (0..num_scale_blocks).fold(seq, |seq, index| {
                let path = path / format!("block_{}", index + 1);
                seq.add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    inner_c,
                    ksize,
                    nn::ConvConfig {
                        stride: 2,
                        padding: padding as i64,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.lrelu())
            });

            // attention blocks
            let seq = {
                (0..num_blocks).fold(seq, |seq, index| {
                    let path = path / format!("block_{}", index + num_scale_blocks + 1);

                    let branch = {
                        let branch = nn::seq_t()
                            .add(
                                SelfAttentionInit::<2> {
                                    ksize: ksize as usize,
                                    n_heads: 1,
                                    key_c: inner_c as usize / 2,
                                    value_c: inner_c as usize / 2,
                                    bias,
                                }
                                .build(&path / "attention1", inner_c as usize, inner_c as usize)
                                .unwrap(),
                            )
                            .add(norm_kind.build(&path / "norm1", inner_c))
                            .add_fn(|xs| xs.lrelu());

                        let branch = if dropout {
                            branch.add_fn_t(|xs, train| xs.dropout(0.5, train))
                        } else {
                            branch
                        };

                        branch
                            .add(
                                SelfAttentionInit::<2> {
                                    ksize: ksize as usize,
                                    n_heads: 1,
                                    key_c: inner_c as usize / 2,
                                    value_c: inner_c as usize / 2,
                                    bias,
                                }
                                .build(&path / "attention2", inner_c as usize, inner_c as usize)
                                .unwrap(),
                            )
                            .add(norm_kind.build(&path / "norm2", inner_c))
                    };

                    seq.add_fn_t(move |xs, train| xs + branch.forward_t(xs, train))
                })
            };

            // up sampling
            let seq = (0..num_scale_blocks).fold(seq, |seq, index| {
                let path = path / format!("block_{}", index + num_scale_blocks + num_blocks + 1);
                seq.add_fn(|xs| {
                    let (_, _, h, w) = xs.size4().unwrap();
                    xs.upsample_nearest2d(&[h * 2, w * 2], None, None)
                })
                .add(padding_kind.build([padding, padding, padding, padding]))
                .add(nn::conv2d(
                    &path / "conv",
                    inner_c,
                    inner_c,
                    ksize,
                    nn::ConvConfig {
                        padding: 0,
                        bias,
                        ..Default::default()
                    },
                ))
                .add(norm_kind.build(&path / "norm", inner_c))
                .add_fn(|xs| xs.lrelu())
            });

            // last block
            let seq = {
                let path = path / format!("block_{}", num_scale_blocks * 2 + num_blocks + 1);
                seq.add(padding_kind.build([padding, padding, padding, padding]))
                    .add(nn::conv2d(
                        &path / "conv",
                        inner_c,
                        out_c,
                        ksize,
                        nn::ConvConfig {
                            padding: 0,
                            ..Default::default()
                        },
                    ))
                    .add_fn(|xs| xs.tanh())
            };

            CustomGenerator { seq }
        }
    }

    #[derive(Debug)]
    pub struct CustomGenerator {
        seq: nn::SequentialT,
    }

    impl nn::ModuleT for CustomGenerator {
        fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            self.seq.forward_t(xs, train)
        }
    }
}

// mod custom {
//     use super::*;

//     #[derive(Debug, Clone)]
//     pub struct CustomGeneratorInit<const DEPTH: usize> {
//         pub ndims: usize,
//         pub input_channels: usize,
//         pub output_channels: usize,
//         pub num_heads: usize,
//         pub strides: [usize; DEPTH],
//         pub block_channels: [usize; DEPTH],
//         pub context_channels: [usize; DEPTH],
//         pub repeats: [usize; DEPTH],
//     }

//     impl<const DEPTH: usize> CustomGeneratorInit<DEPTH> {
//         pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<CustomGenerator> {
//             ensure!(DEPTH >= 1, "zero depth is not allowed");

//             let path = path.borrow();
//             let Self {
//                 ndims,
//                 num_heads,
//                 input_channels,
//                 output_channels,
//                 strides,
//                 block_channels,
//                 context_channels,
//                 repeats,
//             } = self;
//             let first_attention_channels = block_channels[0];
//             let last_attention_channels = *block_channels.last().unwrap();

//             let first_conv = ConvNDInitDyn::new(ndims, 1).build(
//                 path / "first_conv",
//                 input_channels,
//                 first_attention_channels,
//             )?;
//             let last_conv = ConvNDInitDyn {
//                 transposed: true,
//                 ..ConvNDInitDyn::new(ndims, 1)
//             }
//             .build(
//                 path / "last_conv",
//                 first_attention_channels,
//                 output_channels,
//             )?;

//             // encoder blocks
//             let enc_blocks = izip!(
//                 array::IntoIter::new(repeats),
//                 array::IntoIter::new(block_channels),
//                 iter::once(first_attention_channels).chain(array::IntoIter::new(block_channels)),
//                 array::IntoIter::new(context_channels),
//             )
//             .enumerate()
//             .map(|(index, (repeat, out_c, in_c, context_c))| -> Result<_> {
//                 let block = BlockInit {
//                     ndims,
//                     input_channels: in_c,
//                     context_channels: context_c,
//                     output_channels: out_c,
//                     repeat,
//                     num_heads,
//                     keyvalue_channels: context_c,
//                     conv_ksize: 3,
//                     conv_transposed: false,
//                 }
//                 .build(path / format!("enc_block_{}", index))?;
//                 Ok(block)
//             })
//             .try_collect()?;

//             // down sample modules
//             let down_samples: Vec<_> = izip!(
//                 array::IntoIter::new(strides),
//                 array::IntoIter::new(block_channels),
//             )
//             .enumerate()
//             .map(|(index, (stride, inout_c))| -> Result<_> {
//                 let conv = ConvNDInitDyn {
//                     stride: vec![stride; ndims],
//                     transposed: false,
//                     ..ConvNDInitDyn::new(ndims, 1)
//                 }
//                 .build(
//                     path / format!("down_sample_{}", index),
//                     inout_c,
//                     inout_c,
//                 )?;
//                 Ok(conv)
//             })
//             .try_collect()?;

//             // decoder blocks
//             let dec_blocks: Vec<_> = izip!(
//                 array::IntoIter::new(repeats).rev(),
//                 array::IntoIter::new(block_channels).rev(),
//                 array::IntoIter::new(block_channels)
//                     .rev()
//                     .chain(iter::once(first_attention_channels))
//                     .skip(1),
//                 array::IntoIter::new(context_channels).rev(),
//             )
//             .enumerate()
//             .map(|(index, (repeat, in_c, out_c, context_c))| -> Result<_> {
//                 let block = BlockInit {
//                     ndims,
//                     input_channels: in_c,
//                     context_channels: context_c,
//                     output_channels: out_c,
//                     repeat,
//                     num_heads,
//                     keyvalue_channels: context_c,
//                     conv_ksize: 3,
//                     conv_transposed: true,
//                 }
//                 .build(path / format!("enc_block_{}", index))?;

//                 Ok(block)
//             })
//             .try_collect()?;

//             // up sample modules
//             let up_samples: Vec<_> = izip!(
//                 array::IntoIter::new(strides).rev(),
//                 array::IntoIter::new(block_channels).rev(),
//             )
//             .enumerate()
//             .map(|(index, (stride, inout_c))| -> Result<_> {
//                 let conv = ConvNDInitDyn {
//                     stride: vec![stride; ndims],
//                     transposed: true,
//                     ..ConvNDInitDyn::new(ndims, 1)
//                 }
//                 .build(
//                     path / format!("down_sample_{}", index),
//                     inout_c,
//                     inout_c,
//                 )?;
//                 Ok(conv)
//             })
//             .try_collect()?;

//             // top block

//             let top_block = BlockInit {
//                 ndims,
//                 input_channels: last_attention_channels,
//                 context_channels: last_attention_channels,
//                 output_channels: last_attention_channels,
//                 repeat: 3,
//                 num_heads,
//                 keyvalue_channels: last_attention_channels,
//                 conv_ksize: 3,
//                 conv_transposed: false,
//             }
//             .build(path / "top_block")?;

//             Ok(CustomGenerator {
//                 enc_blocks,
//                 dec_blocks,
//                 down_samples,
//                 up_samples,
//                 top_block,
//                 first_conv,
//                 last_conv,
//                 num_contexts: repeats.into(),
//             })
//         }
//     }

//     #[derive(Debug)]
//     pub struct CustomGenerator {
//         enc_blocks: Vec<Block>,
//         dec_blocks: Vec<Block>,
//         down_samples: Vec<ConvND>,
//         up_samples: Vec<ConvND>,
//         top_block: Block,
//         first_conv: ConvND,
//         last_conv: ConvND,
//         num_contexts: Vec<usize>,
//     }

//     impl CustomGenerator {
//         pub fn forward_t(
//             &self,
//             input: &Tensor,
//             contexts: impl OptionalTensorList,
//             train: bool,
//         ) -> Result<CustomGeneratorOutput> {
//             let Self {
//                 enc_blocks,
//                 dec_blocks,
//                 down_samples,
//                 up_samples,
//                 top_block,
//                 first_conv,
//                 last_conv,
//                 num_contexts,
//             } = self;
//             let input_contexts = contexts.into_optional_tensor_list();

//             ensure!(
//                 input_contexts
//                     .as_ref()
//                     .map(|contexts| {
//                         let total: usize = num_contexts.iter().cloned().sum();
//                         contexts.len() == total
//                     })
//                     .unwrap_or(true),
//                 "number of contexts does not match"
//             );

//             // first conv
//             // eprintln!("first_conv");

//             // dbg!(input.mean(Kind::Float));
//             let xs = first_conv.forward(input);
//             // dbg!(xs.mean(Kind::Float));

//             // encoder
//             let (xs, contexts_vec, mut output_paddings_vec) = {
//                 let mut contexts_vec = vec![];
//                 let mut output_paddings_vec = vec![];

//                 let xs = match input_contexts {
//                     Some(input_contexts) => {
//                         let input_contexts_iter = num_contexts.iter().scan(0, |index, len| {
//                             let curr = *index;
//                             let next = curr + len;
//                             *index = next;
//                             Some(&input_contexts[curr..next])
//                         });

//                         izip!(enc_blocks.iter(), down_samples.iter(), input_contexts_iter)
//                             .enumerate()
//                             .try_fold(
//                                 xs,
//                                 |xs, (index, (block, down_sample, in_contexts))| -> Result<_> {
//                                     eprintln!("enc_block {}", index);
//                                     let BlockOutput {
//                                         feature: xs,
//                                         contexts: out_contexts,
//                                         ..
//                                     } = block.forward_t(xs, in_contexts, None, train)?;
//                                     let before_shape = xs.size();
//                                     let xs = down_sample.forward(&xs);
//                                     let after_shape = xs.size();
//                                     let output_paddings: Vec<_> = izip!(
//                                         &before_shape[2..],
//                                         &after_shape[2..],
//                                         down_sample.stride()
//                                     )
//                                     .map(|(&before_size, &after_size, &stride)| {
//                                         before_size - ((after_size - 1) * stride + 1)
//                                     })
//                                     .collect();

//                                     contexts_vec.push(out_contexts);
//                                     output_paddings_vec.push(output_paddings);

//                                     Ok(xs)
//                                 },
//                             )?
//                     }
//                     None => izip!(enc_blocks.iter(), down_samples.iter())
//                         .enumerate()
//                         .try_fold(xs, |xs, (index, (block, down_sample))| -> Result<_> {
//                             eprintln!("enc_block {}", index);
//                             let BlockOutput {
//                                 feature: xs,
//                                 contexts: out_contexts,
//                                 ..
//                             } = block.forward_t(xs, NONE_TENSORS, None, train)?;
//                             let before_shape = xs.size();
//                             let xs = down_sample.forward(&xs);
//                             let after_shape = xs.size();
//                             let output_paddings: Vec<_> =
//                                 izip!(&before_shape[2..], &after_shape[2..], down_sample.stride())
//                                     .map(|(&before_size, &after_size, &stride)| {
//                                         before_size - ((after_size - 1) * stride + 1)
//                                     })
//                                     .collect();

//                             contexts_vec.push(out_contexts);
//                             output_paddings_vec.push(output_paddings);

//                             Ok(xs)
//                         })?,
//                 };

//                 (xs, contexts_vec, output_paddings_vec)
//             };

//             // top
//             eprintln!("top");
//             let BlockOutput { feature: xs, .. } =
//                 top_block.forward_t(xs, NONE_TENSORS, None, train)?;

//             // reverse contexts
//             let contexts_vec: Vec<_> = contexts_vec
//                 .into_iter()
//                 .rev()
//                 .map(|mut contexts| {
//                     contexts.reverse();
//                     contexts
//                 })
//                 .collect();
//             output_paddings_vec.reverse();

//             // decoder
//             let (xs, mut output_contexts) = {
//                 let mut output_contexts = vec![];

//                 let xs = izip!(
//                     dec_blocks.iter(),
//                     up_samples.iter(),
//                     contexts_vec,
//                     output_paddings_vec
//                 )
//                 .enumerate()
//                 .try_fold(
//                     xs,
//                     |xs, (index, (block, up_sample, contexts, output_paddings))| -> Result<_> {
//                         eprintln!("dec_block {}", index);
//                         let xs = up_sample.forward_ext(&xs, Some(&output_paddings));

//                         let BlockOutput {
//                             feature: xs,
//                             contexts: out_contexts,
//                             ..
//                         } = block.forward_t(&xs, &contexts, None, train)?;

//                         output_contexts.extend(out_contexts);
//                         Ok(xs)
//                     },
//                 )?;

//                 (xs, output_contexts)
//             };

//             // last conv
//             let xs = last_conv.forward(&xs);

//             // reverse output contexts
//             output_contexts.reverse();

//             Ok(CustomGeneratorOutput {
//                 output: xs,
//                 contexts: output_contexts,
//             })
//         }

//         // pub fn clamp_bn_var(&mut self) {
//         //     let Self {
//         //         enc_blocks,
//         //         dec_blocks,
//         //         top_block,
//         //         ..
//         //     } = self;

//         //     enc_blocks.iter_mut().for_each(|block| {
//         //         block.clamp_bn_var();
//         //     });

//         //     dec_blocks.iter_mut().for_each(|block| {
//         //         block.clamp_bn_var();
//         //     });

//         //     top_block.clamp_bn_var();
//         // }

//         // pub fn denormalize_bn(&mut self) {
//         //     let Self {
//         //         enc_blocks,
//         //         dec_blocks,
//         //         top_block,
//         //         ..
//         //     } = self;

//         //     enc_blocks.iter_mut().for_each(|block| {
//         //         block.denormalize_bn();
//         //     });

//         //     dec_blocks.iter_mut().for_each(|block| {
//         //         block.denormalize_bn();
//         //     });

//         //     top_block.denormalize_bn();
//         // }

//         pub fn grad(&self) -> CustomGeneratorGrad {
//             let Self {
//                 enc_blocks,
//                 dec_blocks,
//                 down_samples,
//                 up_samples,
//                 top_block,
//                 first_conv,
//                 last_conv,
//                 ..
//             } = self;

//             CustomGeneratorGrad {
//                 enc_blocks: enc_blocks.iter().map(|block| block.grad()).collect(),
//                 dec_blocks: dec_blocks.iter().map(|block| block.grad()).collect(),
//                 down_samples: down_samples.iter().map(|down| down.grad()).collect(),
//                 up_samples: up_samples.iter().map(|down| down.grad()).collect(),
//                 top_block: top_block.grad(),
//                 first_conv: first_conv.grad(),
//                 last_conv: last_conv.grad(),
//             }
//         }
//     }

//     #[derive(Debug)]
//     pub struct CustomGeneratorOutput {
//         pub output: Tensor,
//         pub contexts: Vec<Tensor>,
//     }

//     #[derive(Debug)]
//     pub struct CustomGeneratorGrad {
//         pub enc_blocks: Vec<BlockGrad>,
//         pub dec_blocks: Vec<BlockGrad>,
//         pub down_samples: Vec<ConvNDGrad>,
//         pub up_samples: Vec<ConvNDGrad>,
//         pub top_block: BlockGrad,
//         pub first_conv: ConvNDGrad,
//         pub last_conv: ConvNDGrad,
//     }
// }

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn generator_test() -> Result<()> {
    //     let bs = 2;
    //     let cx = 3;
    //     let cy = 4;
    //     let hx = 11;
    //     let wx = 13;

    //     let vs = nn::VarStore::new(Device::Cpu);
    //     let root = vs.root();
    //     let generator = CustomGeneratorInit {
    //         ndims: 2,
    //         input_channels: cx,
    //         output_channels: cy,
    //         num_heads: 7,
    //         strides: [2, 2],
    //         block_channels: [8, 16],
    //         context_channels: [5, 10],
    //         repeats: [2, 2],
    //     }
    //     .build(root)?;

    //     // first pass, without contexts
    //     let input = Tensor::rand(&[bs, cx as i64, hx, wx], FLOAT_CPU);
    //     let CustomGeneratorOutput {
    //         output, contexts, ..
    //     } = generator.forward_t(&input, NONE_TENSORS, true)?;
    //     ensure!(
    //         output.size() == vec![bs, cy as i64, hx, wx],
    //         "incorrect output shape"
    //     );

    //     // second pass, with contexts
    //     let CustomGeneratorOutput {
    //         output, contexts, ..
    //     } = generator.forward_t(&input, contexts, true)?;
    //     ensure!(
    //         output.size() == vec![bs, cy as i64, hx, wx],
    //         "incorrect output shape"
    //     );

    //     // third pass, with contexts
    //     let CustomGeneratorOutput { output, .. } = generator.forward_t(&input, contexts, true)?;
    //     ensure!(
    //         output.size() == vec![bs, cy as i64, hx, wx],
    //         "incorrect output shape"
    //     );

    //     Ok(())
    // }
}
