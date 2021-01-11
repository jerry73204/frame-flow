use crate::common::*;

pub fn play(url: &str) -> Result<()> {
    gst::init()?;

    let pipeline = gst::Pipeline::new(None);
    let src = gst::ElementFactory::make("filesrc", None)?;
    let decodebin = gst::ElementFactory::make("decodebin", None)?;

    src.set_property("location", &url)?;
    pipeline.add_many(&[&src, &decodebin])?;
    gst::Element::link_many(&[&src, &decodebin])?;

    let pipeline_weak = pipeline.downgrade();

    decodebin.connect_pad_added(move |_dbin, src_pad| {
        let pipeline = (|| -> Option<_> {
            let pipeline = pipeline_weak.upgrade()?;

            let is_video = src_pad
                .get_current_caps()?
                .get_structure(0)?
                .get_name()
                .starts_with("video/");

            if !is_video {
                return None;
            }

            Some(pipeline)
        })();

        let pipeline = match pipeline {
            Some(pipeline) => pipeline,
            None => return,
        };

        let result = (|| -> Result<_> {
            let queue = gst::ElementFactory::make("queue", None)?;
            let videoconvert = gst::ElementFactory::make("videoconvert", None)?;
            let appsink = gst::ElementFactory::make("appsink", None)?;

            let elements = &[&queue, &videoconvert, &appsink];
            pipeline.add_many(elements)?;
            gst::Element::link_many(elements)?;

            elements
                .iter()
                .try_for_each(|elem| elem.sync_state_with_parent())?;

            appsink
                .dynamic_cast::<gst_app::AppSink>()
                .unwrap()
                .set_callbacks(
                    gst_app::AppSinkCallbacks::builder()
                        .new_sample(|sink| {
                            let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;

                            let structure =
                                || -> Option<_> { Some(sample.get_caps()?.get_structure(0)?) }()
                                    .ok_or_else(|| gst::FlowError::Error)?;

                            let caps = || -> Result<_> {
                                let format = structure.get::<String>("format")?;
                                let height = structure.get::<i32>("height")?;
                                let width = structure.get::<i32>("width")?;
                                let framerate = structure.get::<gst::Fraction>("framerate")?;
                                let pixel_aspect_ratio =
                                    structure.get::<gst::Fraction>("pixel-aspect-ratio")?;
                                Ok((format, height, width, framerate, pixel_aspect_ratio))
                            }()
                            .map_err(|_err| gst::FlowError::Error)?;

                            let buffer =
                                sample.get_buffer().ok_or_else(|| gst::FlowError::Error)?;
                            let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                            let pts = buffer.get_pts();
                            let data = map.as_slice().to_vec();

                            Ok(gst::FlowSuccess::Ok)
                        })
                        .build(),
                );

            let sink_pad = queue.get_static_pad("sink").unwrap();
            src_pad.link(&sink_pad)?;

            Ok(())
        })();

        result.unwrap();

        // if let Err(err) = result {
        //     gst::gst_element_error!(
        //         dbin,
        //         gst::LibraryError::Failed,
        //         ("Failed to insert sink"),
        //         details: gst::Structure::builder("error-details")
        //             .field("error", &format!("{:?}", err))
        //             .build()

        //     );
        // }
    });

    pipeline.set_state(gst::State::Playing)?;

    let bus = pipeline.get_bus().unwrap();

    bus.iter_timed(gst::CLOCK_TIME_NONE)
        .try_for_each(|msg| -> Result<_> {
            use gst::MessageView;

            match msg.view() {
                MessageView::Eos(..) => return Ok(()),
                MessageView::Error(err) => {
                    pipeline.set_state(gst::State::Null)?;
                    panic!("{:?}", err);
                }
                MessageView::StateChanged(_state) => {}
                _ => (),
            }

            Ok(())
        })?;

    pipeline.set_state(gst::State::Null)?;

    Ok(())
}
