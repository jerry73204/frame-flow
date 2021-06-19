use anyhow::Result;
use frame_flow::config;
use std::{env, path::PathBuf};
use structopt::StructOpt;
use tracing_subscriber::{filter::LevelFilter, prelude::*, EnvFilter};

#[derive(Debug, Clone, StructOpt)]
/// Implementation for 'frame-flow' model.
pub struct Args {
    #[structopt(long, default_value = "config.json5")]
    pub config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    // setup tracing
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true).compact();
    let filter_layer = {
        let filter = EnvFilter::from_default_env();
        if env::var("RUST_LOG").is_err() {
            filter.add_directive(LevelFilter::INFO.into())
        } else {
            filter
        }
    };

    let tracer = opentelemetry_jaeger::new_pipeline()
        .with_service_name("train")
        .install_simple()?;
    let otel_layer = tracing_opentelemetry::OpenTelemetryLayer::new(tracer);

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    let args = Args::from_args();
    let config = config::Config::load(&args.config)?;
    frame_flow::start(config).await?;

    Ok(())
}
