use crate::common::*;

/// An utility that calculates message rate per period of time.
#[derive(Debug)]
pub struct RateMeter {
    count: f64,
    instant: Instant,
    interval: Duration,
}

impl RateMeter {
    pub fn new(interval: Duration) -> Self {
        Self {
            count: 0.0,
            instant: Instant::now(),
            interval,
        }
    }

    pub fn with_second_intertal() -> Self {
        Self::new(Duration::from_secs(1))
    }

    pub fn add(&mut self, addition: f64) {
        self.count += addition;
    }

    pub fn rate(&mut self) -> Option<f64> {
        let elapsed = self.instant.elapsed();
        if elapsed >= self.interval {
            let rate = self.count / elapsed.as_nanos() as f64 * 1_000_000_000.0;
            self.count = 0.0;
            self.instant = Instant::now();
            Some(rate)
        } else {
            None
        }
    }
}
