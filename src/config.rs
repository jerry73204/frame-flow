use crate::common::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub input_url: String,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = json5::from_str(&fs::read_to_string(path)?)?;
        Ok(config)
    }
}
