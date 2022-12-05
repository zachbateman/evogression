use std::collections::HashMap;


pub struct Standardizer {
    standardizers: HashMap<String, ParamStandardizer>,
}

impl Standardizer {
    pub fn to_py_dicts(&self) -> HashMap<String, HashMap<String, f32>> {
        let mut output = HashMap::new();
        for (key, param_stand) in &self.standardizers {
            let param_map = HashMap::from([
                ("mean".to_string(), param_stand.mean),
                ("stdev".to_string(), param_stand.stdev),
            ]);
            output.insert(key.to_string(), param_map);
        }
        output
    }
}

impl Standardizer {
    pub fn new(data: &[HashMap<String, f32>]) -> Standardizer {
        let mut standardizers = HashMap::new();
        for key in data[0].keys() {
            let values: Vec<&f32> = data.iter().map(|hash| hash.get(key).unwrap()).collect();
            standardizers.insert(key.to_string(), ParamStandardizer::new(&values));
        }
        Standardizer { standardizers }
    }

    pub fn standardized_value(&self, data: &HashMap<String, f32>) -> HashMap<String, f32> {
        let mut standardized = HashMap::new();
        for (key, value) in data {
            standardized.insert(key.to_string(), self.standardizers.get(key)
                    .expect("Did not have a ParamStandardizer for a given key?")
                    .standardize(&value));
        }
        standardized
    }

    pub fn standardized_values(&self, data: &[HashMap<String, f32>]) -> Vec<HashMap<String, f32>> {
        let mut compiled = Vec::new();
        for row in data {
            let mut new_row = HashMap::new();
            for (key, value) in row {
                new_row.insert(key.to_string(), self.standardizers.get(key)
                        .expect("Did not have a ParamStandardizer for a given key?")
                        .standardize(&value));
            }
            compiled.push(new_row);
        }
        compiled
    }
    pub fn unstandardize_value(&self, param: &str, value: f32) -> f32 {
        self.standardizers.get(param)
            .unwrap_or_else(|| panic!("Unable to find ParamStandardizer for {}", param))
            .unstandardize(&value)
    }

    pub fn print_standardization(&self) {
        for (key, param_stand) in &self.standardizers {
            println!("Key: {}  ParamStand: {:?}", key, param_stand);
        }
    }
}

#[derive(Debug)]
pub struct ParamStandardizer {
    mean: f32,
    stdev: f32,
}

impl ParamStandardizer {
    fn new(values: &Vec<&f32>) -> ParamStandardizer {
        ParamStandardizer {
            mean: mean(values).expect("Cannot calculate mean for empty data"),
            stdev: std_deviation(&values[..]).expect("Cannot calculate std_deviation for empty data"),
        }
    }
    fn standardize(&self, value: &f32) -> f32 {
        (value - self.mean) / self.stdev
    }
    fn unstandardize(&self, value: &f32) -> f32 {
        value * self.stdev + self.mean
    }
}

/// Mean function taken from Rust Cookbook
fn mean(data: &[&f32]) -> Option<f32> {
    let sum: f32 = data.iter().copied().sum();
    let count = data.len();
    match count {
        positive if positive > 0 => Some(sum / count as f32),
        _ => None,
    }
}

/// Standard deviation function taken from Rust Cookbook
fn std_deviation(data: &[&f32]) -> Option<f32> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let variance = data.iter().map(|value| {
                let diff = data_mean - *value;
                diff * diff
            }).sum::<f32>() / (count-1) as f32;  // subtracting by 1 as assuming sample
            Some(variance.sqrt())
        },
        _ => None
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_calcs() {
        let v1: Vec<&f32> = vec![&3.0, &5.8, &1.5, &-3.7];
        assert_eq!((mean(&v1[..]).unwrap() - 1.65).abs() < 0.00001, true);
        let v2: Vec<&f32> = vec![&-87.3];
        assert_eq!((mean(&v2[..]).unwrap() - (-87.3)).abs() < 0.00001, true);
        let v3 = vec![];
        assert_eq!(mean(&v3[..]) == None, true);
    }

    #[test]
    fn std_calcs() {
        let data = &[&3.0, &1.1, &6.5, &1.7, &5.9, &8.3];
        let result = std_deviation(&data[..]).unwrap();
        println!("Std: {}", result);
        // checking against "sample" standard deviation method where divide by n-1
        // dividing by n for "population" would instead calculate 2.646
        assert_eq!((result - 2.89856).abs() < 0.0001, true);
    }
}
