use rand::prelude::*;
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Normal, Triangular};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops;
use std::cmp;
use itertools::izip;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use pyo3::prelude::*;

use crate::standardize::Standardizer;


fn num_layers() -> u8 {
    // Generate a random number of Creature modifier layers
    *[1, 1, 1, 2, 2, 3].choose(&mut rand::thread_rng()).unwrap()
}


#[pyfunction]
pub fn load_creature_from_json(json: &str) -> Creature {
    serde_json::from_str(&json).unwrap()
}


/**
A "Creature" is essentially a randomly generated function.
The equation of a creature can be one or more Coefficients in one or more
LayerModifiers which function as one or more layers for a simple neural network.
*/
#[pyclass]
#[derive(Clone)]
#[derive(Serialize, Deserialize)]
pub struct Creature {
    equation: Vec<LayerModifiers>,
    pub cached_error_sum: Option<f32>,
    #[pyo3(get)]
    pub generation: u8,
    #[pyo3(get)]
    pub offspring: u8,
}

impl PartialEq for Creature {
    fn eq(&self, other: &Self) -> bool {
        if self.generation != other.generation {
            return false
        }
        if self.offspring != other.offspring {
            return false
        }
        if self.equation.len() != other.equation.len() {
            return false
        }
        for (cr1_mods, cr2_mods) in izip!(&self.equation, &other.equation) {
            if cr1_mods != cr2_mods {
             return false
            }
        }
        true
    }
}

#[derive(Clone)]
pub enum MutateSpeed {
    Fine,
    Fast,
}

impl Creature {
    pub fn new(parameter_options: &Vec<&str>, max_layers: u8) -> Self {
        let mut equation = Vec::new();

        let mut layer_limit = num_layers();
        if layer_limit > max_layers {
            layer_limit = max_layers;
        }

        for layer in 0..layer_limit {
            equation.push(LayerModifiers::new(
                layer == 0,
                parameter_options,
            ));
        }
        Creature { equation, cached_error_sum: None, generation: 1, offspring: 0 }
    }

    /// Calculate the resulting output value for this creature given an input of Key: Value data.
    pub fn rust_calculate(&self, parameters: &HashMap<String, f32>) -> f32 {
        let mut total = 0.0;
        let mut inner_total = 0.0;

        for layer_modifiers in &self.equation {
            // Run through each input parameter and record impact
            // for each parameter that is used in the curret layer's modifiers.
            for (param, param_value) in parameters {
                if let Some(coefficients) = layer_modifiers.modifiers.get(param) {
                    inner_total += coefficients.calculate(param_value);
                }
            }

            // Check if current layer applies coefficients to the total after previous layer
            // Since "total" is updated at the end of each full layer, that same "total"
            // is the resulf of the prevous layer used as an input parameter.
            if let Some(t_coefficients) = &layer_modifiers.previous_layer_coefficients {
                inner_total += t_coefficients.calculate(&total);
            }

            // Add in the bias "layer_bias" to the current layer's calculation.
            total = inner_total + layer_modifiers.layer_bias;
        }
        total
    }

    pub fn create_many(num_creatures: u32, parameter_options: &Vec<&str>, max_layers: u8) -> Vec<Creature> {
        let creatures: Vec<Creature> = (0..num_creatures)
            .map(|_| Creature::new(parameter_options, max_layers))
            .collect();
        creatures
    }

    pub fn create_many_parallel(num_creatures: u32, parameter_options: &Vec<&str>, max_layers: u8) -> Vec<Creature> {
        let creatures: Vec<Creature> = (0..num_creatures)
            .into_par_iter()
            .map(|_| Creature::new(parameter_options, max_layers))
            .collect();
        creatures
    }

    pub fn mutate(&self, mutate_speed: MutateSpeed) -> Creature {
        let modify_value = match mutate_speed {
                MutateSpeed::Fine => 0.005,
                MutateSpeed::Fast => 0.05,
        };

        let mut rng = thread_rng();
        let norm = Normal::new(0.0, modify_value).unwrap();

        let mut new_equation: Vec<LayerModifiers> = Vec::new();
        for layer_mods in &self.equation {
            let layer_bias = match rng.gen::<f64>() {
                x if x < 0.5 => layer_mods.layer_bias + rng.sample(norm),
                _ => layer_mods.layer_bias,
            };

            let mut modified_coefficients = |coeff: &Coefficients| {
                Coefficients {
                    c: coeff.c + rng.sample(norm),
                    b: coeff.b + rng.sample(norm),
                    z: coeff.z + rng.sample(norm),
                    x: match rng.gen::<f64>() {
                        num if num < 0.2 => coeff.x + 1,
                        num if num < 0.4 && coeff.x > 1 => coeff.x - 1,
                        _ => coeff.x,
                    }
                }
            };

            let previous_layer_coefficients = match &layer_mods.previous_layer_coefficients {
                Some(coeff) => Some(modified_coefficients(coeff)),
                None => None,
            };

            let mut modifiers = HashMap::new();
            for (param, coeff) in &layer_mods.modifiers {
                modifiers.insert(param.to_owned(), modified_coefficients(coeff));
            }

            let new_layer_mods = LayerModifiers {
                modifiers,
                previous_layer_coefficients,
                layer_bias,
            };

            new_equation.push(new_layer_mods);
        }
        Creature { equation: new_equation, cached_error_sum: None, generation: &self.generation + 1, offspring: self.offspring }
    }
}

impl ops::Add for &Creature {
    type Output = Creature;

    fn add(self, other: Self) -> Creature {

        let mut rng = thread_rng();
        let tri = Triangular::new(0.7, 1.3, 1.0).unwrap();
        let norm = Normal::new(0.0, 0.1).unwrap();

        // Generate new number of layers
        let mut new_layers = ((self.num_layers() as f32 + other.num_layers() as f32) / 2.0).round() as usize;

        // Possible mutation to number of layers
        // Only add a layer if one of the input Creatures also has that number of layers
        // (So don't add an extra layer beyond either of the parents)
        // This avoids needing to have a refernce to potential parameters to create a new layer from scratch.
        if rng.gen::<f64>() < 0.05 {
            if new_layers > 1 && rng.gen::<f64>() < 0.5 {
                new_layers -= 1;
            } else if new_layers < cmp::max(self.num_layers(), other.num_layers()) {
                new_layers += 1;
            }
        }

        let mut new_equation = Vec::new();

        // Generate new, mutated coefficients
        //for (layer_mods_1, layer_mods_2) in zip(&self.equation, &other.equation) {
        //println!("New Number of Layers: {}", new_layers);
        for index in 0..new_layers {
            let layer_mods_1 = self.equation.get(index);
            let layer_mods_2 = other.equation.get(index);

            // layer bias
            let new_bias = match (layer_mods_1, layer_mods_2) {
                (Some(mods1), Some(mods2)) => (mods1.layer_bias + mods2.layer_bias) / 2.0 * rng.sample(tri),
                (Some(mods1), None) => mods1.layer_bias * rng.sample(tri),
                (None, Some(mods2)) => mods2.layer_bias * rng.sample(tri),
                (None, None) => {
                    match rng.gen::<f64>() {
                        x if (0.0..=0.2).contains(&x) => 0.0,
                        _ => rng.sample(norm),
                    }
                },
            };

            // previous layer coefficients
            let prev_layer_coeff: Option<Coefficients> = match (layer_mods_1, layer_mods_2) {
                (Some(mods1), Some(mods2)) => match (&mods1.previous_layer_coefficients, &mods2.previous_layer_coefficients) {
                        (Some(prev1), Some(prev2)) => Some(prev1 + prev2),
                        (Some(prev1), None) => Some(prev1.mutate(&mut rng, &tri)),
                        (None, Some(prev2)) => Some(prev2.mutate(&mut rng, &tri)),
                        _ => None,
                    },
                (Some(mods1), None) => match &mods1.previous_layer_coefficients {
                        Some(prev1) => Some(prev1.mutate(&mut rng, &tri)),
                        None => None,
                    },
                (None, Some(mods2)) => match &mods2.previous_layer_coefficients {
                        Some(prev2) => Some(prev2.mutate(&mut rng, &tri)),
                        None => None,
                    },
                (None, None) => None,
            };


            // specific parameter coefficients
            let param_options: HashSet<String> = match (layer_mods_1, layer_mods_2) {
                (Some(mods1), Some(mods2)) => {
                    let mut params: HashSet<String> = mods1.modifiers.keys().cloned().collect();
                    params.extend(mods2.modifiers.keys().cloned().collect::<HashSet<String>>());
                    params
                },
                (Some(mods1), None) => mods1.modifiers.keys().cloned().collect(),
                (None, Some(mods2)) => mods2.modifiers.keys().cloned().collect(),
                (None, None) => HashSet::new(),  // should never be triggered, but need to provide default empty value...
            };

            let mut new_modifiers = HashMap::new();
            for param in param_options {
                let coeffs_1 = match layer_mods_1 {
                    Some(mods1) => mods1.modifiers.get(&param),
                    None => None,
                };
                let coeffs_2 = match layer_mods_2 {
                    Some(mods2) => mods2.modifiers.get(&param),
                    None => None,
                };
                if rng.gen::<f64>() < 0.8 {  // 20% chance of not including current param for this layer
                    match (coeffs_1, coeffs_2) {
                        (Some(coeff), None) => new_modifiers.insert(param.to_string(), coeff.mutate(&mut rng, &tri)),
                        (None, Some(coeff)) => new_modifiers.insert(param.to_string(), coeff.mutate(&mut rng, &tri)),
                        (Some(coeff_1), Some(coeff_2)) => {
                            let new_1 = coeff_1.mutate(&mut rng, &tri);
                            let new_2 = coeff_2.mutate(&mut rng, &tri);
                            new_modifiers.insert(param.to_string(), new_1 + new_2)
                        },
                        (None, None) => {
                            if rng.gen::<f64>() < 0.1 {
                                new_modifiers.insert(param.to_string(), Coefficients::new());
                            }
                            None
                        },
                    };
                }
            }


            let new_layer = LayerModifiers {
                modifiers: new_modifiers,
                previous_layer_coefficients: prev_layer_coeff,
                layer_bias: new_bias,
            };

            new_equation.push(new_layer);
        }

        Creature {
            equation: new_equation,
            cached_error_sum: None,
            generation: cmp::max(self.generation, other.generation),
            offspring: cmp::max(self.offspring, other.offspring) + 1,
        }
    }
}


#[pymethods]
impl Creature {
    #[getter]
    pub fn num_layers(&self) -> usize {
        self.equation.len()
    }

    pub fn used_parameters(&self) -> HashSet<String> {
        let mut params = HashSet::new();
        for layer_modifiers in &self.equation {
            for param in layer_modifiers.modifiers.keys() {
                params.insert(param.clone());
            }
        }
        params
    }

    pub fn __repr__(&self) -> String {
         format!("Rust Creature: {} Layers - {} Generation - {} Offspring", self.num_layers(), self.generation, self.offspring)
    }

    /// Calculate the resulting output value for this creature given an input of Key: Value data.
    /// Python version here which takes an actual HashMap instead of a reference
    pub fn calculate(&self, parameters: HashMap<String, f32>) -> f32 {
        let mut total = 0.0;
        let mut inner_total = 0.0;

        for layer_modifiers in &self.equation {
            // Run through each input parameter and record impact
            // for each parameter that is used in the curret layer's modifiers.
            for (param, param_value) in &parameters {
                if let Some(coefficients) = layer_modifiers.modifiers.get(param) {
                    inner_total += coefficients.calculate(param_value);
                }
            }

            // Check if current layer applies coefficients to the total after previous layer
            // Since "total" is updated at the end of each full layer, that same "total"
            // is the resulf of the prevous layer used as an input parameter.
            if let Some(t_coefficients) = &layer_modifiers.previous_layer_coefficients {
                inner_total += t_coefficients.calculate(&total);
            }

            // Add in the bias "layer_bias" to the current layer's calculation.
            total = inner_total + layer_modifiers.layer_bias;
        }
        total
    }

    fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

impl Creature {
    // Create a string which creates a Python module with callable regression function
    pub fn python_regression_module_string(&self, standardizer: &Standardizer, target: &String) -> String {
        let mut s = "'''\nPython regression function module generated by Evogression\n'''\n\n".to_string();
        s += "def regression(parameters: dict) -> float:\n";
        s += format!("    '''\n    Function for predicting '{}'\n    Generated from an Evogression Creature\n    '''\n\n", target).as_str();

        let used_parameters = self.used_parameters();
        if !used_parameters.is_empty() {
            s += "    # Standardize input data\n";
            s += "    data = {}\n";
            s += "    for param, value in parameters.items():\n";
            for (param, param_stand) in &standardizer.standardizers {
                if used_parameters.contains(param) {
                    s += format!("        if param == '{}':\n", param).as_str();
                    match param_stand.stdev {
                        x if x == 0.0 => s += format!("            data['{}'] = value - {:.5}\n", param, param_stand.mean).as_str(),
                        _ => s += format!("            data['{}'] = (value - {:.5}) / {:.5}\n", param, param_stand.mean, param_stand.stdev).as_str(),
                    }
                }
            }
        }

        s += "\n    T = 0\n\n";

        fn coeff_equation_string(coeff: &Coefficients, value: &str) -> String {
            match coeff.x {
                0 => format!("    T += {:.4}\n", coeff.c),  // zero exponent makes second chunk equal 1
                1 => format!("    T += {:.4} * {} + {:.4}\n", coeff.c * coeff.b, value, coeff.c * coeff.z),
                2 => {
                    // c * (b * param + z) ^ 2
                    // c * [b^2 * param^2 + 2*z*b*param + z^2]
                    // c * b^2 * param^2 + 2*c*z*b*param + c * z^2
                    let c1 = coeff.c * coeff.b.powf(2.0);
                    let c2 = 2.0 * coeff.c * coeff.z * coeff.b;
                    let c3 = coeff.c * coeff.z.powf(2.0);
                    format!("    T += {c1:.4} * {value} ** 2 + {c2:.4} * {value} + {c3:.4}\n")
                },
                // default calculation that the above matches simplify/expand
                _ => format!("    T += {:.4} * ({:.4} * {} + {:.4}) ** {:.2}\n", coeff.c, coeff.b, value, coeff.z, coeff.x),
            }
        }

        for (i, layer_mods) in self.equation.iter().enumerate() {
            s += format!("    # LAYER_{}\n", i+1).as_str();
            s += format!("    T += {:.4}\n", layer_mods.layer_bias).as_str();
            if let Some(coeff) = &layer_mods.previous_layer_coefficients {
                s += coeff_equation_string(coeff, "previous_T").as_str();
            }
            for (param, coeff) in &layer_mods.modifiers {
                let val = format!("data['{}']", param);
                s += coeff_equation_string(coeff, val.as_str()).as_str();
            }
            if i < self.equation.len() - 1 {
                s += "    previous_T, T = T, 0\n";
            }
            s += "\n";
        }

        // if let to take care of Option, but this should ALWAYS trigger
        if let Some(target_std) = standardizer.standardizers.get(target) {
            s += "    # Unstandardize result\n";
            s += format!("    T = T * {:.5} + {:.5}\n\n", target_std.stdev, target_std.mean).as_str();
        }

        s += "    return T\n";

        s.to_string()
    }
}

impl fmt::Display for Creature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, " Creature")?;
        for (i, layer_mod) in self.equation.iter().enumerate() {
            write!(f, "  Layer {}\n{}", i+1, layer_mod)?;
        }
        Ok(())
    }
}

/// Each "LayerModifiers" represents a full neural network layer.
/// "modifiers" is a collection of Coefficents applied to certain input parameters.
/// The "previous_layer_coefficients" field is Coefficients applied to a previous layer's output, if applicable.
/// The "layer_bias" field is a bias added to the layer's calculation.

#[derive(Debug, Clone, PartialEq)]
#[derive(Serialize, Deserialize)]
struct LayerModifiers {
    modifiers: HashMap<String, Coefficients>,
    previous_layer_coefficients: Option<Coefficients>,
    layer_bias: f32,
}

impl LayerModifiers {
    fn new(first_layer: bool, parameter_options: &Vec<&str>) -> LayerModifiers {
        let mut rng = thread_rng();

        let mut modifiers = HashMap::new();
        let param_usage_scalar = 2.5 / (parameter_options.len() as f64 + 1.0);
        for &param in parameter_options {
            if rng.gen::<f64>() < param_usage_scalar {
                modifiers.insert(param.to_string(), Coefficients::new());
            }
        }

        let previous_layer_coefficients = match first_layer {
            false => Some(Coefficients::new()),
            true => None,
        };

        let norm = Normal::new(0.0, 0.1).unwrap();
        let layer_bias = match rng.gen::<f64>() {
            x if (0.0..=0.2).contains(&x) => 0.0,
            _ => rng.sample(norm),
        };
        LayerModifiers { modifiers, previous_layer_coefficients, layer_bias }
    }
}
impl fmt::Display for LayerModifiers {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "    Bias:  {:.4}", self.layer_bias)?;
        if let Some(coeff) = &self.previous_layer_coefficients {
            writeln!(f, "    Previous Layer:   ->  {}", coeff)?
        }
        for (key, coeff) in &self.modifiers {
            writeln!(f, "    Param \"{}\"   ->   {}", key, coeff)?;
        }
        Ok(())
    }
}

/// A "Coefficients" struct contains 4 values which
/// are used to form the following equation given input "param":
/// Value = C * (B * param + Z) ^ X
#[derive(Debug, Clone, PartialEq)]
#[derive(Serialize, Deserialize)]
struct Coefficients { c: f32, b: f32, z: f32, x: u8 }

impl Coefficients {
    fn calculate(&self, &param_value: &f32) -> f32 {
        self.c * (self.b * param_value + self.z).powi(self.x as i32)
    }

    fn new() -> Coefficients {
        let mut rng = thread_rng();
        let tri_a = Triangular::new(0.0, 2.0, 1.0).unwrap();
        let tri_b = Triangular::new(-2.0, 2.0, 0.0).unwrap();

        let mut c = if rng.gen::<f64>() < 0.4 { 1.0 } else { rng.sample(tri_a) };
        let mut b = if rng.gen::<f64>() < 0.3 { 1.0 } else { rng.sample(tri_a) };
        let z = if rng.gen::<f64>() < 0.4 { 0.0 } else { rng.sample(tri_b) };

        if rng.gen::<f64>() < 0.5 { c = -c; }
        if rng.gen::<f64>() < 0.5 { b = -b; }

        let x = match rng.gen::<f64>() {
            x if x <= 0.4 => 1,
            x if x <= 0.75 => 2,
            _ => 3,
        };
        Coefficients { c, b, z, x }
    }

    fn mutate(&self, rng: &mut ThreadRng, distr: &Triangular<f32>) -> Coefficients {
        Coefficients {
            c: self.c * rng.sample(distr),
            b: self.b * rng.sample(distr),
            z: self.z * rng.sample(distr),
            x: (self.x as f32 * rng.sample(distr)).round() as u8,
        }
    }
}

impl ops::Add for Coefficients {
    /// Add just averages two Coefficients together
    type Output = Coefficients;
    fn add(self, other: Self) -> Coefficients {
        Coefficients {
            c: (self.c + other.c) / 2.0,
            b: (self.b + other.b) / 2.0,
            z: (self.z + other.z) / 2.0,
            x: ((self.x + other.x) as f32 / 2.0).round() as u8,
        }
    }
}
impl ops::Add for &Coefficients {
    /// Add just averages two Coefficients together
    type Output = Coefficients;
    fn add(self, other: Self) -> Coefficients {
        Coefficients {
            c: (self.c + other.c) / 2.0,
            b: (self.b + other.b) / 2.0,
            z: (self.z + other.z) / 2.0,
            x: ((self.x + other.x) as f32 / 2.0).round() as u8,
        }
    }
}

impl fmt::Display for Coefficients {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.4} * ({:.4} * param + {:.4}) ^ {}", self.c, self.b, self.z, self.x)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn creature_creation_and_calculation() {
        let param_options = vec!["width", "height", "weight"];
        let creature = Creature::new(&param_options, 3);
        println!("\n\n{}\n", creature);

        assert!(creature.num_layers() >= 1 && creature.num_layers() <= 3);

        let test_coeff = creature.equation[0].modifiers.values().next()
            .expect("\n--> OKAY if this fails occasionally as it is possible to \
                     \ngenerate a creature with no modifiers for the first layer.");
        println!("{}", test_coeff);
        assert!((test_coeff.c.abs() + test_coeff.b.abs()) > 0.0);

        let input_data = HashMap::from([("width".to_string(), 2.1245), ("height".to_string(), 0.52412)]);

        let mut creatures = Vec::new();
        for _ in 0..15 {
            creatures.push(Creature::new(&param_options, 3));
        }
        println!("\n{}", creatures[5]);
        println!("\n{}", creatures[10]);
        println!("\n{}\n", creatures[12]);

        let mut total = 0.0;
        let mut result;
        for cr in creatures {
            result = cr.rust_calculate(&input_data);
            println!("{}", result);
            total += result;
        }
        assert!(total != 0.0);
    }

    #[test]
    fn generate_many_creatures() {
        let param_options = vec!["width", "height", "weight"];

        let t0 = Instant::now();
        Creature::create_many(100000, &param_options, 3);
        let single = Instant::now() - t0;
        println!("\nSingle Thread: {:.2?}", single);

        let t0 = Instant::now();
        Creature::create_many_parallel(100000, &param_options, 3);
        let multi = Instant::now() - t0;
        println!("Multiple Threads: {:.2?}", multi);

        println!("Multicore Speed: {:.1}x\n", single.as_millis() as f32 / multi.as_millis() as f32);
        assert!(multi < single);

        for cr in Creature::create_many_parallel(100000, &param_options, 2) {
            assert!(cr.num_layers() <= 2);
        }
    }

    #[test]
    fn check_mutation() {
        let param_options = vec!["width", "height", "weight"];
        let creature = Creature::new(&param_options, 3);
        let mutant1 = creature.mutate(MutateSpeed::Fast);
        let mutant2 = creature.mutate(MutateSpeed::Fine);
        assert!((mutant1 != creature) && (mutant2 != creature));
    }

    #[test]
    fn creature_addition() {
        let param_options = vec!["x", "y"];
        for _ in 0..1000 {
            let cr1 = Creature::new(&param_options, 3);
            let cr2 = Creature::new(&param_options, 3);
            let offspring = &cr1 + &cr2;
            assert!(offspring.num_layers() > 0);
        }
    }

    #[test]
    fn num_layer_bounds() {
        let layers: Vec<u8> = (0..10000).map(|_| num_layers()).collect();
        assert_eq!(*layers.iter().min().unwrap(), 1);
        assert_eq!(*layers.iter().max().unwrap(), 3);
    }
}
