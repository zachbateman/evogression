use rand::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use crate::standardize::Standardizer;
use crate::creature::{Creature, MutateSpeed};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use pyo3::prelude::*;

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct Evolution {
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    num_creatures: u32,
    #[pyo3(get)]
    num_cycles: u16,
    pub standardizer: Standardizer,
    #[pyo3(get)]
    pub best_creatures: Vec<Creature>,
    #[pyo3(get)]
    pub best_creature: Creature,
}

#[pymethods]
impl Evolution {
    fn predict_point(&self, data_point: HashMap<String, f32>) -> f32 {
        let standardized_point = self.standardizer.standardized_value(&data_point);
        let result = self.best_creature.rust_calculate(&standardized_point);
        self.standardizer.unstandardize_value(&self.target, result)
    }

    fn __repr__(&self) -> String {
        format!("Rust Evolution: {} Creatures - {} Cycles", self.num_creatures, self.num_cycles)
    }

    fn best_error(&self) -> f32 {
        self.best_creature.cached_error_sum.unwrap_or(999.9)
    }

    fn python_regression_module_string(&self) -> String {
        self.generate_python_regression_module_string()
    }

    fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[pyfunction]
pub fn load_evolution_from_json(json: &str) -> Evolution {
    serde_json::from_str(&json).unwrap()
}

impl Evolution {
    // Have this non-pymethods function as passing in &self.standardizer
    // was causing issues in the pymethods impl block
    fn generate_python_regression_module_string(&self) -> String {
        self.best_creature.python_regression_module_string(&self.standardizer, &self.target)
    }
}

impl Evolution {
    pub fn new(
        target: String,
        data: &[HashMap<String, f32>],
        num_creatures: u32,
        num_cycles: u16,
        max_layers: u8,
        optimize: bool,
    ) -> Evolution {
        assert!(num_creatures > 0);
        assert!(num_cycles > 0);
        assert!(max_layers > 0);

        let standardizer = Standardizer::new(data);
        let standardized_data = standardizer.standardized_values(data);

        let param_options = data[0].keys()
                                   .map(|s| s.as_str())
                                   .filter(|s| s != &target.as_str())
                                   .collect();

        let mut creatures = Creature::create_many_parallel(num_creatures, &param_options, max_layers);
        let mut best_creatures = Vec::with_capacity(10);  // Usually will have at least 10 of these
        let mut total_best_error = 999_999_999.0;
        let mut new_best_creature;

        let mut start = Instant::now();

        for cycle in 1..=num_cycles {
            creatures.par_iter_mut().for_each(|creature| {
                if creature.cached_error_sum.is_none() {
                    let err = calc_error_sum(creature, &standardized_data, &target);
                    creature.cached_error_sum = Some(err);
                }
            });

            new_best_creature = false;
            let (min_error, median_error) = error_results_with_median(&creatures);
            if min_error < total_best_error {
                total_best_error = min_error;
                new_best_creature = true;
            }

            let best_creature = creatures
                .iter()
                .find(|creature| creature.cached_error_sum == Some(min_error))
                .expect("Error matching min_error to a creature!");
            best_creatures.push(best_creature.clone());

            // Only print output at most every 0.1 seconds.
            // This should improve performance with less prints.
            // Typically should only affect very low num_creatures runs
            // where running lots quickly and the output isn't important.
            let duration = start.elapsed();
            if duration.as_millis() > 100 {
                start = Instant::now();
                print_cycle_data(cycle, median_error, best_creature, new_best_creature);
            }

            creatures = kill_weak_creatures(creatures, &median_error);
            creatures.append(&mut mutated_top_creatures(&creatures, &min_error, &median_error));

            // Shuffle creatures before mating
            creatures.shuffle(&mut thread_rng());
            creatures.append(&mut mate_creatures(&creatures, num_creatures - (creatures.len() as u32)));


            // Now ensure creatures is correct length by cutting off extras
            // or adding newly generated Creatures to fill to num_creatures length.
            creatures.truncate(num_creatures as usize);
            if creatures.len() < num_creatures as usize {
                creatures.append(&mut Creature::create_many_parallel(
                    num_creatures - creatures.len() as u32, &param_options, max_layers
                ));
            }
        }

        let mut min_error = 100_000_000_000.0;  // arbitrarily large starting number
        for creature in &best_creatures {
            if let Some(error) = creature.cached_error_sum {
                if error < min_error {
                    min_error = error;
                }
            }
        }

        let best_creature = best_creatures
            .iter()
            .find(|creature| creature.cached_error_sum == Some(min_error))
            .expect("Error matching min_error to a creature!");

        let optimized_creature = if optimize {
            // Next line calculates number of mutants per each optimization loop
            // Want to use less than 500 if small num_creatures so that runs faster
            let optimize_count = ((num_creatures / 5) as u16).clamp(30, 500);
            let optimized_creature = optimize_creature(best_creature, &standardized_data, &target, 30, optimize_count);
            print_optimize_data(best_creature.cached_error_sum.unwrap(),
                                optimized_creature.cached_error_sum.unwrap(),
                                &optimized_creature);
            optimized_creature
        } else {
            best_creature.clone()
        };

        Evolution {
            target,
            num_creatures,
            num_cycles,
            standardizer,
            best_creatures,
            best_creature: optimized_creature,
        }
    }
}

fn optimize_creature(creature: &Creature,
    data_points: &Vec<HashMap<String, f32>>,
    target: &str,
    iterations: u16,
    optimize_count: u16) -> Creature {

    let mut errors = Vec::with_capacity(iterations as usize);
    let mut best_error = creature.cached_error_sum.unwrap();
    let mut speed = MutateSpeed::Fast;
    let mut best_creature = creature.clone();
    for i in 0..=iterations {
        let mut creatures = Vec::with_capacity(51);
        creatures.push(best_creature.clone());
        creatures.extend((0..optimize_count).map(|_| best_creature.mutate(speed.clone())));

        creatures.par_iter_mut().for_each(|creature| {
            if creature.cached_error_sum.is_none() {
                let err = calc_error_sum(creature, data_points, target);
                creature.cached_error_sum = Some(err);
            }
        });

        let min_error = error_results(&creatures);
        errors.push(min_error);

        if min_error < best_error {
            best_error = min_error;
            best_creature = creatures
                .iter()
                .find(|creature| creature.cached_error_sum == Some(min_error))
                .expect("Error matching min_error to a creature!").clone();
        }

        if i > 5 && min_error / errors.get(errors.len() - 4).unwrap() > 0.9999 {
            speed = MutateSpeed::Fine;
        }
    }
    best_creature
}

fn print_optimize_data(start_error: f32, end_error: f32, best_creature: &Creature) {
    println!("\n\n--- FINAL OPTIMIZATION COMPLETE ---");
    println!("Start: {}    Best: {}", start_error, end_error);
    println!("  Generation: {}  Offspring: {}  Error: {}", best_creature.generation, best_creature.offspring, best_creature.cached_error_sum.unwrap());
    println!("{}", best_creature);
}

fn print_cycle_data(cycle: u16, median_error: f32, best_creature: &Creature, new_best_creature: bool) {
    println!("---------------------------------------");
    println!("Cycle - {} -", cycle);
    println!("Median error: {}", median_error);
    if new_best_creature {
        println!("New Best Creature:");
        println!("  Generation: {}  Offspring: {}  Error: {}", best_creature.generation, best_creature.offspring, best_creature.cached_error_sum.unwrap());
        println!("{}", best_creature);
    }
}

fn error_results_with_median(creatures: &[Creature]) -> (f32, f32) {
    let mut errors = Vec::with_capacity(creatures.len());
    for creature in creatures.iter() {
        // Now DON'T include any anomalous NaN calculations in resulting errors!
        match creature.cached_error_sum.unwrap() {
            x if !x.is_nan() => errors.push(x),
            _ => (),
        }
    }
    errors.sort_by(|a, b| a.total_cmp(b));

    match errors.len() {
        x if x > 0 => {
            let median_error = errors[errors.len() / 2];
            let min_error = errors[0];
            (min_error, median_error)
        },
        _ => {  // if an issue of no errors (very rare), throw huge errors instead of a panic
            (99999.9, 99999.9)
        }
    }

}

fn error_results(creatures: &[Creature]) -> f32 {
    let mut errors = Vec::with_capacity(creatures.len());
    for creature in creatures.iter() {
        // Now DON'T include any anomalous NaN calculations in resulting errors!
        match creature.cached_error_sum.unwrap() {
            x if !x.is_nan() => errors.push(x),
            _ => (),
        }
    }
    errors.sort_by(|a, b| a.total_cmp(b));
    match errors.len() {
        x if x > 0 => {
            errors[0]
        },
        _ => 99999.9  // if an issue of no errors (very rare), throw huge errors instead of a panic
    }
}

fn kill_weak_creatures(creatures: Vec<Creature>, median_error: &f32) -> Vec<Creature> {
    creatures.into_par_iter()
             .filter(|creature| creature.cached_error_sum.unwrap() < *median_error)
             .collect()
}

fn mutated_top_creatures(creatures: &Vec<Creature>, min_error: &f32, median_error: &f32) -> Vec<Creature> {
    let error_cutoff = (min_error + median_error) / 2.0;
    creatures.into_par_iter()
             .filter(|cr| cr.cached_error_sum.unwrap() < error_cutoff)
             .map(|cr| cr.mutate(MutateSpeed::Fast))
             .collect()
}

fn mate_creatures(creatures: &Vec<Creature>, max_new_creatures: u32) -> Vec<Creature> {
    let chunk_size: usize = 1000;

    let max_new_per_chunk = match creatures.len() as u32 / chunk_size as u32 {
        x if x == 0 => 2,
        x => max_new_creatures / x,
    };

    creatures.chunks(chunk_size)
        .collect::<Vec<&[Creature]>>()  // have to turn into a type Rayon can use
        .into_par_iter()
        .map(|cr_vec| {
                // Pre-allocate vector of expected length
                let mut chunk_offspring = Vec::with_capacity(chunk_size / 2);
                for chunk in cr_vec.chunks(2) {
                    if let [cr1, cr2] = chunk {
                        chunk_offspring.push(cr1 + cr2);
                    }
                    if chunk_offspring.len() as u32 >= max_new_per_chunk {
                        break;
                    }
                }
                chunk_offspring
            })
        .flatten()
        .collect()
}

fn calc_error_sum(creature: &Creature,
                  data_points: &Vec<HashMap<String, f32>>,
                  target_param: &str) -> f32 {
    let mut total: f32 = 0.0;
    for point in data_points {
        let calc = creature.rust_calculate(point);
        let diff = calc - point.get(target_param)
                               .expect("Data point missing target_param");
        total += diff.powi(2);
    }
    total / (data_points.len() as f32)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use itertools::izip;

    #[test]
    fn basic_evolution() {
        let target = "target_param";
        let data = vec![
            HashMap::from([("target_param".to_string(), 5.2), ("p2".to_string(), 7.8), ("p3".to_string(), 8.3)]),
            HashMap::from([("target_param".to_string(), 6.0), ("p2".to_string(), 4.4), ("p3".to_string(), 8.1)]),
            HashMap::from([("target_param".to_string(), 7.1), ("p2".to_string(), 3.9), ("p3".to_string(), 9.5)]),
            HashMap::from([("target_param".to_string(), 8.6), ("p2".to_string(), 2.7), ("p3".to_string(), 11.6)]),
            HashMap::from([("target_param".to_string(), 9.4), ("p2".to_string(), -2.6), ("p3".to_string(), 13.0)]),
        ];

        let evo = Evolution::new(target.into(), &data, 10000, 10, 3, true);
        assert!(evo.num_creatures == 10000);
    }

    #[test]
    fn parabola() {
        let parabola_data = vec![
            HashMap::from([("x".to_string(), -20.0), ("y".to_string(), 195.0967073301952)]),
            HashMap::from([("x".to_string(), -19.0), ("y".to_string(), 205.88669941695193)]),
            HashMap::from([("x".to_string(), -18.0), ("y".to_string(), 75.05183418690936)]),
            HashMap::from([("x".to_string(), -17.0), ("y".to_string(), 153.31304897814132)]),
            HashMap::from([("x".to_string(), -16.0), ("y".to_string(), 180.72678834266526)]),
            HashMap::from([("x".to_string(), -15.0), ("y".to_string(), 81.73490536370575)]),
            HashMap::from([("x".to_string(), -14.0), ("y".to_string(), 76.98269474497451)]),
            HashMap::from([("x".to_string(), -13.0), ("y".to_string(), 106.65404246488129)]),
            HashMap::from([("x".to_string(), -12.0), ("y".to_string(), 101.81854634039516)]),
            HashMap::from([("x".to_string(), -11.0), ("y".to_string(), 32.735790537057994)]),
            HashMap::from([("x".to_string(), -10.0), ("y".to_string(), 3.5140689599924273)]),
            HashMap::from([("x".to_string(), -9.0), ("y".to_string(), 21.979234525796137)]),
            HashMap::from([("x".to_string(), -8.0), ("y".to_string(), 2.101943660864327)]),
            HashMap::from([("x".to_string(), -7.0), ("y".to_string(), 4.083877304799986)]),
            HashMap::from([("x".to_string(), -6.0), ("y".to_string(), 0.12110473958116565)]),
            HashMap::from([("x".to_string(), -5.0), ("y".to_string(), 16.57223235311977)]),
            HashMap::from([("x".to_string(), -4.0), ("y".to_string(), 0.14511553873582717)]),
            HashMap::from([("x".to_string(), -3.0), ("y".to_string(), 2.510511396206416)]),
            HashMap::from([("x".to_string(), -2.0), ("y".to_string(), 56.587670650914006)]),
            HashMap::from([("x".to_string(), -1.0), ("y".to_string(), 4.880296227847032)]),
            HashMap::from([("x".to_string(), 0.0), ("y".to_string(), 15.393806879686704)]),
            HashMap::from([("x".to_string(), 1.0), ("y".to_string(), 19.980723972406757)]),
            HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 46.44040802736543)]),
            HashMap::from([("x".to_string(), 3.0), ("y".to_string(), 76.32570640372656)]),
            HashMap::from([("x".to_string(), 4.0), ("y".to_string(), 28.344936970432833)]),
            HashMap::from([("x".to_string(), 5.0), ("y".to_string(), 107.80487596755955)]),
            HashMap::from([("x".to_string(), 6.0), ("y".to_string(), 90.52490037859376)]),
            HashMap::from([("x".to_string(), 7.0), ("y".to_string(), 157.59858818802704)]),
            HashMap::from([("x".to_string(), 8.0), ("y".to_string(), 143.33624805335427)]),
            HashMap::from([("x".to_string(), 9.0), ("y".to_string(), 145.24993288695646)]),
            HashMap::from([("x".to_string(), 10.0), ("y".to_string(), 260.1807578980633)]),
            HashMap::from([("x".to_string(), 11.0), ("y".to_string(), 185.66458035427738)]),
            HashMap::from([("x".to_string(), 12.0), ("y".to_string(), 399.47143038541725)]),
            HashMap::from([("x".to_string(), 13.0), ("y".to_string(), 461.637154269764)]),
            HashMap::from([("x".to_string(), 14.0), ("y".to_string(), 224.52939759007862)]),
            HashMap::from([("x".to_string(), 15.0), ("y".to_string(), 435.1803248133029)]),
            HashMap::from([("x".to_string(), 16.0), ("y".to_string(), 624.3116876259189)]),
            HashMap::from([("x".to_string(), 17.0), ("y".to_string(), 453.5298507352485)]),
            HashMap::from([("x".to_string(), 18.0), ("y".to_string(), 396.33513809585935)]),
            HashMap::from([("x".to_string(), 19.0), ("y".to_string(), 415.8142609595538)]),
            HashMap::from([("x".to_string(), 20.0), ("y".to_string(), 758.0144333664495)]),
        ];
        let target = String::from("y");
        let model = Evolution::new(target, &parabola_data, 5000, 7, 2, true);

        for creature in &model.best_creatures {
            assert!(creature.num_layers() <= 2);  // Light check on max_layers
        }

        let output_data: Vec<f32> = (-20..=20)
            .map(|x| model.predict_point(HashMap::from([("x".to_string(), x as f32)])))
            .collect();
        let mut output_string = String::from("x,y,\n");
        for (x, y) in izip!(-20..=20, output_data) {
            output_string += &format!("{},{},\n", x, y);
        }
        fs::write("parabola_output.csv", output_string).expect("Unable to write to file");
    }

}
