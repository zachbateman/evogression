mod creature;
mod standardize;
mod evolution;
use std::collections::HashMap;

use pyo3::prelude::*;



#[pyfunction]
fn standardize_data(data: Vec<HashMap<String, f32>>) -> Vec<HashMap<String, f32>> {
    let standardizer = standardize::Standardizer::new(&data);
    standardizer.standardized_values(&data)
}


#[pyfunction]
fn run_evolution(target: String,
                 data: Vec<HashMap<String, f32>>,
                 num_creatures: u32,
                 num_cycles: u16,
                 max_layers: u8,
                ) -> evolution::Evolution {
    evolution::Evolution::new(target, &data, num_creatures, num_cycles, max_layers)
}


#[pymodule]
fn rust_evogression(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(standardize_data, m)?)?;
    m.add_function(wrap_pyfunction!(run_evolution, m)?)?;
    m.add_class::<creature::Creature>()?;
    m.add_class::<evolution::Evolution>()?;
    Ok(())
}
