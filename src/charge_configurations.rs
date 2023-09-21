use ndarray::{Array1, Array2};

use itertools;

use cached::proc_macro::cached;

#[cached]
pub fn closed_charge_configurations_brute_force(
    n_charge: u64,
    n_dot: u64,
    floor_values: Array1<u64>,
) -> Array2<u64> {
    // Create a clone of floor_values with a 'static lifetime

    let floor_sum: u64 = floor_values.sum();

    if floor_sum > n_charge {
        return Array2::default((0, n_dot as usize)); // Return an empty array
    }

    if floor_values.iter().map(|&x| x + 1).sum::<u64>() < n_charge {
        return Array2::default((0, n_dot as usize)); // Return an empty array
    }

    let num_combinations = 2u64.pow(floor_values.len() as u32);
    let result = (0..num_combinations)
        .map(|i| {
            let binary_str = format!("{:01$b}", i, floor_values.len());
            let binary_values: Vec<u64> = binary_str
                .chars()
                .map(|c| c.to_digit(2).unwrap() as u64)
                .collect();
            let sum: u64 = binary_values.iter().sum();
            if sum == n_charge - floor_sum { Some(binary_values) } else { None }
        })
        .filter_map(|x| x)
        .map(|mut comb| {
            for (i, val) in floor_values.iter().enumerate() {
                comb[i] += val;
            }
            comb
        })
        .flatten()
        .collect::<Vec<u64>>();
    Array2::from_shape_vec((result.len() / n_dot as usize, n_dot as usize), result).unwrap()
}