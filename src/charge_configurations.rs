use std::usize;

use itertools;

use ndarray::{Array2, ArrayView, Ix1};
use rayon::prelude::*;


pub fn closed_charge_configurations_brute_force(
    n_charge: i64,
    n_dot: i64,
    floor_values: ArrayView<i64, Ix1>,
) -> Array2<i64> {
    let floor_sum: i64 = floor_values.sum();

    if floor_sum > n_charge {
        return Array2::default((0, n_dot as usize)); // Return an empty array
    }

    if floor_values.iter().map(|&x| x + 1).sum::<i64>() < n_charge {
        return Array2::default((0, n_dot as usize)); // Return an empty array
    }

    let num_combinations = 2i64.pow(floor_values.len() as u32);
    let result = (0..num_combinations)
        .map(|i| {
            let binary_str = format!("{:01$b}", i, floor_values.len());
            let binary_values: Vec<i64> = binary_str
                .chars()
                .map(|c| c.to_digit(2).unwrap() as i64)
                .collect();
            let sum: i64 = binary_values.iter().sum();
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
        .collect::<Vec<i64>>();
    Array2::from_shape_vec((result.len() / n_dot as usize, n_dot as usize), result).unwrap()
}