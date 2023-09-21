use std::usize;

use itertools;

use ndarray::{Array2, ArrayView, Ix1};
use rayon::prelude::*;


pub fn closed_charge_configurations_brute_force(
    n_charge: isize,
    n_dot: isize,
    floor_values: ArrayView<isize, Ix1>,
) -> Array2<isize> {
    let floor_sum: isize = floor_values.sum();

    if floor_sum > n_charge {
        return Array2::default((0, n_dot as usize)); // Return an empty array
    }

    if floor_values.iter().map(|&x| x + 1).sum::<isize>() < n_charge {
        return Array2::default((0, n_dot as usize)); // Return an empty array
    }

    let num_combinations = 2isize.pow(floor_values.len() as u32);
    let result = (0..num_combinations)
        .map(|i| {
            let binary_str = format!("{:01$b}", i, floor_values.len());
            let binary_values: Vec<isize> = binary_str
                .chars()
                .map(|c| c.to_digit(2).unwrap() as isize)
                .collect();
            let sum: isize = binary_values.iter().sum();
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
        .collect::<Vec<isize>>();
    Array2::from_shape_vec((result.len() / n_dot as usize, n_dot as usize), result).unwrap()
}