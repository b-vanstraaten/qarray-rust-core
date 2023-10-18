use cached::proc_macro::cached;
use ndarray::{s, Array, Array1, Array2, Axis, Ix2};

pub fn open_charge_configurations(n_continuous: Array1<f64>, threshold: f64) -> Array<f64, Ix2> {
    if threshold >= 1.0 {
        let floor_values = n_continuous.mapv(|x| x.floor() as u64);
        return _open_charge_configurations(floor_values).mapv(|x| x as f64);
    }

    let (floor_ceil_args, round_args): (Vec<usize>, Vec<usize>) = (0..n_continuous.len())
        .partition(|&i| (n_continuous[i].fract() - 0.5).abs() < threshold / 2.0);

    if floor_ceil_args.is_empty() {
        // All the continuous values are integers, so we can just return the floor values
        return n_continuous.mapv(|x| x.round()).insert_axis(Axis(0));
    }

    let mut charge_configurations = Array2::zeros((1 << floor_ceil_args.len(), n_continuous.len()));
    let floor_values: Array1<u64> = floor_ceil_args
        .iter()
        .map(|&i| n_continuous[i].floor() as u64)
        .collect();

    let rounded_values: Array1<u64> = round_args
        .iter()
        .map(|&i| n_continuous[i].round() as u64)
        .collect();

    let floor_charge_configurations = _open_charge_configurations(floor_values);

    for (i, &j) in floor_ceil_args.iter().enumerate() {
        // Ensure that the shape of floor_charge_configurations and rounded_values is compatible for broadcasting.
        // The dimensions should either match or one of them should be 1.
        // For example, if floor_charge_configurations.shape() is (N, M) and rounded_values.len() is M, it should work.
        charge_configurations
            .slice_mut(s![.., j])
            .assign(&floor_charge_configurations.slice(s![.., i]));
    }

    for (i, &j) in round_args.iter().enumerate() {
        charge_configurations
            .slice_mut(s![.., j])
            .fill(rounded_values[i]);
    }
    return charge_configurations.mapv(|x| x as f64);
}

pub fn closed_charge_configurations(
    n_continuous: Array1<f64>,
    n_charge: u64,
    threshold: f64,
) -> Array<f64, Ix2> {
    // if the threshold is greater than 1.0 we can return all configurations without worrying about rounding
    if threshold >= 1.0 {
        let floor_values = n_continuous.mapv(|x| x.floor() as u64);
        return _closed_charge_configurations(floor_values, n_charge).mapv(|x| x as f64);
    }

    let (floor_ceil_args, round_args): (Vec<usize>, Vec<usize>) = (0..n_continuous.len())
        .partition(|&i| (n_continuous[i].fract() - 0.5).abs() < threshold / 2.0);

    if floor_ceil_args.is_empty() {
        let rounded_values = n_continuous.mapv(|x| x.round());
        if rounded_values.map(|x| x.to_owned() as u64).sum() == n_charge {
            return rounded_values.insert_axis(Axis(0));
        }

        // All the continuous values are integers, so we can just return the floor values
        let floor_values = n_continuous.mapv(|x| x.floor() as u64);
        return _closed_charge_configurations(floor_values, n_charge).mapv(|x| x as f64);
    }

    let floor_values: Array1<u64> = floor_ceil_args
        .iter()
        .map(|&i| n_continuous[i].floor() as u64)
        .collect();

    let rounded_values: Array1<u64> = round_args
        .iter()
        .map(|&i| n_continuous[i].round() as u64)
        .collect();

    let floor_charge_configurations =
        _closed_charge_configurations(floor_values, n_charge - rounded_values.sum());

    // a recursively calling closed_charge_configurations with double the threshold if the floor_charge_configurations is empty
    if floor_charge_configurations.is_empty() {
        return closed_charge_configurations(n_continuous, n_charge, f64::min(2. * threshold, 1.0));
    }

    let n_dot = n_continuous.len();
    let n_combinations = floor_charge_configurations.shape()[0];
    let mut charge_configurations = Array2::zeros((n_combinations, n_dot));

    for (i, &j) in floor_ceil_args.iter().enumerate() {
        charge_configurations
            .slice_mut(s![.., j])
            .assign(&floor_charge_configurations.slice(s![.., i]));
    }

    for (i, &j) in round_args.iter().enumerate() {
        charge_configurations
            .slice_mut(s![.., j])
            .fill(rounded_values[i]);
    }
    return charge_configurations.mapv(|x| x as f64);
}

#[cached(size = 1024)]
fn _open_charge_configurations(floor_values: Array1<u64>) -> Array2<u64> {
    let n_dot = floor_values.len() as u64;
    let num_combinations = 1u64 << n_dot;
    let mut result = Vec::with_capacity((num_combinations * n_dot) as usize);

    for i in 0..num_combinations {
        let mut configuration = floor_values.to_vec(); // Start with a clone of floor_values
        for j in 0..floor_values.len() {
            let bit = (i >> j) & 1;
            configuration[j] += bit;
        }
        result.extend_from_slice(&configuration);
    }
    let rows = result.len() / n_dot as usize;
    Array2::from_shape_vec((rows, n_dot as usize), result).expect("Failed to reshape array")
}

#[cached(size = 1024)]
fn _closed_charge_configurations(floor_values: Array1<u64>, n_charge: u64) -> Array2<u64> {
    let n_dot = floor_values.len() as u64;
    let floor_sum: u64 = floor_values.sum();

    if floor_sum > n_charge || floor_sum + n_dot < n_charge {
        return Array2::default((0, n_dot as usize)); // Return an empty array
    }

    let num_combinations = 1u64 << n_dot;
    let mut result = Vec::new();
    // precompute this value to avoid recomputing it in the loop
    let n_charge_floor_sum_diff = n_charge - floor_sum;

    for i in 0..num_combinations {
        let mut sum = 0;
        let mut configuration = floor_values.to_vec(); // Start with a clone of floor_values
        for j in 0..floor_values.len() {
            let bit = (i >> j) & 1;
            sum += bit;
            configuration[j] += bit;
        }
        if sum == n_charge_floor_sum_diff {
            result.extend_from_slice(configuration.as_slice());
        }
    }
    let rows = result.len() / n_dot as usize;
    return Array2::from_shape_vec((rows, n_dot as usize), result)
        .expect("Failed to reshape array");
}
