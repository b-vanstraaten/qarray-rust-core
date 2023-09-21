use std::isize;

use itertools;
use itertools::{Itertools, repeat_n};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2, s};
use rayon::prelude::*;
use osqp::{CscMatrix, Problem, Settings};
use crate::charge_configurations::closed_charge_configurations_brute_force;


fn init_osqp_problem_open<'a>(v_g: ArrayView<f64, Ix1>, c_gd: ArrayView<'a, f64, Ix2>,
                              c_dd_inv: ArrayView<'a, f64, Ix2>) -> Problem {


    let dim = c_dd_inv.shape()[0];
    let P = CscMatrix::from(c_dd_inv.rows()).into_upper_tri();

    let q_array = -c_dd_inv.dot(&c_gd.dot(&v_g));
    let q = q_array.as_slice().unwrap();

    let l_array = Array1::<f64>::zeros(dim);
    let l = l_array.as_slice().unwrap();

    let u_array = Array1::<f64>::from_elem(dim, 100.);
    let u = u_array.as_slice().unwrap();
    let A = {
        let identity = Array2::<f64>::eye(dim);
        CscMatrix::from(identity.rows())
    };

    // % TODO find out what alpha does and if it is important
    let settings = Settings::default()
        .alpha(1.0)
        .verbose(false)
        .polish(false);
    Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem")
}


fn init_osqp_problem_closed<'a>(v_g: ArrayView<f64, Ix1>, c_gd: ArrayView<'a, f64, Ix2>,
                              c_dd_inv: ArrayView<'a, f64, Ix2>, n_charge: u64) -> Problem {

    let dim = c_dd_inv.shape()[0];
    let P = CscMatrix::from(c_dd_inv.rows()).into_upper_tri();

    let q_array = -c_dd_inv.dot(&c_gd.dot(&v_g));
    let q = q_array.as_slice().unwrap();

    let l_array = ndarray::concatenate(
        Axis(0), &[
            Array1::<f64>::from_elem(1, n_charge as f64).view(),
            Array1::<f64>::zeros(dim).view()]
    ).unwrap();
    let l = l_array.as_slice().unwrap();

    let u_array = ndarray::concatenate(
        Axis(0), &[
            Array1::<f64>::from_elem(1, n_charge as f64).view(),
            Array1::<f64>::from_elem(dim, 100.).view()]
    ).unwrap();
    let u = u_array.as_slice().unwrap();

    let A = {
        let ones = Array2::<f64>::ones((1, dim));
        let identity = Array2::<f64>::eye(dim);
        let A = ndarray::concatenate(
            Axis(0), &[
                ones.view(),
                identity.view()]
        ).unwrap();
        CscMatrix::from(A.rows())
    };

    // % TODO find out what alpha does and if it is important
    let settings = Settings::default()
        .alpha(1.0)
        .verbose(false)
        .polish(false);
    Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem")
}


pub fn ground_state_open_1d<'a>(
    v_g: ArrayView<'a, f64, Ix2>,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
    threshold: f64,
) -> Array<f64, Ix2> {


    let n = v_g.shape()[0];
    let m = c_gd.shape()[0];
    let mut results_array = Array2::zeros((n, m));

    let mut rows: Vec<_> = results_array.axis_iter_mut(ndarray::Axis(0)).collect();

    rows.par_iter_mut().enumerate().for_each(|(j, result_row)| {
        let v_g_row = v_g.slice(s![j, ..]);
        let n_charge = ground_state_open_0d(v_g_row, c_gd, c_dd_inv, threshold);
        result_row.assign(&n_charge);
    });

    results_array
}

pub fn ground_state_closed_1d<'a>(
    v_g: ArrayView<'a, f64, Ix2>,
    n_charge: u64,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
) -> Array<f64, Ix2> {
    let n = v_g.shape()[0];
    let m = c_gd.shape()[0];
    let mut results_array = Array2::zeros((n, m));

    let mut rows: Vec<_> = results_array.axis_iter_mut(ndarray::Axis(0)).collect();

    rows.par_iter_mut().enumerate().for_each(|(j, result_row)| {
        let v_g_row = v_g.slice(s![j, ..]);
        let n_charge = ground_state_closed_0d(v_g_row, n_charge, c_gd, c_dd_inv);
        result_row.assign(&n_charge);
    });

    results_array
}


pub fn ground_state_open_0d<'a>(v_g: ArrayView<f64, Ix1>, c_gd: ArrayView<'a, f64, Ix2>, c_dd_inv: ArrayView<'a, f64, Ix2>, threshold: f64) -> Array<f64, Ix1> {
    let mut problem = init_osqp_problem_open(v_g, c_gd, c_dd_inv);
    let result = problem.solve();

    // compute the continuous part of the ground state
    let mut n_continuous = Array1::<f64>::from(result
        .x()
        .expect("failed to solve problem")
        .to_owned());

    // clip the continuous part to be positive, as we have turned off polishing in the solver
    n_continuous.mapv_inplace(|x| x.max(0.0));

    // determine if rounding is required
    let requires_floor_ceil = n_continuous.iter().any(|x| (x.fract() - 0.5).abs() < threshold / 2.);
    if !requires_floor_ceil {
        // round every element to the nearest integer
        return n_continuous.mapv(|x| f64::round(x));
    } else {
        let floor_ceil_funcs = [
            |x| f64::floor(x) - x,
            |x| f64::ceil(x) - x
        ];

        let mut min_u = f64::MAX; // Initialize with a high value
        let mut min_delta = Array1::<f64>::zeros(n_continuous.len());
        let mut delta = Array1::<f64>::zeros(n_continuous.len());

        // floor_ceil_args are the indices that need to be checked whether they need to be rounded up or down not to the nearest integer
        let floor_ceil_args: Array1<usize> = (0..n_continuous.len())
            .filter(|i| (n_continuous[*i].fract() - 0.5).abs() < threshold / 2.)
            .collect();

        // round args are the indices not in floor_ceil_args, which can just be normally rounded to the nearest integer
        let round_args: Array1<usize> = (0..n_continuous.len())
            .filter(|i| (n_continuous[i.to_owned()].fract() - 0.5).abs() >= threshold / 2.)
            .collect();

        for ops in repeat_n(floor_ceil_funcs, floor_ceil_args.len()).multi_cartesian_product()
        {
            // Calculate delta based on ops and round_args
            for (j, op) in floor_ceil_args.iter().zip(&ops) {
                delta[*j] = op(n_continuous[*j]);
            }

            // Calculate delta based and round_args
            for j in &round_args {
                let x = n_continuous[*j];
                delta[*j] = f64::round(x) - x
            }

            // Calculate u
            let u = delta.dot(&c_dd_inv.dot(&delta));

            // Check if this combination has a lower u value
            if u < min_u {
                min_u = u;
                min_delta.assign(&delta); // Store the delta for the minimum u
            }
        }
        n_continuous + min_delta
    }
}

pub fn ground_state_closed_0d<'a>(v_g: ArrayView<f64, Ix1>, n_charge: u64,
                                  c_gd: ArrayView<'a, f64, Ix2>, c_dd_inv: ArrayView<'a, f64, Ix2>) -> Array<f64, Ix1> {

    let n_dot = c_dd_inv.shape()[0];
    let mut problem = init_osqp_problem_closed(v_g, c_gd, c_dd_inv, n_charge);
    let result = problem.solve();

    // compute the continuous part of the ground state
    let mut n_continuous = Array1::<f64>::from(result
        .x()
        .expect("failed to solve problem")
        .to_owned());

    // clip the continuous part to be positive, as we have turned off polishing in the solver
    n_continuous.mapv_inplace(|x| x.max(0.0).min(n_charge as f64));
    let floor_list = n_continuous
        .mapv(|x| f64::floor(x))
        .mapv(|x| x as u64);

    let n_list = closed_charge_configurations_brute_force(n_charge, n_dot as u64, floor_list.view());

    // type conversion from i64 to f64
    let n_list = n_list.mapv(|x| x as f64);

    let n_min = n_list
        .outer_iter()
        .map(|x| x.to_owned() - &n_continuous)
        .map(|x| x.dot(&c_dd_inv.dot(&x)))
        .enumerate()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
        .map(|(idx, _)| n_list.index_axis(Axis(0), idx))
        .unwrap()
        .to_owned();

    return n_min
}