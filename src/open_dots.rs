use std::isize;

use itertools;
use itertools::{Itertools, repeat_n};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2, s};
use rayon::prelude::*;
use osqp::{CscMatrix, Problem, Settings};



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
    return results_array
}

fn ground_state_open_0d<'a>(v_g: ArrayView<f64, Ix1>, c_gd: ArrayView<'a, f64, Ix2>, c_dd_inv: ArrayView<'a, f64, Ix2>, threshold: f64) -> Array<f64, Ix1> {

    let analytical_solution = analytical_solution(c_gd, v_g);

    if analytical_solution.iter().all(|x| x >= &0.0) {
        return compute_argmin_open(analytical_solution, c_dd_inv, c_gd, v_g, threshold);
    } else {

        let mut problem = init_osqp_problem_open(v_g, c_gd, c_dd_inv);
        let result = problem.solve();

        // compute the continuous part of the ground state
        let mut n_continuous = Array1::<f64>::from(result
            .x()
            .expect("failed to solve problem")
            .to_owned());

        // clip the continuous part to be positive, as we have turned off polishing in the solver
        n_continuous.mapv_inplace(|x| x.max(0.0));
        return compute_argmin_open(n_continuous, c_dd_inv, c_gd, v_g, threshold)

    }


}

fn analytical_solution(c_gd: ArrayView<f64, Ix2>, v_g: ArrayView<f64, Ix1>) -> Array1<f64> {
    return c_gd.dot(&v_g)
}

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


    let settings = Settings::default()
        .alpha(1.0)
        .adaptive_rho(true)
        .verbose(false)
        .polish(true);
    return Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem")
}

fn compute_argmin_open(n_continuous: Array1<f64>, c_dd_inv: ArrayView<f64, Ix2>, c_gd: ArrayView<f64, Ix2>, v_g: ArrayView<f64, Ix1>, threshold: f64) -> Array1<f64> {
    let requires_floor_ceil = n_continuous.iter().any(|x| (x.fract() - 0.5).abs() < threshold / 2.);
    if !requires_floor_ceil {
        // round every element to the nearest integer
        return n_continuous.mapv(|x| f64::round(x));
    } else {

        let floor_ceil_funcs = [f64::floor, f64::ceil];

        // floor_ceil_args are the indices that need to be checked whether they need to be rounded up or down not to the nearest integer
        let floor_ceil_args: Array1<usize> = (0..n_continuous.len())
            .filter(|i| (n_continuous[*i].fract() - 0.5).abs() < threshold / 2.)
            .collect();

        // round args are the indices not in floor_ceil_args, which can just be normally rounded to the nearest integer
        let round_args: Array1<usize> = (0..n_continuous.len())
            .filter(|i| (n_continuous[i.to_owned()].fract() - 0.5).abs() >= threshold / 2.)
            .collect();

        let vg_dash = c_gd.dot(&v_g);

        let (min_u, min_n) = repeat_n(&floor_ceil_funcs, floor_ceil_args.len())
            .multi_cartesian_product()
            .map(|ops| {

                let mut delta = Array1::<f64>::zeros(n_continuous.len());

                // Calculate delta based on ops and round_args
                for (i, op) in floor_ceil_args.iter().zip(&ops) {
                    let j = i.to_owned();
                    delta[j] = op(n_continuous[j]) - vg_dash[j];
                }

                // Calculate delta based and round_args
                for i in &round_args {
                    let j = i.to_owned();
                    delta[j] = f64::round(n_continuous[j]) - vg_dash[j];
                }

                // Calculate u
                let u = delta.dot(&c_dd_inv.dot(&delta));
                (u, delta.to_owned())
            }).min_by(|(u1, _), (u2, _)| u1.partial_cmp(u2).unwrap())
            .map(|(u, delta)| (u, delta + vg_dash))
            .unwrap();

        return min_n
    }
}