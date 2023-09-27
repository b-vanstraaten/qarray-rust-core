use std::isize;

use itertools;
use itertools::{Itertools, repeat_n};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2, s};
use rayon::prelude::*;
use osqp::{CscMatrix, Problem, Settings};
use crate::charge_configurations::closed_charge_configurations;


pub fn ground_state_closed_1d<'a>(
    v_g: ArrayView<'a, f64, Ix2>,
    n_charge: u64,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
    threshold: f64,
) -> Array<f64, Ix2> {
    let n = v_g.shape()[0];
    let m = c_gd.shape()[0];
    let mut results_array = Array2::zeros((n, m));

    let mut rows: Vec<_> = results_array.axis_iter_mut(ndarray::Axis(0)).collect();

    rows.par_iter_mut().enumerate().for_each(|(j, result_row)| {
        let v_g_row = v_g.slice(s![j, ..]);
        let n_charge = ground_state_closed_0d(v_g_row, n_charge, c_gd, c_dd, c_dd_inv, threshold);
        result_row.assign(&n_charge);
    });
    results_array
}

pub fn ground_state_closed_0d<'a>(v_g: ArrayView<f64, Ix1>, n_charge: u64,
                                  c_gd: ArrayView<'a, f64, Ix2>, c_dd: ArrayView<'a, f64, Ix2>, c_dd_inv: ArrayView<'a, f64, Ix2>, threshold: f64) -> Array<f64, Ix1> {
    let analytical_solution = analytical_solution(c_gd, c_dd, v_g, n_charge);
    if analytical_solution.iter().all(|x| x >= &0.0 && x <= &(n_charge as f64)) {
        // the analytical solution is a valid charge configuration therefore we don't need to solve
        // the constrained optimization problem
        return compute_argmin_closed(analytical_solution, c_dd_inv, c_gd, v_g, n_charge, threshold);
    } else {
        // the analytical solution is not a valid charge configuration
        // therefore we need to solve the constrained optimization problem
        let mut problem = init_osqp_problem_closed(v_g, c_gd, c_dd_inv, n_charge);
        let result = problem.solve();

        // compute the continuous part of the ground state
        let n_continuous = Array1::<f64>::from(result
            .x()
            .expect("failed to solve problem")
            .to_owned());

        return compute_argmin_closed(n_continuous, c_dd_inv, c_gd, v_g, n_charge, threshold);
    }
}

fn analytical_solution(c_gd: ArrayView<f64, Ix2>, c_dd: ArrayView<f64, Ix2>, v_g: ArrayView<f64, Ix1>, n_charge: u64) -> Array1<f64> {
    let n_continuous = c_gd.dot(&v_g);
    let isolation_correction = (n_charge as f64 - n_continuous.sum()) * (c_dd.sum_axis(Axis(0)) / c_dd.sum());
    return n_continuous + isolation_correction;
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
            Array1::<f64>::zeros(dim).view()],
    ).unwrap();
    let l = l_array.as_slice().unwrap();

    let u_array = ndarray::concatenate(
        Axis(0), &[
            Array1::<f64>::from_elem(1, n_charge as f64).view(),
            Array1::<f64>::from_elem(dim, 100.).view()],
    ).unwrap();
    let u = u_array.as_slice().unwrap();

    let A = {
        let ones = Array2::<f64>::ones((1, dim));
        let identity = Array2::<f64>::eye(dim);
        let A = ndarray::concatenate(
            Axis(0), &[
                ones.view(),
                identity.view()],
        ).unwrap();
        CscMatrix::from(A.rows())
    };

    let settings = Settings::default()
        .alpha(1.0)
        .verbose(false)
        .polish(false);
    Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem")
}


fn compute_argmin_closed(n_continuous: Array1<f64>, c_dd_inv: ArrayView<f64, Ix2>, c_gd: ArrayView<f64, Ix2>, vg: ArrayView<f64, Ix1>,n_charge: u64, threshold: f64) -> Array1<f64> {
    let n_list = closed_charge_configurations(n_continuous, n_charge, threshold);

    // type conversion from i64 to f64
    let n_list = n_list.mapv(|x| x as f64);

    let n_min = n_list
        .outer_iter()
        .map(|x| x.to_owned() - &c_gd.dot(&vg))
        .map(|x| x.dot(&c_dd_inv.dot(&x)))
        .enumerate()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
        .map(|(idx, _)| n_list.index_axis(Axis(0), idx))
        .unwrap()
        .to_owned();

    return n_min;
}
