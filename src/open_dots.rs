// Computes the ground state for open dots.

use ndarray::{s, Array, Array1, Array2, ArrayView, Ix1, Ix2};
use osqp::{CscMatrix, Problem, Settings};
use rayon::prelude::*;

use crate::charge_configurations::open_charge_configurations;
use crate::helper_functions::{hard_argmin, soft_argmin};

#[allow(non_snake_case)]
pub fn ground_state_open_1d<'a>(
    v_g: ArrayView<'a, f64, Ix2>,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
    threshold: f64,
    polish: bool,
    T: f64,
) -> Array<f64, Ix2> {
    let n = v_g.shape()[0];
    let m = c_gd.shape()[0];
    let mut results_array = Array2::zeros((n, m));

    let mut rows: Vec<_> = results_array.axis_iter_mut(ndarray::Axis(0)).collect();

    rows.par_iter_mut().enumerate().for_each(|(j, result_row)| {
        let v_g_row = v_g.slice(s![j, ..]);
        let n_charge = ground_state_open_0d(v_g_row, c_gd, c_dd_inv, threshold, polish, T);
        result_row.assign(&n_charge);
    });
    return results_array;
}

#[allow(non_snake_case)]
fn ground_state_open_0d<'a>(
    v_g: ArrayView<f64, Ix1>,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
    threshold: f64,
    polish: bool,
    T: f64,
) -> Array<f64, Ix1> {
    let analytical_solution = analytical_solution(c_gd, v_g);

    if analytical_solution.iter().all(|x| x >= &0.0) {
        return compute_argmin_open(analytical_solution, c_dd_inv, c_gd, v_g, threshold, T);
    } else {
        let mut problem = init_osqp_problem_open(v_g, c_gd, c_dd_inv, polish);
        let result = problem.solve();

        // compute the continuous part of the ground state
        let mut n_continuous =
            Array1::<f64>::from(result.x().expect("failed to solve problem").to_owned());

        // clip the continuous part to be positive, as we have turned off polishing in the solver
        n_continuous.mapv_inplace(|x| x.max(0.0));
        return compute_argmin_open(n_continuous, c_dd_inv, c_gd, v_g, threshold, T);
    }
}

fn analytical_solution(c_gd: ArrayView<f64, Ix2>, v_g: ArrayView<f64, Ix1>) -> Array1<f64> {
    return c_gd.dot(&v_g);
}

#[allow(non_snake_case)]
fn init_osqp_problem_open<'a>(
    v_g: ArrayView<f64, Ix1>,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
    polish: bool,
) -> Problem {
    let dim = c_dd_inv.shape()[0];
    let P = CscMatrix::from(c_dd_inv.rows()).into_upper_tri();

    let q_array = -c_dd_inv.dot(&c_gd.dot(&v_g));
    let q = q_array.as_slice().expect("failed to get slice of q");

    let l_array = Array1::<f64>::zeros(dim);
    let l = l_array.as_slice().expect("failed to get slice of l");

    let u_array = Array1::<f64>::from_elem(dim, f64::MAX);
    let u = u_array.as_slice().expect("failed to get slice of u");
    let A = {
        let identity = Array2::<f64>::eye(dim);
        CscMatrix::from(identity.rows())
    };

    let settings = Settings::default().alpha(1.0).verbose(false).polish(polish);
    return Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");
}

#[allow(non_snake_case)]
fn compute_argmin_open(
    n_continuous: Array1<f64>,
    c_dd_inv: ArrayView<f64, Ix2>,
    c_gd: ArrayView<f64, Ix2>,
    v_g: ArrayView<f64, Ix1>,
    threshold: f64,
    T: f64,
) -> Array1<f64> {
    let vg_dash = c_gd.dot(&v_g);

    let n_list = open_charge_configurations(n_continuous, threshold);

    match T > 0.0 {
        false => hard_argmin(n_list, c_dd_inv, vg_dash),
        true => soft_argmin(n_list, c_dd_inv, vg_dash, T),
    }
}
