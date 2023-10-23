// Computes the ground state for closed dots.

use ndarray::{s, Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2};
use osqp::{CscMatrix, Problem, Settings};
use rayon::prelude::*;

use crate::charge_configurations::closed_charge_configurations;
use crate::helper_functions::{hard_argmin, soft_argmin};

#[allow(non_snake_case)]
pub fn ground_state_closed_1d<'a>(
    v_g: ArrayView<'a, f64, Ix2>,
    n_charge: u64,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd: ArrayView<'a, f64, Ix2>,
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
        let n_charge = ground_state_closed_0d(
            v_g_row, n_charge, c_gd, c_dd, c_dd_inv, threshold, polish, T,
        );
        result_row.assign(&n_charge);
    });
    results_array
}

#[allow(non_snake_case)]
pub fn ground_state_closed_0d<'a>(
    v_g: ArrayView<f64, Ix1>,
    n_charge: u64,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
    threshold: f64,
    polish: bool,
    T: f64,
) -> Array<f64, Ix1> {
    let analytical_solution = analytical_solution(c_gd, c_dd, v_g, n_charge);
    if analytical_solution
        .iter()
        .all(|x| x >= &0.0 && x <= &(n_charge as f64))
    {
        // the analytical solution is a valid charge configuration therefore we don't need to solve
        // the constrained optimization problem
        return compute_argmin_closed(
            analytical_solution,
            c_dd_inv,
            c_gd,
            v_g,
            n_charge,
            threshold,
            T,
        );
    } else {
        // the analytical solution is not a valid charge configuration
        // therefore we need to solve the constrained optimization problem
        let mut problem = init_osqp_problem_closed(v_g, c_gd, c_dd_inv, n_charge, polish);
        let result = problem.solve();

        // compute the continuous part of the ground state
        let n_continuous =
            Array1::<f64>::from(result.x().expect("failed to solve problem").to_owned());

        return compute_argmin_closed(n_continuous, c_dd_inv, c_gd, v_g, n_charge, threshold, T);
    }
}

fn analytical_solution(
    c_gd: ArrayView<f64, Ix2>,
    c_dd: ArrayView<f64, Ix2>,
    v_g: ArrayView<f64, Ix1>,
    n_charge: u64,
) -> Array1<f64> {
    let n_continuous = c_gd.dot(&v_g);
    let isolation_correction =
        (n_charge as f64 - n_continuous.sum()) * (c_dd.sum_axis(Axis(0)) / c_dd.sum());
    return n_continuous + isolation_correction;
}

#[allow(non_snake_case)]
fn init_osqp_problem_closed<'a>(
    v_g: ArrayView<f64, Ix1>,
    c_gd: ArrayView<'a, f64, Ix2>,
    c_dd_inv: ArrayView<'a, f64, Ix2>,
    n_charge: u64,
    polish: bool,
) -> Problem {
    let dim = c_dd_inv.shape()[0];
    let P = CscMatrix::from(c_dd_inv.rows()).into_upper_tri();

    let q_array = -c_dd_inv.dot(&c_gd.dot(&v_g));
    let q = q_array.as_slice().expect("failed to get slice of q");

    let l_array = ndarray::concatenate(
        Axis(0),
        &[
            Array1::<f64>::from_elem(1, n_charge as f64).view(),
            Array1::<f64>::zeros(dim).view(),
        ],
    )
    .expect("failed to concatenate arrays");
    let l = l_array.as_slice().expect("failed to get slice of l");

    let u_array = ndarray::concatenate(
        Axis(0),
        &[
            Array1::<f64>::from_elem(1, n_charge as f64).view(),
            Array1::<f64>::from_elem(dim, n_charge as f64).view(),
        ],
    )
    .expect("failed to concatenate arrays");
    let u = u_array.as_slice().expect("failed to get slice of u");

    let A = {
        let ones = Array2::<f64>::ones((1, dim));
        let identity = Array2::<f64>::eye(dim);
        let A = ndarray::concatenate(Axis(0), &[ones.view(), identity.view()])
            .expect("failed to concatenate arrays");
        CscMatrix::from(A.rows())
    };

    let settings = Settings::default().alpha(1.).verbose(false).polish(polish);
    Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem")
}

#[allow(non_snake_case)]
fn compute_argmin_closed(
    n_continuous: Array1<f64>,
    c_dd_inv: ArrayView<f64, Ix2>,
    c_gd: ArrayView<f64, Ix2>,
    v_g: ArrayView<f64, Ix1>,
    n_charge: u64,
    threshold: f64,
    T: f64,
) -> Array1<f64> {
    let n_list = closed_charge_configurations(n_continuous, n_charge, threshold);
    let vg_dash = c_gd.dot(&v_g);

    match T > 0.0 {
        false => hard_argmin(n_list, c_dd_inv, vg_dash),
        true => soft_argmin(n_list, c_dd_inv, vg_dash, T),
    }
}
