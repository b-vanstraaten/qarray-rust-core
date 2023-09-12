use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};


// NOTE
// * numpy defaults to np.float64, if you use other type than f64 in Rust
//   you will have to change type in Python before calling the Rust function.

// The name of the module must be the same as the rust package name
#[pymodule]
fn rusty_capacitance_model_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // This is a pure function (no mutations of incoming data).
    // You can see this as the python array in the function arguments is readonly.
    // The object we return will need ot have the same lifetime as the Python.
    // Python will handle the objects deallocation.
    // We are having the Python as input with a lifetime parameter.
    // Basically, none of the data that comes from Python can survive
    // longer than Python itself. Therefore, if Python is dropped, so must our Rust Python-dependent variables.

    #[pyfn(m)]
    fn ground_state<'py>(py: Python<'py>,
                         v_g: PyReadonlyArray2<f64>,
                         c_gd: PyReadonlyArray2<f64>,
                         c_dd_inv: PyReadonlyArray2<f64>,
                         tolerance: f64,
    ) -> &'py PyArray2<f64> {
        let v_g = v_g.as_array();
        let c_gd = c_gd.as_array();
        let c_dd_inv = c_dd_inv.as_array();

        let results_array = rust_fn::ground_state_1d(v_g, c_gd, c_dd_inv, tolerance);
        results_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn ground_state_isolated<'py>(py: Python<'py>,
                                  v_g: PyReadonlyArray2<f64>,
                                  n_charge: f64,
                                  c_gd: PyReadonlyArray2<f64>,
                                  cdd: PyReadonlyArray2<f64>,
                                  c_dd_inv: PyReadonlyArray2<f64>,
                                  tolerance: f64,
    ) -> &'py PyArray2<f64> {
        let v_g = v_g.as_array();
        let c_gd = c_gd.as_array();
        let c_dd = cdd.as_array();
        let c_dd_inv = c_dd_inv.as_array();

        let results_array = rust_fn::ground_state_1d_isolated(v_g, n_charge, c_gd, c_dd, c_dd_inv, tolerance);
        results_array.into_pyarray(py)
    }

    Ok(())
}

// The rust side functions
// Put it in mod to separate it from the python bindings
// These are just some random operations
// you probably want to do something more meaningful.
mod rust_fn {
    use std::usize;

    use itertools;
    use itertools::{Itertools, repeat_n};
    use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix1, Ix2, s};
    use rayon::prelude::*;

    pub fn ground_state_1d<'a>(
        v_g: ArrayView<'a, f64, Ix2>,
        c_gd: ArrayView<'a, f64, Ix2>,
        c_dd_inv: ArrayView<'a, f64, Ix2>,
        tolerance: f64,
    ) -> Array<f64, Ix2> {
        let n = v_g.shape()[0];
        let m = c_gd.shape()[0];
        let mut results_array = Array2::zeros((n, m));

        let mut rows: Vec<_> = results_array.axis_iter_mut(ndarray::Axis(0)).collect();

        rows.par_iter_mut().enumerate().for_each(|(j, result_row)| {
            let v_g_row = v_g.slice(s![j, ..]);
            let n_charge = ground_state_0d(v_g_row, c_gd, c_dd_inv, tolerance);
            result_row.assign(&n_charge);
        });

        results_array
    }

    pub fn ground_state_1d_isolated<'a>(
        v_g: ArrayView<'a, f64, Ix2>,
        n_charge: f64,
        c_gd: ArrayView<'a, f64, Ix2>,
        c_dd: ArrayView<'a, f64, Ix2>,
        c_dd_inv: ArrayView<'a, f64, Ix2>,
        tolerance: f64,
    ) -> Array<f64, Ix2> {
        let n = v_g.shape()[0];
        let m = c_gd.shape()[0];
        let mut results_array = Array2::zeros((n, m));

        let mut rows: Vec<_> = results_array.axis_iter_mut(ndarray::Axis(0)).collect();

        rows.par_iter_mut().enumerate().for_each(|(j, result_row)| {
            let v_g_row = v_g.slice(s![j, ..]);
            let n_charge = ground_state_0d_isolated(v_g_row, n_charge, c_gd, c_dd, c_dd_inv, tolerance);
            result_row.assign(&n_charge);
        });

        results_array
    }


    pub fn ground_state_0d<'a>(v_g: ArrayView<f64, Ix1>, c_gd: ArrayView<'a, f64, Ix2>, c_dd_inv: ArrayView<'a, f64, Ix2>, tolerance: f64) -> Array<f64, Ix1> {

        // compute the continuous part of the ground state
        let mut n_continuous = c_gd.dot(&v_g);

        // clip the continuous part to be positive
        n_continuous.mapv_inplace(|x| x.max(0.0));

        // determine if rounding is required
        let requires_floor_ceil = n_continuous.iter().any(|x| (x.fract() - 0.5).abs() < tolerance);

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
            let floor_ceil_args = (0..n_continuous.len())
                .filter(|i| (n_continuous[*i].fract() - 0.5).abs() < tolerance)
                .collect::<Vec<usize>>();

            // round args are the indices not in floor_ceil_args, which can just be normally rounded to the nearest integer
            let round_args: Array1<usize> = (0..n_continuous.len())
                .filter(|i| (n_continuous[i.to_owned()].fract() - 0.5).abs() >= tolerance)
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

    pub fn ground_state_0d_isolated<'a>(v_g: ArrayView<f64, Ix1>, n_charge: f64,
                                        c_gd: ArrayView<'a, f64, Ix2>, c_dd: ArrayView<'a, f64, Ix2>, c_dd_inv: ArrayView<'a, f64, Ix2>, tolerance: f64) -> Array<f64, Ix1> {

        // compute the continuous part of the ground state
        let mut n_continuous = c_gd.dot(&v_g);
        let isolation_correction = (n_charge - n_continuous.sum()) * (c_dd.sum_axis(Axis(0)) / c_dd.sum());
        n_continuous = n_continuous + isolation_correction;

        // clip the continuous part to be positive
        n_continuous.mapv_inplace(|x| x.max(0.0).min(n_charge));

        // determine if rounding is required
        let requires_floor_ceil = n_continuous.iter().any(|x| (x.fract() - 0.5).abs() < tolerance);

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
            let floor_ceil_args = (0..n_continuous.len())
                .filter(|i| (n_continuous[*i].fract() - 0.5).abs() < tolerance)
                .collect::<Vec<usize>>();

            // round args are the indices not in floor_ceil_args, which can just be normally rounded to the nearest integer
            let round_args: Array1<usize> = (0..n_continuous.len())
                .filter(|i| (n_continuous[i.to_owned()].fract() - 0.5).abs() >= tolerance)
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
                if u < min_u && (&n_continuous + &delta).sum() - n_charge < 1e-6 {
                    min_u = u;
                    min_delta.assign(&delta); // Store the delta for the minimum u
                }
            }
            n_continuous + min_delta
        }
    }
}