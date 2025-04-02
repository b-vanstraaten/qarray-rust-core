use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod charge_configurations;

mod closed_dots;
mod helper_functions;
mod open_dots;

/// A Python module implemented in Rust.
#[pymodule]
fn qarray_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add functions to the Python module
    m.add_function(wrap_pyfunction!(open_charge_configurations, m)?)?;
    m.add_function(wrap_pyfunction!(closed_charge_configurations, m)?)?;
    m.add_function(wrap_pyfunction!(ground_state_open, m)?)?;
    m.add_function(wrap_pyfunction!(ground_state_closed, m)?)?;
    Ok(())
}

/// Open charge configurations.
#[pyfunction]
fn open_charge_configurations(
    py: Python,
    n_continuous: PyReadonlyArray1<f64>,
    threshold: f64,
) -> Py<PyArray2<f64>> {
    let n_continuous = n_continuous.as_array();
    let results_array =
        charge_configurations::open_charge_configurations(n_continuous.to_owned(), threshold);
    results_array.into_pyarray(py).into()
}

/// Closed charge configurations.
#[pyfunction]
fn closed_charge_configurations(
    py: Python,
    n_continuous: PyReadonlyArray1<f64>,
    n_charge: u64,
    threshold: f64,
) -> Py<PyArray2<f64>> {
    let n_continuous = n_continuous.as_array();
    let results_array = charge_configurations::closed_charge_configurations(
        n_continuous.to_owned(),
        n_charge,
        threshold,
    );
    results_array.into_pyarray(py).into()
}

/// Ground state for open configurations.
#[pyfunction]
fn ground_state_open(
    py: Python,
    v_g: PyReadonlyArray2<f64>,
    c_gd: PyReadonlyArray2<f64>,
    c_dd_inv: PyReadonlyArray2<f64>,
    threshold: f64,
    polish: bool,
    t: f64,
) -> Py<PyArray2<f64>> {
    let v_g = v_g.as_array();
    let c_gd = c_gd.as_array();
    let c_dd_inv = c_dd_inv.as_array();

    let results_array = open_dots::ground_state_open_1d(v_g, c_gd, c_dd_inv, threshold, polish, t);
    results_array.into_pyarray(py).into()
}

/// Ground state for closed configurations.
#[pyfunction]
fn ground_state_closed(
    py: Python,
    v_g: PyReadonlyArray2<f64>,
    n_charge: u64,
    c_gd: PyReadonlyArray2<f64>,
    c_dd: PyReadonlyArray2<f64>,
    c_dd_inv: PyReadonlyArray2<f64>,
    threshold: f64,
    polish: bool,
    t: f64,
) -> Py<PyArray2<f64>> {
    let v_g = v_g.as_array();
    let c_gd = c_gd.as_array();
    let c_dd = c_dd.as_array();
    let c_dd_inv = c_dd_inv.as_array();

    let results_array = closed_dots::ground_state_closed_1d(
        v_g, n_charge, c_gd, c_dd, c_dd_inv, threshold, polish, t,
    );
    results_array.into_pyarray(py).into()
}
