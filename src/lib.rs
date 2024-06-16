use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

mod charge_configurations;
mod closed_dots;
mod helper_functions;
mod open_dots;

#[pymodule]
fn qarray_rust_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn open_charge_configurations<'py>(
        py: Python<'py>,
        n_continuous: PyReadonlyArray1<f64>,
        threshold: f64,
    ) -> &'py PyArray2<f64> {
        let n_continuous = n_continuous.as_array();
        let results_array =
            charge_configurations::open_charge_configurations(n_continuous.to_owned(), threshold);
        results_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn closed_charge_configurations<'py>(
        py: Python<'py>,
        n_continuous: PyReadonlyArray1<f64>,
        n_charge: u64,
        threshold: f64,
    ) -> &'py PyArray2<f64> {
        let n_continuous = n_continuous.as_array();
        let results_array = charge_configurations::closed_charge_configurations(
            n_continuous.to_owned(),
            n_charge,
            threshold,
        );
        results_array.into_pyarray(py)
    }

    #[pyfn(m)]
    #[allow(non_snake_case)]
    fn ground_state_open<'py>(
        py: Python<'py>,
        v_g: PyReadonlyArray2<f64>,
        c_gd: PyReadonlyArray2<f64>,
        c_dd_inv: PyReadonlyArray2<f64>,
        threshold: f64,
        polish: bool,
        T: f64,
    ) -> &'py PyArray2<f64> {
        let v_g = v_g.as_array();
        let c_gd = c_gd.as_array();
        let c_dd_inv = c_dd_inv.as_array();

        let results_array =
            open_dots::ground_state_open_1d(v_g, c_gd, c_dd_inv, threshold, polish, T);
        results_array.into_pyarray(py)
    }

    #[pyfn(m)]
    #[allow(non_snake_case)]
    fn ground_state_closed<'py>(
        py: Python<'py>,
        v_g: PyReadonlyArray2<f64>,
        n_charge: u64,
        c_gd: PyReadonlyArray2<f64>,
        c_dd: PyReadonlyArray2<f64>,
        c_dd_inv: PyReadonlyArray2<f64>,
        threshold: f64,
        polish: bool,
        T: f64,
    ) -> &'py PyArray2<f64> {
        let v_g = v_g.as_array();
        let c_gd = c_gd.as_array();
        let c_dd = c_dd.as_array();
        let c_dd_inv = c_dd_inv.as_array();

        let results_array = closed_dots::ground_state_closed_1d(
            v_g, n_charge, c_gd, c_dd, c_dd_inv, threshold, polish, T,
        );
        results_array.into_pyarray(py)
    }
    Ok(())
}
