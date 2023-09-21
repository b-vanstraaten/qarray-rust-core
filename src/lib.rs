mod rust_fn;
mod charge_configurations;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn rusty_capacitance_model_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    fn closed_charge_configurations_brute_force<'py>(py: Python<'py>,
                              n_charge: isize,
                              n_dot: isize,
                              floor_values: PyReadonlyArray1<isize>,
    ) -> &'py PyArray2<isize> {

        let results_array = charge_configurations::closed_charge_configurations_brute_force(n_charge, n_dot, floor_values.as_array());
        results_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn ground_state_open<'py>(py: Python<'py>,
                              v_g: PyReadonlyArray2<f64>,
                              c_gd: PyReadonlyArray2<f64>,
                              c_dd_inv: PyReadonlyArray2<f64>,
                              threshold: f64,
    ) -> &'py PyArray2<f64> {
        let v_g = v_g.as_array();
        let c_gd = c_gd.as_array();
        let c_dd_inv = c_dd_inv.as_array();

        let results_array = rust_fn::ground_state_open_1d(v_g, c_gd, c_dd_inv, threshold);
        results_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn ground_state_closed<'py>(py: Python<'py>,
                                v_g: PyReadonlyArray2<f64>,
                                n_charge: f64,
                                c_gd: PyReadonlyArray2<f64>,
                                c_dd_inv: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let v_g = v_g.as_array();
        let c_gd = c_gd.as_array();
        let c_dd_inv = c_dd_inv.as_array();

        let results_array = rust_fn::ground_state_closed_1d(v_g, n_charge, c_gd, c_dd_inv);
        results_array.into_pyarray(py)
    }
    Ok(())
}

