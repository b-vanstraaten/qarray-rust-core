use ndarray::{Array1, Array2, ArrayView, Axis, Ix2};

pub fn hard_argmin(
    n_list: Array2<f64>,
    c_dd_inv: ArrayView<f64, Ix2>,
    vg_dash: Array1<f64>,
) -> Array1<f64> {
    let argmin_index = n_list
        .outer_iter()
        .map(|x| x.to_owned() - &vg_dash)
        .map(|x| x.dot(&c_dd_inv.dot(&x)))
        .enumerate()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).expect("failed to compare floats"))
        .map(|(idx, _)| idx);

    match argmin_index {
        Some(idx) => n_list.index_axis(Axis(0), idx).to_owned(),
        None => panic!("failed to compute argmin"),
    }
}

#[allow(non_snake_case)]
pub fn soft_argmin(
    n_list: Array2<f64>,
    c_dd_inv: ArrayView<f64, Ix2>,
    vg_dash: Array1<f64>,
    T: f64,
) -> Array1<f64> {
    let F: Array1<f64> = n_list
        .outer_iter()
        .map(|x| x.to_owned() - &vg_dash)
        .map(|x| x.dot(&c_dd_inv.dot(&x)))
        .map(|x| -x / T)
        .collect();

    let max_value = F.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let mut F = F - max_value;
    F.mapv_inplace(f64::exp);

    let sum_weights = F.sum_axis(Axis(0));
    let weighted_n_list = &n_list * &F.insert_axis(Axis(1));
    let n_min = weighted_n_list.sum_axis(Axis(0)) / &sum_weights;
    n_min.to_owned()
}
