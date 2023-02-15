use super::{
    matrix::{dot_product, matrix_addition},
    types::NetworkParams,
};

#[allow(dead_code)]
fn relu(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let row_count = input.len();
    let column_count = input.first().unwrap().len();
    let mut output = vec![vec![0.0; row_count]; column_count];

    for (i, row) in input.iter().enumerate().take(row_count) {
        for (j, cell) in row.iter().enumerate().take(column_count) {
            output[i][j] = if *cell > 0.0 { *cell } else { 0.0 }
        }
    }

    output
}

#[allow(dead_code)]
pub fn softmax(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let row_count = input.len();
    let column_count = input.first().unwrap().len();
    let mut output = vec![vec![0.0; row_count]; column_count];

    for (i, row) in input.iter().enumerate().take(row_count) {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let denominator = row.iter().cloned().map(|x| (x - max).exp()).sum::<f64>();

        for (j, cell) in row.iter().enumerate().take(column_count) {
            output[i][j] = (*cell - max).exp() / denominator
        }
    }

    output
}

#[allow(dead_code)]
pub fn forward_propagation(
    network_params: NetworkParams,
    input_image: Vec<Vec<f64>>,
) -> NetworkParams {
    let (w_1, _b_1, w_2, b_2) = network_params;

    let weighted_input = dot_product(&w_1, &input_image);

    let z_1 = matrix_addition(&weighted_input, &_b_1);

    let activation_1 = relu(&z_1);

    let weighted_l1 = dot_product(&w_2, &activation_1);

    let z_2 = matrix_addition(&weighted_l1, &b_2);

    let activation_2 = softmax(&z_2);

    (z_1, activation_1, z_2, activation_2)
}

#[allow(dead_code)]
pub fn backward_propagation(
    network_params: NetworkParams,
    w_2: Vec<Vec<f64>>,
    labels: Vec<Vec<f64>>,
) {
    todo!()
}
