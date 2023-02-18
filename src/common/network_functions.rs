#![allow(dead_code)]
use super::matrix::Operation::{ADD, SUBTRACT};

use super::{
    matrix::{dot_product, linear_op, transpose, zeroes},
    types::NetworkParams,
};

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

fn relu_derivative(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

pub fn transform_labels_to_network_output(labels: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let labels_first_col = match labels {
        [x] => x,
        _ => panic!("expected single element"),
    };

    let usize_col: Vec<usize> = labels_first_col.iter().map(|x| *x as usize).collect();
    let rows_len = labels_first_col.len();
    let cols_len = usize_col.iter().max().unwrap();

    let mut zeroes_matrix = zeroes(rows_len, *cols_len + 1);

    for i in 0..rows_len {
        let label = usize_col[i];

        zeroes_matrix[i][label] = 1.0;
    }

    transpose::<f64>(&zeroes_matrix)
}

pub fn forward_propagation(
    network_params: NetworkParams,
    input_image: Vec<Vec<f64>>,
) -> NetworkParams {
    let (w_1, _b_1, w_2, b_2) = network_params;

    let weighted_input = dot_product(&w_1, &input_image);

    let z_1 = linear_op(ADD, &weighted_input, &_b_1);

    let activation_1 = relu(&z_1);

    let weighted_l1 = dot_product(&w_2, &activation_1);

    let z_2 = linear_op(ADD, &weighted_l1, &b_2);

    let activation_2 = softmax(&z_2);

    (z_1, activation_1, z_2, activation_2)
}

pub fn back_propagation(
    network_params: NetworkParams,
    _w_2: Vec<Vec<f64>>,
    _labels: Vec<Vec<f64>>,
) {
    let (z_1, activation_1, z_2, activation_2) = network_params;
    // let output_labels = to_output_labels(rows_len, cols_len, &labels);

    let expected_labels = transform_labels_to_network_output(&_labels);

    let delta_z_2 = linear_op(SUBTRACT, &activation_2, &expected_labels);

    let transposed_a_1 = transpose(&activation_1);

    let delta_w_2 = dot_product(&delta_z_2, &transposed_a_1);

    todo!()
}
