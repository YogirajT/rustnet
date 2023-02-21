#![allow(dead_code)]
use super::matrix::Operation::{Add, Subtract};

use super::matrix::{col_sum, matrix_max, matrix_multiply, multiply, row_sum};
use super::{
    matrix::{dot_product, linear_op, transpose, zeroes},
    types::NetworkParams,
};

fn relu(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let row_count = input.len();
    let column_count = input.first().unwrap().len();
    let mut output = vec![vec![0.0; column_count]; row_count];

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
    let mut output = vec![vec![0.0; column_count]; row_count];

    for (i, row) in input.iter().enumerate().take(row_count) {
        for (j, cell) in row.iter().enumerate().take(column_count) {
            output[i][j] = if *cell > 0.0 { 1.0 } else { 0.0 }
        }
    }

    output
}

pub fn softmax(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut softmax_output: Vec<Vec<f64>> = vec![vec![0.0; matrix[0].len()]; matrix.len()];

    let max = matrix_max(matrix);

    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            softmax_output[i][j] = f64::exp(matrix[i][j] - max);
        }
    }

    let column_sums = col_sum(&softmax_output);

    softmax_output
        .iter()
        .map(|m_row| {
            m_row
                .iter()
                .zip(&column_sums)
                .map(|(&cell, column_sum)| cell / column_sum)
                .collect()
        })
        .collect()
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

pub fn get_predictions(matrix: &[Vec<f64>]) -> Vec<usize> {
    let ncol = matrix[0].len();
    let mut result = vec![0; ncol];
    for col in 0..ncol {
        let mut max_index = 0;
        let mut max_value = matrix[0][col];
        for row in 1..matrix.len() {
            if matrix[row][col] > max_value {
                max_index = row;
                max_value = matrix[row][col];
            }
        }
        result[col] = max_index;
    }
    result
}

pub fn get_accuracy(labels: &[Vec<f64>], prediction: Vec<usize>) -> f64 {
    let labels_arr = labels.first().unwrap();
    let mut accuracy = 0.0;
    for (i, cell) in labels_arr.iter().enumerate() {
        if *cell as usize == prediction[i] {
            accuracy += 1.0
        }
    }

    accuracy / labels_arr.len() as f64 * 100.0
}

pub fn forward_propagation(
    network_params: NetworkParams,
    input_image: &[Vec<f64>],
) -> NetworkParams {
    let (w_1, b_1, w_2, _b_2) = network_params;

    let weighted_input = dot_product(&w_1, input_image);

    let z_1 = linear_op(Add, &weighted_input, &b_1);

    let activation_1 = relu(&z_1);

    let weighted_l1 = dot_product(&w_2, &activation_1);

    let z_2 = linear_op(Add, &weighted_l1, &_b_2);

    let activation_2 = softmax(&z_2);

    (z_1, activation_1, z_2, activation_2)
}

pub fn back_propagation(
    network_params: NetworkParams,
    _w_2: Vec<Vec<f64>>,
    labels: Vec<Vec<f64>>,
    input_image: &[Vec<f64>],
) -> NetworkParams {
    let (z_1, activation_1, _z_2, activation_2) = network_params;

    let m_inverse = 1.0 / (labels.first().unwrap().len() as f64);

    let expected_labels = transform_labels_to_network_output(&labels);

    let delta_z_2 = linear_op(Subtract, &activation_2, &expected_labels);

    let transposed_a_1 = transpose(&activation_1);

    let delta_w_2 = multiply(&dot_product(&delta_z_2, &transposed_a_1), m_inverse);

    let sum_delta_z_2 = row_sum(&delta_z_2);

    let delta_b_2 = multiply(&sum_delta_z_2, m_inverse);

    let dot_w_2_d_z_2 = dot_product(&transpose(&_w_2), &delta_z_2);

    let deriv_z_1 = relu_derivative(&z_1);

    let delta_z_1 = matrix_multiply(&dot_w_2_d_z_2, &deriv_z_1);

    let delta_w_1 = multiply(&dot_product(&delta_z_1, &transpose(input_image)), m_inverse);

    let delta_b_1 = multiply(&row_sum(&delta_z_1), m_inverse);

    (delta_w_1, delta_b_1, delta_w_2, delta_b_2)
}
