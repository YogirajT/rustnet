#![allow(dead_code)]
use super::io::load_network_params;
use super::matrix::Operation::Add;
use super::matrix::{
    col_sum, create_network_params, divide, flip_rotate, get_nth_column, matrix_max,
    matrix_multiply, multiply, row_sum, shuffle_matrix, split_matrix,
};
use super::{
    matrix::{dot_product, linear_op, matrix_subtract, transpose, zeroes},
    types::NetworkParams,
};

pub fn relu(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let row_count = input.len();
    let column_count = input.first().unwrap().len();
    let mut output = vec![vec![0.0; column_count]; row_count];

    for (i, row) in input.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            output[i][j] = if cell > &0.0 { *cell } else { 0.0 }
        }
    }

    output
}

fn relu_derivative(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let row_count = input.len();
    let column_count = input.first().unwrap().len();
    let mut output = vec![vec![0.0; column_count]; row_count];

    for (i, row) in input.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            output[i][j] = if *cell > 0.0 { 1.0 } else { 0.0 }
        }
    }

    output
}

pub fn softmax(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut softmax_output: Vec<Vec<f32>> =
        vec![vec![0.0; matrix.first().unwrap().len()]; matrix.len()];

    let max = matrix_max(matrix);

    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            softmax_output[i][j] = f32::exp(matrix[i][j] - max);
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

pub fn transform_labels_to_network_output(labels: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let labels_first_col = match labels {
        [x] => x,
        _ => panic!("expected single element"),
    };

    let rows_len = labels_first_col.len();

    let mut zeroes_matrix = zeroes(rows_len, 10);

    for (i, label) in labels_first_col.iter().enumerate() {
        zeroes_matrix[i][*label as usize] = 1.0;
    }

    transpose(&zeroes_matrix)
}

pub fn get_predictions(matrix: &[Vec<f32>]) -> Vec<usize> {
    let ncol = matrix.first().unwrap().len();
    let mut result = vec![0; ncol];
    for (i, _) in matrix.first().unwrap().iter().enumerate() {
        let mut max_index = 0;
        let mut max_value = matrix.first().unwrap()[i];
        for (j, _) in matrix.iter().enumerate().skip(1) {
            if matrix[j][i] > max_value {
                max_index = j;
                max_value = matrix[j][i];
            }
        }
        result[i] = max_index;
    }
    result
}

pub fn get_accuracy(labels: &[Vec<f32>], prediction: Vec<usize>) -> f32 {
    let labels_arr = labels.first().unwrap();
    let mut accuracy = 0.0;
    for (i, cell) in labels_arr.iter().enumerate() {
        if *cell as usize == prediction[i] {
            accuracy += 1.0
        }
    }

    accuracy / labels_arr.len() as f32 * 100.0
}

pub fn forward_propagation(
    network_params: NetworkParams,
    input_image: &[Vec<f32>],
) -> NetworkParams {
    let (w_1, b_1, w_2, b_2) = network_params;

    let weighted_input = dot_product(&w_1, input_image);

    let z_1 = linear_op(Add, &weighted_input, &b_1);

    let activation_1 = relu(&z_1);

    let weighted_l1 = dot_product(&w_2, &activation_1);

    let z_2 = linear_op(Add, &weighted_l1, &b_2);

    let activation_2 = softmax(&z_2);

    (z_1, activation_1, z_2, activation_2)
}

pub fn back_propagation(
    network_params: NetworkParams,
    w_2: Vec<Vec<f32>>,
    labels: Vec<Vec<f32>>,
    input_image: &[Vec<f32>],
) -> NetworkParams {
    let (z_1, activation_1, _z_2, activation_2) = network_params;

    let m_inverse = 1.0 / (labels.first().unwrap().len() as f32);

    let expected_labels = transform_labels_to_network_output(&labels);

    let delta_z_2 = matrix_subtract(&activation_2, &expected_labels);

    let transposed_a_1 = transpose(&activation_1);

    let delta_w_2 = multiply(&dot_product(&delta_z_2, &transposed_a_1), m_inverse);

    let sum_delta_z_2 = row_sum(&delta_z_2);

    let delta_b_2 = multiply(&sum_delta_z_2, m_inverse);

    let dot_w_2_d_z_2 = dot_product(&transpose(&w_2), &delta_z_2);

    let deriv_z_1 = relu_derivative(&z_1);

    let delta_z_1 = matrix_multiply(&dot_w_2_d_z_2, &deriv_z_1);

    let delta_w_1 = multiply(&dot_product(&delta_z_1, &transpose(input_image)), m_inverse);

    let delta_b_1 = multiply(&row_sum(&delta_z_1), m_inverse);

    (delta_w_1, delta_b_1, delta_w_2, delta_b_2)
}

pub fn train(
    train_labels: Vec<Vec<f32>>,
    train_data: Vec<Vec<f32>>,
    iterations: usize,
    alpha: f32,
) -> NetworkParams {
    let (mut w_1, mut b_1, mut w_2, mut b_2) = create_network_params();

    for i in 0..iterations {
        let forward_prop = forward_propagation(
            (w_1.clone(), b_1.clone(), w_2.clone(), b_2.clone()),
            &train_data.clone(),
        );

        let (delta_w_1, delta_b_1, delta_w_2, delta_b_2) = back_propagation(
            forward_prop.clone(),
            w_2.clone(),
            train_labels.clone(),
            &train_data.clone(),
        );

        w_1 = matrix_subtract(&w_1, &multiply(&delta_w_1, alpha));
        b_1 = matrix_subtract(&b_1, &multiply(&delta_b_1, alpha));
        w_2 = matrix_subtract(&w_2, &multiply(&delta_w_2, alpha));
        b_2 = matrix_subtract(&b_2, &multiply(&delta_b_2, alpha));

        println!("Iteration: {}", i + 1);
        let prediction = get_predictions(&forward_prop.3.clone());
        println!(
            "Accuracy: {}",
            get_accuracy(&train_labels.clone(), prediction)
        )
    }

    (w_1, b_1, w_2, b_2)
}

pub fn prepare_data(mut dev_set: Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    shuffle_matrix(&mut dev_set);

    let transposed_dev_matrix = transpose(&dev_set);

    let (train_labels, dev_data) = split_matrix(&transposed_dev_matrix, 1);

    let train_data = divide(&dev_data, 255.0);

    (train_labels, train_data)
}

pub fn predict(input: Vec<Vec<f32>>) -> String {
    let mut matrix = input;

    flip_rotate(&mut matrix);

    let flat_array = transpose(&[matrix.concat()]);

    let (w_1, b_1, w_2, b_2) = load_network_params();

    let (_, _, _, ac2) = forward_propagation((w_1, b_1, w_2, b_2), &flat_array);

    let col = get_nth_column(&ac2, 0);

    let (index, _) = col
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    index.to_string()
}
