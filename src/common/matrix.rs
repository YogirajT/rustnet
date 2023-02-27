#![allow(dead_code)]
use super::types::NetworkParams;
use csv::Reader;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::clone::Clone;

pub enum Operation {
    Subtract,
    Add,
}

pub fn dot_product(matrix_1: &[Vec<f32>], matrix_2: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let m1_rows = matrix_1.len();
    let m1_cols = matrix_1.first().unwrap().len();
    let m2_rows = matrix_2.len();
    let m2_cols = matrix_2.first().unwrap().len();

    if m1_cols != m2_rows {
        panic!("The number of columns in the first matrix must be equal to the number of rows in the second matrix!");
    }

    let mut result = vec![vec![0.0; m2_cols]; m1_rows];

    for i in 0..m1_rows {
        for j in 0..m2_cols {
            let mut sum = 0.0;
            for (k, _) in matrix_2.iter().enumerate().take(m1_cols) {
                sum += matrix_1[i][k] * matrix_2[k][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

pub fn create_vec_from_csv<T: std::io::Read>(mut rdr: Reader<T>) -> Vec<Vec<f32>> {
    let mut vec = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();

        vec.push(
            record
                .iter()
                .map(|field| field.parse::<f32>().unwrap())
                .collect(),
        );
    }
    vec
}

pub fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix.first().unwrap().len();
    let mut result = vec![vec![0.0; rows]; cols];

    for (i, row) in matrix.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            result[j][i] = *cell;
        }
    }

    result
}

pub fn shuffle_matrix<T>(matrix: &mut [Vec<T>]) {
    let mut rng = rand::thread_rng();
    shuffle(matrix, &mut rng);
}

fn shuffle<T, R: Rng>(matrix: &mut [Vec<T>], rng: &mut R) {
    let row_count = matrix.len();
    let range = 0..row_count;

    for i in range.clone() {
        let j = rng.gen_range(range.clone());
        matrix.swap(i, j);
    }
}

pub fn split_matrix(matrix: &[Vec<f32>], n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (first_n_rows, last_n_rows) = matrix.split_at(n);
    (first_n_rows.to_vec(), last_n_rows.to_vec())
}

pub fn get_nth_column<T>(matrix: &[Vec<T>], n: usize) -> Vec<T>
where
    T: Clone,
{
    matrix.iter().map(|row| row[n].clone()).collect()
}

pub fn rand_matrix(rows: usize, columns: usize) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::from_entropy();
    let mut result = Vec::new();

    for _ in 0..rows {
        let mut row = Vec::new();

        for _ in 0..columns {
            let random_number = rng.gen_range(-0.5..0.5);
            row.push(random_number);
        }

        result.push(row);
    }

    result
}

pub fn create_network_params() -> NetworkParams {
    let w_1 = rand_matrix(10, 784);
    let b_1 = rand_matrix(10, 1);
    let w_2 = rand_matrix(10, 10);
    let b_2 = rand_matrix(10, 1);

    (w_1, b_1, w_2, b_2)
}

pub fn linear_op(action: Operation, matrix: &[Vec<f32>], bias: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let row_len = matrix.len();
    let col_len = matrix.first().unwrap().len();
    let mut result = vec![vec![0.0; col_len]; row_len];

    for (i, row) in matrix.iter().enumerate() {
        let bias_row = bias.get(i).unwrap();
        let bias_term = bias_row.first().unwrap();
        for (j, cell) in row.iter().enumerate() {
            if matches!(action, Operation::Subtract) {
                result[i][j] = *cell - *bias_term
            } else {
                result[i][j] = *cell + *bias_term
            }
        }
    }
    result
}

pub fn multiply(matrix: &[Vec<f32>], coeff: f32) -> Vec<Vec<f32>> {
    let row_len = matrix.len();
    let col_len = matrix.first().unwrap().len();
    let mut result = vec![vec![0.0; col_len]; row_len];

    for (i, row) in matrix.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            result[i][j] = *cell * coeff;
        }
    }

    result
}

pub fn divide(matrix: &[Vec<f32>], coeff: f32) -> Vec<Vec<f32>> {
    let m = matrix.len();
    let n = matrix.first().unwrap().len();
    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        for j in 0..n {
            result[i][j] = matrix[i][j] / coeff;
        }
    }

    result
}

pub fn matrix_multiply(matrix_1: &[Vec<f32>], matrix_2: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let m = matrix_1.len();
    let n = matrix_1.first().unwrap().len();
    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        for j in 0..n {
            result[i][j] = matrix_1[i][j] * matrix_2[i][j];
        }
    }

    result
}

pub fn matrix_subtract(matrix_1: &[Vec<f32>], matrix_2: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let m = matrix_1.len();
    let n = matrix_1.first().unwrap().len();
    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        for j in 0..n {
            result[i][j] = matrix_1[i][j] - matrix_2[i][j];
        }
    }

    result
}

pub fn zeroes(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    vec![vec![0.0; cols]; rows]
}

pub fn row_sum(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    matrix.iter().map(|row| vec![row.iter().sum()]).collect()
}

pub fn col_sum(matrix: &[Vec<f32>]) -> Vec<f32> {
    let num_cols = matrix.first().unwrap().len();
    let mut result = vec![0.0; num_cols];

    for row in matrix {
        for (j, value) in row.iter().enumerate() {
            result[j] += value;
        }
    }

    result
}

pub fn matrix_max(m: &[Vec<f32>]) -> f32 {
    let mut max_value = m[0][0];
    for row in m {
        for &value in row {
            if value > max_value {
                max_value = value;
            }
        }
    }
    max_value
}

pub fn matrix_min(m: &[Vec<f32>]) -> f32 {
    let mut min_value = m[0][0];
    for row in m {
        for &value in row {
            if value < min_value {
                min_value = value;
            }
        }
    }
    min_value
}

pub fn matrix_avg(m: &[Vec<f32>]) -> f32 {
    let num_rows = m.len();
    let num_cols = m.first().unwrap().len();

    let mut total = 0.0;
    for row in m {
        for &value in row {
            total += value
        }
    }
    total / (num_rows * num_cols) as f32
}
