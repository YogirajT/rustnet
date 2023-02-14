use csv::Reader;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::clone::Clone;
use std::io::Stdin;

use super::types::NetworkParams;

#[allow(dead_code)]
pub fn dot_product(matrix_1: &Vec<Vec<f64>>, matrix_2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m1_rows = matrix_1.len();
    let m1_cols = matrix_1[0].len();
    let m2_rows = matrix_2.len();
    let m2_cols = matrix_2[0].len();

    if m1_cols != m2_rows {
        panic!("The number of columns in the first matrix must be equal to the number of rows in the second matrix!");
    }
    let mut result = vec![vec![0.0; m2_cols]; m1_rows];

    for (i, own_row) in matrix_1.iter().enumerate().take(m1_rows) {
        for (j, _other_col) in matrix_2.iter().enumerate().take(m2_cols) {
            for (k, cell) in own_row.iter().enumerate().take(m1_cols) {
                result[i][j] += cell * matrix_2[k][j];
            }
        }
    }

    result
}

#[allow(dead_code)]
pub fn create_vec_from_csv(mut rdr: Reader<Stdin>) -> Vec<Vec<f64>> {
    let mut vec = vec![];
    for result in rdr.records() {
        let record = result.unwrap();

        vec.push(
            record
                .iter()
                .map(|field| field.parse::<f64>().unwrap())
                .collect(),
        );
    }
    vec
}

#[allow(dead_code)]
pub fn transpose<T: Clone>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let row_count = matrix.len();
    let column_count = matrix[0].len();

    let mut transposed = vec![vec![matrix[0][0].clone(); row_count]; column_count];

    for (i, row) in matrix.iter().enumerate().take(row_count) {
        for (j, cell) in row.iter().enumerate().take(column_count) {
            transposed[j][i] = cell.clone();
        }
    }

    transposed
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn split_matrix(matrix: &[Vec<f64>], n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (first_n_rows, last_n_rows) = matrix.split_at(n);
    (first_n_rows.to_vec(), last_n_rows.to_vec())
}

#[allow(dead_code)]
pub fn get_nth_column<T>(matrix: &[Vec<T>], n: usize) -> Vec<T>
where
    T: Clone,
{
    matrix.iter().map(|row| row[n].clone()).collect()
}

#[allow(dead_code)]
pub fn rand_matrix(rows: usize, columns: usize) -> Vec<Vec<f64>> {
    let mut rng = Pcg64::seed_from_u64(0);
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

#[allow(dead_code)]
pub fn get_network_params() -> NetworkParams {
    let w_1 = rand_matrix(10, 784);
    let b_1 = rand_matrix(10, 1);
    let w_2 = rand_matrix(10, 10);
    let b_2 = rand_matrix(10, 1);

    (w_1, b_1, w_2, b_2)
}
