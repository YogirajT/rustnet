use csv::Reader;
use rand::Rng;
use std::clone::Clone;
use std::io::Stdin;

pub trait DotProduct {
    fn dot_product(&self, other: &Self) -> Self;
}

#[derive(Debug, PartialEq)]
pub struct IMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl DotProduct for IMatrix {
    fn dot_product(&self, other: &Self) -> Self {
        if self.cols != other.rows {
            panic!("The number of columns in the first matrix must be equal to the number of rows in the second matrix!");
        }
        let mut result = vec![vec![0.0; other.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        IMatrix {
            rows: self.rows,
            cols: other.cols,
            data: result,
        }
    }
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
pub fn transpose<T: Clone>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let row_count = matrix.len();
    let column_count = matrix[0].len();

    let mut transposed = vec![vec![matrix[0][0].clone(); row_count]; column_count];

    for i in 0..row_count {
        for j in 0..column_count {
            transposed[j][i] = matrix[i][j].clone();
        }
    }

    transposed
}

#[allow(dead_code)]
pub fn shuffle_matrix<T>(matrix: &mut Vec<Vec<T>>) {
    let mut rng = rand::thread_rng();
    shuffle(matrix, &mut rng);
}

fn shuffle<T, R: Rng>(matrix: &mut Vec<Vec<T>>, rng: &mut R) {
    let row_count = matrix.len();
    let range = 0..row_count;

    for i in range.clone() {
        let j = rng.gen_range(range.clone());
        matrix.swap(i, j);
    }
}

#[allow(dead_code)]
pub fn split_matrix(matrix: &Vec<Vec<f64>>, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let len = matrix.len();
    let (first_n_rows, last_n_rows) = matrix.split_at(len - n);
    (last_n_rows.to_vec(), first_n_rows.to_vec())
}

#[allow(dead_code)]
pub fn get_nth_column<T>(matrix: &Vec<Vec<T>>, n: usize) -> Vec<T>
where
    T: Clone,
{
    matrix.iter().map(|row| row[n].clone()).collect()
}
