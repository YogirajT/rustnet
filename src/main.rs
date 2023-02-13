mod common;

use common::matrix::DotProduct;
use common::matrix::IMatrix;

use csv::Reader;
use rustnet::common::matrix::create_vec_from_csv;
use rustnet::common::matrix::shuffle_matrix;
use rustnet::common::matrix::split_matrix;
use rustnet::common::matrix::transpose;
use std::error::Error;
use std::io;
use std::process;

fn init() -> Result<(), Box<dyn Error>> {
    let reader = Reader::from_reader(io::stdin());

    let mut matrix = create_vec_from_csv(reader);

    shuffle_matrix(&mut matrix);

    let (dev_set, test_set) = split_matrix(&matrix, 5000);

    let transposed_dev_matrix = transpose(&dev_set);

    let _transposed_test_set = transpose(&test_set);

    let (_dev_labels, _dev_data) = split_matrix(&transposed_dev_matrix, 1);

    println!("{:?}", _dev_labels[0].len());

    Ok(())
}

fn main() {
    let matrix1 = IMatrix {
        rows: 2,
        cols: 3,
        data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    };

    let matrix2 = IMatrix {
        rows: 3,
        cols: 2,
        data: vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]],
    };

    let product = matrix2.dot_product(&matrix1);

    println!("{:?}", product);

    if let Err(err) = init() {
        println!("error running example: {}", err);
        process::exit(1);
    }
}
