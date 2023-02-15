mod common;

use csv::Reader;
use rustnet::common::matrix::create_vec_from_csv;
use rustnet::common::matrix::dot_product;
use rustnet::common::matrix::get_network_params;
use rustnet::common::matrix::shuffle_matrix;
use rustnet::common::matrix::split_matrix;
use rustnet::common::matrix::transpose;
use rustnet::common::network_functions::forward_propagation;
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

    let network_params = get_network_params();

    let l1 = forward_propagation(network_params, _dev_data);

    Ok(())
}

fn main() {
    let matrix1 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

    let matrix2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];

    let product = dot_product(&matrix1, &matrix2);

    println!("{product:?}");

    if let Err(err) = init() {
        println!("error running example: {err}");
        process::exit(1);
    }
}
