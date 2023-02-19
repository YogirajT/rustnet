mod common;

use csv::Reader;
use rustnet::common::matrix::create_vec_from_csv;
use rustnet::common::matrix::get_network_params;
use rustnet::common::matrix::shuffle_matrix;
use rustnet::common::matrix::split_matrix;
use rustnet::common::matrix::transpose;
use rustnet::common::network_functions::back_propagation;
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

    let (w_1, b_1, w_2, b_2) = get_network_params();

    let forward_prop = forward_propagation((w_1, b_1, w_2.clone(), b_2), _dev_data);

    back_propagation(forward_prop, &w_2, _dev_labels);

    Ok(())
}

fn main() {
    if let Err(err) = init() {
        println!("error running example: {err}");
        process::exit(1);
    }
}
