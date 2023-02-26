mod common;

use common::matrix::split_matrix;
use csv::Reader;
use rustnet::common::matrix::create_vec_from_csv;
use rustnet::common::matrix::divide;
use rustnet::common::matrix::get_network_params;
use rustnet::common::matrix::matrix_subtract;
use rustnet::common::matrix::multiply;
use rustnet::common::matrix::shuffle_matrix;
use rustnet::common::matrix::transpose;
use rustnet::common::network_functions::back_propagation;
use rustnet::common::network_functions::forward_propagation;
use rustnet::common::network_functions::get_accuracy;
use rustnet::common::network_functions::get_predictions;
use rustnet::common::types::NetworkParams;
use rustnet::save_to_file;
use std::io;

fn init(alpha: f64, rounds: usize) -> NetworkParams {
    let reader = Reader::from_reader(io::stdin());

    let mut dev_set = create_vec_from_csv(reader);

    shuffle_matrix(&mut dev_set);

    let transposed_dev_matrix = transpose(&dev_set);

    let (_dev_labels, dev_data) = split_matrix(&transposed_dev_matrix, 1);

    let (mut w_1, mut b_1, mut w_2, mut b_2) = get_network_params();

    let normalized_input = divide(&dev_data, 255.0);

    for _i in 0..rounds {
        let forward_prop = forward_propagation(
            (w_1.clone(), b_1.clone(), w_2.clone(), b_2.clone()),
            &normalized_input.clone(),
        );

        let (delta_w_1, delta_b_1, delta_w_2, delta_b_2) = back_propagation(
            forward_prop.clone(),
            w_2.clone(),
            _dev_labels.clone(),
            &normalized_input.clone(),
        );

        w_1 = matrix_subtract(&w_1, &multiply(&delta_w_1, alpha));
        b_1 = matrix_subtract(&b_1, &multiply(&delta_b_1, alpha));
        w_2 = matrix_subtract(&w_2, &multiply(&delta_w_2, alpha));
        b_2 = matrix_subtract(&b_2, &multiply(&delta_b_2, alpha));

        let prediction = get_predictions(&forward_prop.3.clone());
        println!(
            "Accuracy: {}",
            get_accuracy(&_dev_labels.clone(), prediction)
        )
    }

    (w_1, b_1, w_2, b_2)
}

fn main() {
    let (w_1, b_1, w_2, b_2) = init(0.10, 1);

    save_to_file!(w_1);
    save_to_file!(b_1);
    save_to_file!(w_2);
    save_to_file!(b_2);


    // draw(&get_nth_column(&_dev_data, 0));
    // println!("\n Number: {:?}", &_dev_labels[0][0]);
}
