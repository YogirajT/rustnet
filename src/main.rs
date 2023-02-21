mod common;

use csv::Reader;
use rustnet::common::matrix::create_vec_from_csv;
use rustnet::common::matrix::divide;
use rustnet::common::matrix::get_network_params;
use rustnet::common::matrix::linear_op;
use rustnet::common::matrix::multiply;
use rustnet::common::matrix::shuffle_matrix;
use rustnet::common::matrix::split_matrix;
use rustnet::common::matrix::transpose;
use rustnet::common::matrix::Operation::Subtract;
use rustnet::common::network_functions::back_propagation;
use rustnet::common::network_functions::forward_propagation;
use rustnet::common::network_functions::get_accuracy;
use rustnet::common::network_functions::get_predictions;
use rustnet::common::types::NetworkParams;
use std::io;

fn init(alpha: f64, rounds: usize) -> NetworkParams {
    let reader = Reader::from_reader(io::stdin());

    let mut matrix = create_vec_from_csv(reader);

    shuffle_matrix(&mut matrix);

    let (dev_set, test_set) = split_matrix(&matrix, 5000);

    let transposed_dev_matrix = transpose(&dev_set);

    let _transposed_test_set = transpose(&test_set);

    let (_dev_labels, _dev_data) = split_matrix(&transposed_dev_matrix, 1);

    // draw(&get_nth_column(&_dev_data, 0));
    // println!("\n Number: {:?}", &_dev_labels[0][0]);

    let (mut w_1, mut b_1, mut w_2, mut b_2) = get_network_params();

    let normalized_input = divide(&_dev_data, 255.0);

    for i in 0..rounds {
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

        w_1 = linear_op(Subtract, &w_1, &multiply(&delta_w_1, alpha));
        b_1 = linear_op(Subtract, &b_1, &multiply(&delta_b_1, alpha));
        w_2 = linear_op(Subtract, &w_2, &multiply(&delta_w_2, alpha));
        b_2 = linear_op(Subtract, &b_2, &multiply(&delta_b_2, alpha));

        if (i + 1) % (rounds / 10) == 0 {
            println!("Iteration: {i}");
            let prediction = get_predictions(&forward_prop.3.clone());
            println!(
                "Accuracy: {}",
                get_accuracy(&_dev_labels.clone(), prediction)
            )
        }
    }

    (w_1, b_1, w_2, b_2)
}

fn main() {
    let _params = init(0.10, 100);
}
