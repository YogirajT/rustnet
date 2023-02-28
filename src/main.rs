mod common;

use dotenv::dotenv;
use rustnet::common::canvas::init_canvas;
use rustnet::common::io::{check_results_exist, read_file_into_vector, save_predictors};
use rustnet::common::network_functions::{prepare_data, train};
use rustnet::save_to_file;

fn main() {
    dotenv().ok();

    let bool = check_results_exist();

    match bool {
        true => {
            init_canvas();
        }
        false => {
            println!("Predictors not found, re-training");

            let training_set = read_file_into_vector();

            let (train_labels, train_data) = prepare_data(training_set);

            let iterations = std::env::var("ITERATIONS")
                .expect("ITERATIONS must be set.")
                .parse::<usize>()
                .unwrap();

            let alpha = std::env::var("ALPHA")
                .expect("ALPHA must be set.")
                .parse::<f32>()
                .unwrap();

            let (w_1, b_1, w_2, b_2) = train(train_labels, train_data, iterations, alpha);

            save_predictors(w_1, b_1, w_2, b_2);

            println!("Predictors generated, please rerun the program to launch prediction canvas");
        }
    }
}
