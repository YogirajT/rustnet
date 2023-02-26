mod common;
use dotenv::dotenv;
use rustnet::common::io::read_file_into_vector;
use rustnet::common::network_functions::{prepare_data, train};
use rustnet::save_to_file;

fn main() {
    dotenv().ok();

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

    save_to_file!(w_1);
    save_to_file!(b_1);
    save_to_file!(w_2);
    save_to_file!(b_2);

    // draw(&get_nth_column(&_dev_data, 0));
    // println!("\n Number: {:?}", &_dev_labels[0][0]);
}
