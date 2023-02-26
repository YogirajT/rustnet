mod common;

use rustnet::common::io::read_file_into_vector;
use rustnet::common::network_functions::{prepare_data, train};
use rustnet::save_to_file;

fn main() {
    let training_set = read_file_into_vector();

    let (train_labels, train_data) = prepare_data(training_set);

    let (w_1, b_1, w_2, b_2) = train(train_labels, train_data, 100, 0.15);

    save_to_file!(w_1);
    save_to_file!(b_1);
    save_to_file!(w_2);
    save_to_file!(b_2);

    // draw(&get_nth_column(&_dev_data, 0));
    // println!("\n Number: {:?}", &_dev_labels[0][0]);
}
