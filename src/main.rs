mod common;

use common::matrix::create_vec_from_csv;
use common::matrix::DotProduct;
use common::matrix::IMatrix;

use csv::Reader;
use std::error::Error;
use std::io;
use std::process;

fn example() -> Result<(), Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = Reader::from_reader(io::stdin());
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result?;
        println!("{:?}", record);
    }
    let vector = create_vec_from_csv(rdr);

    println!("{:?}", vector[0]);

    Ok(())
}

fn main() {
    if let Err(err) = example() {
        println!("error running example: {}", err);
        process::exit(1);
    }

    let matrix1 = IMatrix {
        rows: 2,
        cols: 3,
        data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    };
    let matrix2 = IMatrix {
        rows: 4,
        cols: 2,
        data: vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
            vec![13.0, 14.0],
        ],
    };
    matrix1.dot_product(&matrix2);
}
