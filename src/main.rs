mod common;

use common::matrix::DotProduct;
use common::matrix::IMatrix;

use csv::Reader;
use std::error::Error;
use std::io;
use std::process;

fn example() -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_reader(io::stdin());

    for result in reader.records() {
        let record = result.unwrap();
        println!("Name: {:?}", record);
        break;
    }

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

    if let Err(err) = example() {
        println!("error running example: {}", err);
        process::exit(1);
    }
}
