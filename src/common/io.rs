#![allow(dead_code)]
use std::fmt::Display;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use csv::ReaderBuilder;

use super::matrix::create_vec_from_csv;
use super::types::NetworkParams;

const FILE_PATH: &str = "results/";

pub enum ResultFiles {
    B1,
    B2,
    W1,
    W2,
}

impl ResultFiles {
    fn as_str(&self) -> &'static str {
        match self {
            ResultFiles::B1 => "b_1",
            ResultFiles::B2 => "b_2",
            ResultFiles::W1 => "w_1",
            ResultFiles::W2 => "w_2",
        }
    }
}

const RESULT_FILES: [ResultFiles; 4] = [
    ResultFiles::B1,
    ResultFiles::B2,
    ResultFiles::W1,
    ResultFiles::W2,
];

pub fn write_csv<T>(filename: &str, matrix: &Vec<Vec<T>>) -> std::io::Result<()>
where
    T: Display,
{
    let mut file = File::create(filename)?;

    for row in matrix {
        let row_str = row
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(",");
        file.write_all(row_str.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

pub fn read_file_into_vector() -> Vec<Vec<f32>> {
    let reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(io::stdin());

    create_vec_from_csv(reader)
}

pub fn check_results_exist() -> bool {
    let mut ready_file_counter = 0;

    for (_, name) in RESULT_FILES.iter().enumerate() {
        let filename = name.as_str();
        let file_path = format!("{FILE_PATH}{filename}.csv");
        let file = Path::new(&file_path);

        if file.exists() {
            match fs::read_to_string(file) {
                Ok(content) => {
                    if !content.is_empty() {
                        ready_file_counter += 1;
                    }
                }
                Err(e) => {
                    println!("Error reading file, consider emptying results/ folder in the project root:  {e}");
                }
            }
        }
    }

    ready_file_counter == 4
}

pub fn load_network_params() -> NetworkParams {
    let b_1_path = ResultFiles::B1.as_str();
    let b_2_path = ResultFiles::B2.as_str();
    let w_1_path = ResultFiles::W1.as_str();
    let w_2_path = ResultFiles::W2.as_str();

    let b_1_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(format!("{FILE_PATH}{b_1_path}.csv"))
        .unwrap();
    let b_2_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(format!("{FILE_PATH}{b_2_path}.csv"))
        .unwrap();
    let w_1_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(format!("{FILE_PATH}{w_1_path}.csv"))
        .unwrap();
    let w_2_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(format!("{FILE_PATH}{w_2_path}.csv"))
        .unwrap();

    let b_1 = create_vec_from_csv(b_1_reader);
    let b_2 = create_vec_from_csv(b_2_reader);
    let w_1 = create_vec_from_csv(w_1_reader);
    let w_2 = create_vec_from_csv(w_2_reader);

    (w_1, b_1, w_2, b_2)
}
