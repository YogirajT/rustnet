#![allow(dead_code)]
use std::fmt::Display;
use std::fs::File;
use std::io::Write;

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
