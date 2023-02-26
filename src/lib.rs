pub mod common;

#[macro_export]
macro_rules! save_to_file {
    ( $n:expr ) => {
        let folder = "results/";
        let full_path = format!("{}/{}.csv", folder, stringify!($n));
        {
            match common::io::write_csv(&full_path, &$n) {
                Ok(_) => println!("Successfully wrote CSV file"),
                Err(e) => println!("Error writing CSV file: {e:?}"),
            };
        }
    };
}
