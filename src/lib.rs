pub mod common;

#[macro_export]
macro_rules! save_to_file {
    ( $n:expr ) => {
        let full_path = format!("{}/{}.csv", PREDICTORS_FOLDER, stringify!($n));
        {
            match write_csv(&full_path, &$n) {
                Ok(_) => println!("Successfully wrote CSV file"),
                Err(e) => println!("Error writing CSV file: {e:?}"),
            };
        }
    };
}
