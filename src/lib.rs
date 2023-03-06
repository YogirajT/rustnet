pub mod common;
use std::ops::Add;
use std::ops::{Deref, DerefMut};
use std::vec::Vec;

pub struct NumpyVec<T>(pub Vec<T>);

impl<T: Add<Output = T>> NumpyVec<T> {
    pub fn new(data: Vec<T>) -> NumpyVec<T> {
        NumpyVec(data)
    }
}

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

// operator overloading
impl<T> Deref for NumpyVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for NumpyVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Add<Output = T> + Copy + 'static> Add<NumpyVec<NumpyVec<T>>> for NumpyVec<NumpyVec<T>>
where
    &'static T: std::ops::Add<T>,
{
    type Output = Vec<Vec<T>>;

    fn add(self, _rhs: NumpyVec<NumpyVec<T>>) -> Vec<Vec<T>> {
        assert!(
            self.len() == _rhs.len()
                && (self.first().unwrap().len() == _rhs.first().unwrap().len()
                    || _rhs.first().unwrap().len() == 1),
            "Matrix dimensions must match for addition."
        );

        let mut result: Vec<Vec<T>> = Vec::new();

        for (i, row) in self.iter().enumerate() {
            let mut new_row = Vec::new();
            for (j, cell) in row.iter().enumerate() {
                if _rhs.first().unwrap().len() == 1 {
                    new_row.push(*cell + _rhs[i][0]);
                } else {
                    new_row.push(*cell + _rhs[i][j]);
                }
            }
            result.push(new_row);
        }
        result
    }
}
