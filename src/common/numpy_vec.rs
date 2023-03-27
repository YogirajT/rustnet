use std::ops::{Add, Deref, DerefMut, Mul, Sub};

#[derive(Clone)]
pub struct NumpyVec<T>(pub Vec<T>);

impl<T: Add<Output = T> + Mul<Output = T> + Copy> NumpyVec<T> {}

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

impl<T: Add<Output = T> + Copy + 'static> Add<NumpyVec<Vec<T>>> for NumpyVec<Vec<T>>
where
    &'static T: std::ops::Add<T>,
{
    type Output = Vec<Vec<T>>;

    fn add(self, _rhs: NumpyVec<Vec<T>>) -> Self::Output {
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

impl<T: Sub<Output = T> + Copy + 'static> Sub<NumpyVec<Vec<T>>> for NumpyVec<Vec<T>>
where
    &'static T: std::ops::Sub<T>,
{
    type Output = Vec<Vec<T>>;

    fn sub(self, _rhs: NumpyVec<Vec<T>>) -> Self::Output {
        assert!(
            self.len() == _rhs.len()
                && (self.first().unwrap().len() == _rhs.first().unwrap().len()
                    || _rhs.first().unwrap().len() == 1),
            "Matrix dimensions must match for subtraction."
        );

        let mut result: Vec<Vec<T>> = Vec::new();

        for (i, row) in self.iter().enumerate() {
            let mut new_row = Vec::new();
            for (j, cell) in row.iter().enumerate() {
                if _rhs.first().unwrap().len() == 1 {
                    new_row.push(*cell - _rhs[i][0]);
                } else {
                    new_row.push(*cell - _rhs[i][j]);
                }
            }
            result.push(new_row);
        }
        result
    }
}

impl<T: Mul<Output = T> + Copy + 'static> Mul<NumpyVec<Vec<T>>> for NumpyVec<Vec<T>>
where
    &'static T: std::ops::Mul<T>,
{
    type Output = Vec<Vec<T>>;

    fn mul(self, _rhs: NumpyVec<Vec<T>>) -> Self::Output {
        assert!(
            self.len() == _rhs.len()
                && (self.first().unwrap().len() == _rhs.first().unwrap().len()
                    || _rhs.first().unwrap().len() == 1),
            "Matrix dimensions must match for multiplication."
        );

        let mut result: Vec<Vec<T>> = Vec::new();

        for (i, row) in self.iter().enumerate() {
            let mut new_row = Vec::new();
            for (j, cell) in row.iter().enumerate() {
                if _rhs.first().unwrap().len() == 1 {
                    new_row.push(*cell * _rhs[i][0]);
                } else {
                    new_row.push(*cell * _rhs[i][j]);
                }
            }
            result.push(new_row);
        }
        result
    }
}
