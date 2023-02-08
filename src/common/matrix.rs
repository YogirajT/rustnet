use csv::StringRecord;

pub trait DotProduct {
    fn dot_product(&self, other: &Self) -> Self;
}

#[derive(Debug, PartialEq)]
pub struct IMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>,
}

impl DotProduct for IMatrix {
    fn dot_product(&self, other: &Self) -> Self {
        if self.cols != other.rows {
            panic!("The number of columns in the first matrix must be equal to the number of rows in the second matrix!");
        }
        let mut result = vec![vec![0.0; other.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        IMatrix {
            rows: self.rows,
            cols: other.cols,
            data: result,
        }
    }
}

pub fn create_vec_from_csv(rdr: StringRecord) -> Vec<Vec<f64>> {
    let mut vec = vec![];
    for result in rdr.records() {
        let record = result.unwrap();
        vec.push(
            record
                .iter()
                .map(|field| field.parse::<f64>().unwrap())
                .collect(),
        );
    }
    vec
}
