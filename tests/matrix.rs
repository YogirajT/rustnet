#[cfg(test)]
mod tests {
    use rustnet::common::matrix::{dot_product, transpose};

    #[test]
    fn test_dot_product() {
        let matrix1 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let expected = vec![vec![58.0, 64.0], vec![139.0, 154.0]];
        assert_eq!(dot_product(&matrix1, &matrix2), expected);
    }

    #[test]
    #[should_panic(
        expected = "The number of columns in the first matrix must be equal to the number of rows in the second matrix"
    )]
    fn test_dot_product_invalid_dimensions() {
        let matrix1 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix2 = vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
            vec![13.0, 14.0],
        ];
        dot_product(&matrix1, &matrix2);
    }

    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let expected = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];
        let result = transpose(&matrix);
        assert_eq!(result, expected);
    }
}
