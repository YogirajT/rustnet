#[cfg(test)]
mod tests {
    use rustnet::{
        common::{
            integration_test_vars::{
                get_b_1_test, get_b_2_test, get_image_label_test, get_image_test, get_w_1_test,
                get_w_2_test,
            },
            matrix::{
                dot_product, get_nth_column, linear_op, matrix_avg, matrix_max, matrix_min,
                matrix_multiply, matrix_subtract, row_sum, transpose, Operation,
            },
            network_functions::{
                back_propagation, forward_propagation, get_predictions, relu, softmax,
                transform_labels_to_network_output,
            },
        },
        NumpyVec,
    };

    #[test]
    fn test_dot_product() {
        let matrix1 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let expected = vec![vec![58.0, 64.0], vec![139.0, 154.0]];
        assert_eq!(dot_product(&matrix1, &matrix2), expected);

        let matrix3 = vec![vec![1.0, 2.0], vec![-0.1, 0.2]];
        let matrix4 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(
            dot_product(&matrix3, &matrix4),
            vec![vec![7., 10.], vec![0.5, 0.6]]
        );
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
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let expected = vec![
            vec![1.0, 4.0, 7.0],
            vec![2.0, 5.0, 8.0],
            vec![3.0, 6.0, 9.0],
        ];
        let result = transpose(&matrix);
        assert_eq!(result, expected);

        let matrix2 = vec![vec![1.0, 2.0, 3.0]];
        let expected2 = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result2 = transpose(&matrix2);
        assert_eq!(result2, expected2);

        let result3 = transpose(&expected2);
        assert_eq!(result3, matrix2);
    }

    #[test]
    fn test_get_nth_column() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = get_nth_column(&x, 0);
        assert_eq!(result, vec![1.0, 4.0]);
    }

    #[test]
    fn test_softmax() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = softmax(&x);
        assert_eq!(
            result,
            vec![
                vec![0.047425874, 0.047425874, 0.047425874],
                vec![0.95257413, 0.95257413, 0.95257413]
            ]
        );

        let y = vec![
            vec![-5.0, -9.0, -1.0],
            vec![-0.5, -0.1, -0.9],
            vec![1.0, 5.0, 9.0],
        ];

        assert_eq!(
            softmax(&y),
            vec![
                vec![0.002022466, 8.264891e-7, 4.539559e-5],
                vec![0.18205658, 0.006059794, 5.0169907e-5],
                vec![0.815921, 0.9939394, 0.9999044]
            ]
        );
    }

    #[test]
    fn test_get_predictions() {
        let x = vec![vec![1.0, 7.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = get_predictions(&x);
        assert_eq!(result, vec![1, 0, 1]);

        let matrix = vec![
            vec![1.0, 5.0, 3.0],
            vec![4.0, 6.0, 2.0],
            vec![7.0, 2.0, 9.0],
        ];
        assert_eq!(get_predictions(&matrix), vec![2, 1, 2]);
    }

    #[test]
    fn test_linear_op() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![1.0], vec![2.0]];
        let result = linear_op(Operation::Add, &x, &b);

        assert_eq!(result, vec![vec![2.0, 3.0, 4.0], vec![6.0, 7.0, 8.0]]);
    }

    #[test]
    fn test_matrix_subtract() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let y = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let result = matrix_subtract(&x, &y);

        assert_eq!(result, vec![vec![0.9, 1.8, 2.7], vec![3.6, 4.5, 5.4]]);
    }

    #[test]
    fn test_transform_labels_to_network_output() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let result = transform_labels_to_network_output(&x);

        assert_eq!(
            result,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ]
        );
    }

    #[test]
    fn test_row_sum() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = row_sum(&x);

        assert_eq!(result, vec![vec![6.0], vec![15.0]]);
    }

    #[test]
    fn test_matrix_multiply() {
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = matrix_multiply(&x, &x);

        assert_eq!(result, vec![vec![1.0, 4.0, 9.0], vec![16.0, 25.0, 36.0]]);
    }

    #[test]
    fn test_relu() {
        let x = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-0.1, -0.2, -0.3],
            vec![1.0, 2.0, 3.0],
            vec![0.1, 0.2, 0.3],
        ];
        let result = relu(&x);

        assert_eq!(
            result,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![1.0, 2.0, 3.0],
                vec![0.1, 0.2, 0.3]
            ]
        );
    }

    #[test]
    fn test_matrix_avg() {
        let x = vec![
            vec![5.0, 9.0, 13.0],
            vec![14.0, 10.0, 12.0],
            vec![11.0, 15.0, 19.0],
        ];
        let result = matrix_avg(&x);

        assert_eq!(result, 12.0);
    }

    #[test]
    fn test_forward_prop() {
        let (_z_1, _activation_1, _z_2, activation_2) = forward_propagation(
            (
                get_w_1_test(),
                get_b_1_test(),
                get_w_2_test(),
                get_b_2_test(),
            ),
            &get_image_test(),
        );

        let _activation_2_min = matrix_min(&activation_2);
        let _activation_2_max = matrix_max(&activation_2);

        assert_eq!(_activation_2_min, 0.00323996);
        assert_eq!(_activation_2_max, 0.53724295);
    }

    #[test]
    fn test_back_prop() {
        let (_z_1, _activation_1, _z_2, activation_2) = forward_propagation(
            (
                get_w_1_test(),
                get_b_1_test(),
                get_w_2_test(),
                get_b_2_test(),
            ),
            &get_image_test(),
        );

        let _activation_2_min = matrix_min(&activation_2);
        let _activation_2_max = matrix_max(&activation_2);

        assert_eq!(_activation_2_min, 0.00323996);
        assert_eq!(_activation_2_max, 0.53724295);

        let (delta_w_1, delta_b_1, delta_w_2, delta_b_2) = back_propagation(
            (_z_1, _activation_1, _z_2, activation_2),
            get_w_2_test(),
            get_image_label_test(),
            &get_image_test(),
        );

        let _delta_w_2_min = matrix_min(&delta_w_2);
        let _delta_w_2_max = matrix_max(&delta_w_2);

        assert_eq!(_delta_w_2_min, -4.479972);
        assert_eq!(_delta_w_2_max, 2.558414);

        let _delta_w_1_avg = matrix_avg(&delta_w_1);
        let _delta_w_1_min = matrix_min(&delta_w_1);
        let _delta_w_1_max = matrix_max(&delta_w_1);

        assert_eq!(_delta_w_1_min, -0.39199317);
        assert_eq!(_delta_w_1_max, 0.5273504);

        let _delta_b_1_avg = matrix_avg(&delta_b_1);
        let _delta_b_1_min = matrix_min(&delta_b_1);
        let _delta_b_1_max = matrix_max(&delta_b_1);

        assert_eq!(_delta_b_1_avg, 0.07719006);
        assert_eq!(_delta_b_1_min, -0.39199317);
        assert_eq!(_delta_b_1_max, 0.5273504);

        let _delta_b_2_avg = matrix_avg(&delta_b_2);
        let _delta_b_2_min = matrix_min(&delta_b_2);
        let _delta_b_2_max = matrix_max(&delta_b_2);

        assert_eq!(_delta_b_2_avg, 5.9604646e-9);
        assert_eq!(_delta_b_2_min, -0.9407521);
        assert_eq!(_delta_b_2_max, 0.53724295);
    }

    #[test]
    fn test_operator_overload() {
        let r1_1 = NumpyVec(vec![1.0, 2.0, 3.0]);
        let r1_2 = NumpyVec(vec![4.0, 5.0, 6.0]);

        let x = NumpyVec(vec![r1_1, r1_2]);

        let r2_1 = NumpyVec(vec![1.0, 2.0, 3.0]);
        let r2_2 = NumpyVec(vec![4.0, 5.0, 6.0]);

        let y = NumpyVec(vec![r2_1, r2_2]);

        let result = x + y;

        assert_eq!(result, vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]);
    }

    #[test]
    fn test_single_column_addition_operator_overload() {
        let r1_1 = NumpyVec(vec![1.0, 2.0, 3.0]);
        let r1_2 = NumpyVec(vec![4.0, 5.0, 6.0]);

        let x = NumpyVec(vec![r1_1, r1_2]);

        let r2_1 = NumpyVec(vec![1.0]);
        let r2_2 = NumpyVec(vec![2.0]);

        let y = NumpyVec(vec![r2_1, r2_2]);

        let result = x + y;

        assert_eq!(result, vec![vec![2.0, 3.0, 4.0], vec![6.0, 7.0, 8.0]]);
    }
}
