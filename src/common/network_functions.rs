use super::{matrix::dot_product, types::NetworkParams};

#[allow(dead_code)]
fn relu(input: &[f64]) -> Vec<f64> {
    input
        .iter()
        .map(|x| if *x > 0.0 { *x } else { 0.0 })
        .collect()
}

#[allow(dead_code)]
pub fn softmax(input: &[f64]) -> Vec<f64> {
    let max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let denominator = input.iter().cloned().map(|x| (x - max).exp()).sum::<f64>();
    input
        .iter()
        .map(|x| (*x - max).exp() / denominator)
        .collect()
}

#[allow(dead_code)]
pub fn forward_propagation(
    network_params: NetworkParams,
    input_image: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let (w_1, _b_1, _w_2, _b_2) = network_params;

    dot_product(&w_1, &input_image)
}
