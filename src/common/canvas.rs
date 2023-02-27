#![allow(dead_code)]
use super::network_functions::predict;
use console_engine::{pixel, Color, ConsoleEngine, KeyCode, MouseButton};
use std::ops::ControlFlow;

pub fn draw_canvas_bounds(
    engine: &mut console_engine::ConsoleEngine,
    max_px: u32,
    bottom_instructions: &str,
    top_instructions: &str,
) {
    engine.line(
        0,
        0,
        0,
        max_px.try_into().unwrap(),
        pixel::pxl_fg('*', Color::White),
    );
    engine.line(
        0,
        0,
        max_px.try_into().unwrap(),
        0,
        pixel::pxl_fg('*', Color::White),
    );
    engine.line(
        max_px as i32,
        0,
        max_px as i32,
        max_px as i32,
        pixel::pxl_fg('*', Color::White),
    );
    engine.line(
        0,
        max_px as i32,
        max_px as i32,
        max_px as i32,
        pixel::pxl_fg('*', Color::White),
    );

    engine.print(3, max_px.try_into().unwrap(), bottom_instructions); // prints some value at [0,4]
    engine.print(
        ((max_px - top_instructions.len() as u32) / 2)
            .try_into()
            .unwrap(),
        0,
        top_instructions,
    ); // prints some value at [0,4]
}

pub fn setup_canvas_controls(
    engine: &console_engine::ConsoleEngine,
    user_input: &mut Vec<Vec<f32>>,
    top_instructions: &mut String,
    max_px: u32,
    top_msg: &str,
) -> ControlFlow<()> {
    if engine.is_key_pressed(KeyCode::Char('q')) {
        // if the user presses 'q' :
        return ControlFlow::Break(()); // exits app
    }

    predict_hook(engine, user_input, top_instructions);

    if engine.is_key_pressed(KeyCode::Char('e')) {
        // erase the console
        *user_input = vec![vec![0.0; max_px.try_into().unwrap()]; max_px.try_into().unwrap()];
        *top_instructions = top_msg.to_owned();
    }
    ControlFlow::Continue(())
}

pub fn predict_hook(
    engine: &console_engine::ConsoleEngine,
    user_input: &[Vec<f32>],
    top_instructions: &mut String,
) {
    if engine.is_key_pressed(KeyCode::Char('p')) {
        // remove the first row and column of the canvas as it is 1 px larger than our input
        let prediction = predict(
            user_input[1..]
                .iter()
                .map(|row| row[1..].to_vec())
                .collect(),
        );
        *top_instructions = format!("Prediction:{prediction}");
    }
}

pub fn setup_mouse_actions(
    engine: &ConsoleEngine,
    top_instructions: &mut String,
    top_msg: &str,
    user_input: &mut [Vec<f32>],
) {
    let mouse_pos = engine.get_mouse_held(MouseButton::Left);

    if let Some(mouse_pos) = mouse_pos {
        *top_instructions = top_msg.to_owned();
        //prevent drawing on the edges of the box
        if mouse_pos.0 < 29 && mouse_pos.1 < 29 && mouse_pos.0 > 0 && mouse_pos.1 > 0 {
            user_input[mouse_pos.0 as usize][mouse_pos.1 as usize] = 0.9;
        }
    }
}
