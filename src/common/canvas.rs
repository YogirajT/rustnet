#![allow(dead_code)]
use crate::common::console::draw;

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
    ControlFlow::Continue(()) // continues the loop
}

pub fn predict_hook(
    engine: &console_engine::ConsoleEngine,
    user_input: &[Vec<f32>],
    top_instructions: &mut String,
) {
    if engine.is_key_pressed(KeyCode::Char('s')) {
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

pub fn init_canvas() {
    println!("Predictors found, please use the terminal to draw a number");

    let max_px = 29;

    let mut engine = console_engine::ConsoleEngine::init_fill_require(max_px, max_px, 30).unwrap();

    // main loop, be aware that you'll have to break it because ctrl+C is captured
    let mut user_input: Vec<Vec<f32>> =
        vec![vec![0.0; max_px.try_into().unwrap()]; max_px.try_into().unwrap()];

    let bottom_instructions = "Controls e-CLEAR, q-QUIT".to_owned();

    let top_msg = "Controls p-PREDICT".to_owned();
    let mut top_instructions = top_msg.clone();

    loop {
        engine.wait_frame(); // wait for next frame + capture inputs
        engine.clear_screen(); // reset the screen

        if let ControlFlow::Break(_) = setup_canvas_controls(
            &engine,
            &mut user_input,
            &mut top_instructions,
            max_px,
            &top_msg,
        ) {
            break;
        }

        // draws the boundaries for canvas
        draw_canvas_bounds(&mut engine, max_px, &bottom_instructions, &top_instructions);

        setup_mouse_actions(&engine, &mut top_instructions, &top_msg, &mut user_input);

        draw(&mut engine, user_input.clone());

        engine.draw(); // draw the screen
    }
}
