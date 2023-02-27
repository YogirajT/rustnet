mod common;
use std::fs;
use std::ops::ControlFlow;

use dotenv::dotenv;
use rustnet::common::canvas::{draw_canvas_bounds, setup_canvas_controls, setup_mouse_actions};
use rustnet::common::console::draw;
use rustnet::common::constants::PREDICTORS_FOLDER;
use rustnet::common::io::{check_results_exist, read_file_into_vector};
use rustnet::common::network_functions::{prepare_data, train};
use rustnet::save_to_file;

fn main() {
    dotenv().ok();

    let bool = check_results_exist();

    match bool {
        true => {
            println!("Predictors found, please use the terminal to draw a number");

            let max_px = 29;

            let mut engine =
                console_engine::ConsoleEngine::init_fill_require(max_px, max_px, 30).unwrap();

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
        false => {
            println!("Predictors not found, re-training");

            let training_set = read_file_into_vector();

            let (train_labels, train_data) = prepare_data(training_set);

            let iterations = std::env::var("ITERATIONS")
                .expect("ITERATIONS must be set.")
                .parse::<usize>()
                .unwrap();

            let alpha = std::env::var("ALPHA")
                .expect("ALPHA must be set.")
                .parse::<f32>()
                .unwrap();

            let (w_1, b_1, w_2, b_2) = train(train_labels, train_data, iterations, alpha);

            fs::create_dir_all(PREDICTORS_FOLDER).expect("Already Exists");

            save_to_file!(w_1);
            save_to_file!(b_1);
            save_to_file!(w_2);
            save_to_file!(b_2);

            println!("Predictors generated, please rerun the program to launch prediction canvas");
        }
    }
}
