mod common;
use std::fs;

use console_engine::{pixel, Color, KeyCode, MouseButton};
use dotenv::dotenv;
use rustnet::common::console::draw;
use rustnet::common::constants::PREDICTORS_FOLDER;
use rustnet::common::io::{check_results_exist, read_file_into_vector};
use rustnet::common::network_functions::{predict, prepare_data, train};
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

                if engine.is_key_pressed(KeyCode::Char('q')) {
                    // if the user presses 'q' :
                    break; // exits app
                }

                if engine.is_key_pressed(KeyCode::Char('p')) {
                    let prediction = predict(
                        user_input[1..]
                            .iter()
                            .map(|row| row[1..].to_vec())
                            .collect(),
                    );
                    top_instructions = format!("Prediction:{prediction}");
                }

                if engine.is_key_pressed(KeyCode::Char('e')) {
                    // erase the console
                    user_input =
                        vec![vec![0.0; max_px.try_into().unwrap()]; max_px.try_into().unwrap()];
                    top_instructions = top_msg.clone();
                }

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

                engine.print(3, max_px.try_into().unwrap(), &bottom_instructions); // prints some value at [0,4]
                engine.print(
                    ((max_px - top_instructions.len() as u32) / 2)
                        .try_into()
                        .unwrap(),
                    0,
                    &top_instructions,
                ); // prints some value at [0,4]

                let mouse_pos = engine.get_mouse_held(MouseButton::Left);

                if let Some(mouse_pos) = mouse_pos {
                    top_instructions = top_msg.clone();
                    if mouse_pos.0 < 29 && mouse_pos.1 < 29 && mouse_pos.0 > 0 && mouse_pos.1 > 0 {
                        user_input[mouse_pos.0 as usize][mouse_pos.1 as usize] = 1.0;
                    }
                }

                draw(&mut engine, user_input.clone());

                engine.draw(); // draw the screen
            }
        }
        false => {
            println!("Predictors not found, retraining");

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

            println!(
                "Predictors generated, please rerun the program to launch console draw-and-predict"
            );
        }
    }
}
