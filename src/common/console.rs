#![allow(dead_code)]

use console_engine::{pixel, screen::Screen};

pub fn draw(digit: &[f32]) {
    let sqrt = (digit.len() as f32).sqrt() as usize;

    let mut scr = Screen::new(sqrt as u32, sqrt as u32);

    for i in 0..sqrt {
        for j in 0..sqrt {
            // flip and rorate image changed due to transpose;

            let row_index = sqrt - 1 - j;
            let col_index = i * sqrt;

            if digit[row_index + col_index] > 100.0 {
                scr.set_pxl(sqrt as i32 - 1 - j as i32, i as i32, pixel::pxl('#'));
            }
        }
    }

    scr.draw()
}
