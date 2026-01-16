use crate::error::{PixelSnapperError, Result};
use image::{ImageBuffer, RgbaImage};
use rayon::prelude::*;
use std::collections::HashMap;

pub fn resample(img: &RgbaImage, cols: &[usize], rows: &[usize]) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len().max(1) - 1) as u32;
    let out_h = (rows.len().max(1) - 1) as u32;

    let mut final_img: RgbaImage = ImageBuffer::new(out_w, out_h);

    // Pre-compute input buffer access for performance
    let in_samples = img.as_flat_samples().samples;
    let in_width = img.width() as usize;
    let in_stride = in_width * 4;

    {
        let w = out_w;
        let samples = final_img.as_flat_samples_mut().samples;

        samples
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(idx, pixel_sample)| {
                let x_i = (idx as u32 % w) as usize;
                let y_i = (idx as u32 / w) as usize;

                let ys = rows[y_i];
                let ye = rows[y_i + 1];
                let xs = cols[x_i];
                let xe = cols[x_i + 1];

                let best_pixel = if xe <= xs || ye <= ys {
                    [0, 0, 0, 0]
                } else if xe - xs == 1 && ye - ys == 1 {
                    // Extreme fast path for 1:1 mapped cells
                    if xs < in_width && ys < img.height() as usize {
                        let base = ys * in_stride + xs * 4;
                        unsafe {
                            let ptr = in_samples.as_ptr();
                            [*ptr.add(base), *ptr.add(base+1), *ptr.add(base+2), *ptr.add(base+3)]
                        }
                    } else {
                        [0, 0, 0, 0]
                    }
                } else {
                    // HashMap-based counting for cells
                    // For small K (typical pixel art), this is very fast
                    let mut counts: HashMap<[u8; 4], u32> = HashMap::with_capacity(16);

                    for y in ys..ye {
                        let y_offset = y * in_stride;
                        for x in xs..xe {
                            let base = y_offset + x * 4;
                            unsafe {
                                let ptr = in_samples.as_ptr();
                                let a = *ptr.add(base + 3);
                                if a > 0 {
                                    let key = [*ptr.add(base), *ptr.add(base+1), *ptr.add(base+2), a];
                                    *counts.entry(key).or_insert(0) += 1;
                                }
                            }
                        }
                    }

                    // Find mode (most frequent color)
                    let mut best_p = [0u8; 4];
                    let mut max_count = 0u32;
                    for (&color, &count) in counts.iter() {
                        if count > max_count {
                            max_count = count;
                            best_p = color;
                        }
                    }
                    best_p
                };

                pixel_sample[0] = best_pixel[0];
                pixel_sample[1] = best_pixel[1];
                pixel_sample[2] = best_pixel[2];
                pixel_sample[3] = best_pixel[3];
            });
    }

    Ok(final_img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_insufficient_grid() {
        let img = RgbaImage::new(10, 10);
        let result = resample(&img, &vec![0], &vec![0, 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_simple_grid() {
        let mut img = RgbaImage::new(10, 10);
        // Fill the entire image with red
        for p in img.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let result = resample(&img, &vec![0, 10], &vec![0, 10]).unwrap();
        assert_eq!(result.dimensions(), (1, 1));
        assert_eq!(result.get_pixel(0, 0), &image::Rgba([255, 0, 0, 255]));
    }
}
