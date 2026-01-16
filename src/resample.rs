use crate::error::{PixelSnapperError, Result};
use image::{ImageBuffer, RgbaImage};
use rayon::prelude::*;
use std::collections::HashMap;

/// Maximum palette size for fixed-array optimization
const MAX_PALETTE: usize = 64;

pub fn resample(img: &RgbaImage, cols: &[usize], rows: &[usize]) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len().max(1) - 1) as u32;
    let out_h = (rows.len().max(1) - 1) as u32;

    let in_samples = img.as_flat_samples().samples;
    let in_width = img.width() as usize;
    let in_height = img.height() as usize;
    let in_stride = in_width * 4;

    // Build palette and pre-index the entire image (single pass)
    let mut color_to_idx: HashMap<[u8; 4], u8> = HashMap::with_capacity(MAX_PALETTE);
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(MAX_PALETTE);
    
    // Pre-indexed image: each pixel maps to its palette index (or 255 for transparent)
    let mut indexed_img: Vec<u8> = vec![255u8; in_width * in_height];
    
    for y in 0..in_height {
        for x in 0..in_width {
            let base = y * in_stride + x * 4;
            let a = in_samples[base + 3];
            if a > 0 {
                let pixel = [in_samples[base], in_samples[base + 1], in_samples[base + 2], a];
                let idx = if let Some(&existing) = color_to_idx.get(&pixel) {
                    existing
                } else {
                    let new_idx = palette.len() as u8;
                    if palette.len() < MAX_PALETTE {
                        color_to_idx.insert(pixel, new_idx);
                        palette.push(pixel);
                        new_idx
                    } else {
                        255 // Overflow - treat as transparent for counting purposes
                    }
                };
                indexed_img[y * in_width + x] = idx;
            }
        }
    }
    
    let palette_size = palette.len();

    let mut final_img: RgbaImage = ImageBuffer::new(out_w, out_h);

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
                    // 1:1 fast path - direct lookup
                    if xs < in_width && ys < in_height {
                        let idx = indexed_img[ys * in_width + xs];
                        if idx < palette_size as u8 { palette[idx as usize] } else { [0,0,0,0] }
                    } else {
                        [0, 0, 0, 0]
                    }
                } else {
                    // ZERO-ALLOCATION: Fixed array counting using pre-indexed image
                    let mut counts = [0u32; MAX_PALETTE];
                    
                    for y in ys..ye.min(in_height) {
                        let row_offset = y * in_width;
                        for x in xs..xe.min(in_width) {
                            let color_idx = indexed_img[row_offset + x];
                            if color_idx < palette_size as u8 {
                                counts[color_idx as usize] += 1;
                            }
                        }
                    }

                    // Find mode in O(K)
                    let mut best_idx = 0;
                    let mut max_count = 0u32;
                    for (i, &count) in counts[..palette_size].iter().enumerate() {
                        if count > max_count {
                            max_count = count;
                            best_idx = i;
                        }
                    }
                    
                    if max_count > 0 { palette[best_idx] } else { [0, 0, 0, 0] }
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
        for p in img.pixels_mut() {
            *p = image::Rgba([255, 0, 0, 255]);
        }
        let result = resample(&img, &vec![0, 10], &vec![0, 10]).unwrap();
        assert_eq!(result.dimensions(), (1, 1));
        assert_eq!(result.get_pixel(0, 0), &image::Rgba([255, 0, 0, 255]));
    }
}
