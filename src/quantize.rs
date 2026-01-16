use crate::config::Config;
use crate::error::Result;
use image::RgbaImage;
use palette::{white_point::D65, FromColor, Lab, Srgba};

use rayon::prelude::*;

/// Detects the number of unique colors in the image to suggest an optimal K for quantization.
pub fn auto_detect_k_colors(img: &RgbaImage) -> usize {
    let mut unique_colors = std::collections::HashSet::new();
    for p in img.pixels() {
        if p[3] > 0 {
            unique_colors.insert(p.0);
            if unique_colors.len() >= 32 {
                return 32;
            }
        }
    }
    unique_colors.len().clamp(1, 32)
}

/// Quantizes the image to a fixed number of colors using K-means clustering.
/// Uses Lab color space for better perceptual color selection.
/// If the image already has ≤K unique colors, K-means is bypassed for O(N) mapping.
pub fn quantize_image(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    let k_target = if config.k_colors == 0 {
        auto_detect_k_colors(img)
    } else {
        config.k_colors
    };

    // Fast path: Collect unique colors. If ≤ k_target, bypass K-means.
    let mut unique_colors = std::collections::HashMap::new();
    let mut color_count = 0usize;
    for p in img.pixels() {
        if p[3] > 0 {
            if !unique_colors.contains_key(&p.0) {
                unique_colors.insert(p.0, color_count);
                color_count += 1;
            }
            if color_count > k_target {
                break;
            }
        }
    }

    // If unique colors ≤ k_target, image is already "quantized"
    if color_count <= k_target && color_count > 0 {
        return Ok(img.clone());
    }

    let width = img.width() as usize;
    let height = img.height() as usize;
    let in_samples = img.as_flat_samples().samples;

    // Collect opaque pixel indices
    let opaque_indices: Vec<usize> = (0..width * height)
        .filter(|&i| in_samples[i * 4 + 3] > 0)
        .collect();

    if opaque_indices.is_empty() {
        return Ok(img.clone());
    }

    // Generate sRGB -> Linear LUT once
    static SRGB_LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    let lut = SRGB_LUT.get_or_init(|| {
        let mut table = [0.0; 256];
        for (i, val) in table.iter_mut().enumerate() {
            let s = i as f32 / 255.0;
            *val = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
        }
        table
    });

    // Convert to Lab in parallel
    let lab_pixels: Vec<Lab<D65, f32>> = opaque_indices
        .par_iter()
        .map(|&i| {
            let base = i * 4;
            let r = lut[in_samples[base] as usize];
            let g = lut[in_samples[base + 1] as usize];
            let b = lut[in_samples[base + 2] as usize];
            let a = in_samples[base + 3] as f32 / 255.0;

            let linear = palette::LinSrgba::new(r, g, b, a);
            Lab::from_color(linear)
        })
        .collect();

    let k = k_target.min(lab_pixels.len());
    let max_iter = config.max_kmeans_iterations;
    let converge = 0.01;
    let verbose = false;
    let seed = config.k_seed;

    // K-means clustering
    let result = kmeans_colors::get_kmeans_hamerly(k, max_iter, converge, verbose, &lab_pixels, seed);

    // Pre-compute centroid RGB values (avoid repeated conversion in loop)
    let centroid_rgba: Vec<[u8; 4]> = result.centroids
        .iter()
        .map(|&lab_c| {
            let srgba: Srgba = Srgba::from_color(lab_c);
            [
                (srgba.red * 255.0).round() as u8,
                (srgba.green * 255.0).round() as u8,
                (srgba.blue * 255.0).round() as u8,
                255, // Alpha handled separately
            ]
        })
        .collect();

    // Create output buffer - DIRECT WRITE instead of clone + scatter
    let mut out_samples = in_samples.to_vec(); // Single copy

    // Direct parallel write using index-based access (each write is to unique index)
    opaque_indices
        .par_iter()
        .enumerate()
        .for_each(|(lab_idx, &pixel_idx)| {
            let centroid_idx = result.indices[lab_idx] as usize;
            let rgba = &centroid_rgba[centroid_idx];
            let base = pixel_idx * 4;
            
            // Safe: each pixel_idx is unique, no data races
            unsafe {
                let ptr = out_samples.as_ptr() as *mut u8;
                *ptr.add(base) = rgba[0];
                *ptr.add(base + 1) = rgba[1];
                *ptr.add(base + 2) = rgba[2];
                // Alpha preserved from original (already in out_samples)
            }
        });

    // Construct image from buffer - no additional copy
    RgbaImage::from_raw(width as u32, height as u32, out_samples)
        .ok_or_else(|| crate::error::PixelSnapperError::ProcessingError(
            "Failed to create output image".to_string()
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use image::Rgba;

    #[test]
    fn test_quantize_image_auto_colors() {
        let img = RgbaImage::new(10, 10);
        let config = Config {
            k_colors: 0,
            ..Config::default()
        };
        let result = quantize_image(&img, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantize_image_empty_image() {
        let img = RgbaImage::new(10, 10);
        let config = Config::default();
        let result = quantize_image(&img, &config).unwrap();
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_quantize_image_single_color() {
        let mut img = RgbaImage::new(10, 10);
        for p in img.pixels_mut() {
            *p = Rgba([255, 0, 0, 255]);
        }
        let config = Config {
            k_colors: 1,
            ..Config::default()
        };
        let result = quantize_image(&img, &config).unwrap();
        let p = result.get_pixel(0, 0);
        assert!(p[0] > 250);
        assert!(p[1] < 10);
        assert!(p[2] < 10);
        assert_eq!(p[3], 255);
    }
}
