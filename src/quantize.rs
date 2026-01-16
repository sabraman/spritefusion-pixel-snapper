use crate::config::Config;
use crate::error::Result;
use image::RgbaImage;
use palette::Srgb;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Maximum palette size
pub const MAX_PALETTE: usize = 64;

/// Quantized image with pre-built palette and indexed buffer for efficient resampling
pub struct QuantizedImage {
    pub img: RgbaImage,
    pub palette: Vec<[u8; 4]>,
    pub indexed: Vec<u8>, // Each pixel maps to palette index (255 = transparent)
}

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

/// Quantizes the image and returns a QuantizedImage with pre-built palette and indexed buffer.
/// This eliminates the need for resample to rebuild the palette.
pub fn quantize_image(img: &RgbaImage, config: &Config) -> Result<QuantizedImage> {
    let k_target = if config.k_colors == 0 {
        auto_detect_k_colors(img)
    } else {
        config.k_colors
    };

    let width = img.width() as usize;
    let height = img.height() as usize;
    let in_samples = img.as_flat_samples().samples;
    let in_stride = width * 4;

    // Check for custom palette
    if let Some(palette_path) = &config.palette {
        let p_img = image::open(palette_path)
            .map_err(|e| {
                crate::error::PixelSnapperError::ProcessingError(format!(
                    "Failed to load palette: {}",
                    e
                ))
            })?
            .to_rgba8();

        let mut unique_colors = std::collections::HashSet::new();
        let mut palette_vec = Vec::new();

        for p in p_img.pixels() {
            if p[3] > 128 {
                // Ignore transparent pixels
                // Check if color already exists
                if unique_colors.insert(p.0) {
                    palette_vec.push(p.0);
                    if palette_vec.len() >= MAX_PALETTE {
                        break;
                    }
                }
            }
        }

        if palette_vec.is_empty() {
            return Err(crate::error::PixelSnapperError::ProcessingError(
                "Palette image contains no opaque pixels".to_string(),
            ));
        }

        // Map pixels to nearest palette color
        let mut indexed = vec![255u8; width * height];

        // Integer-only palette for fast distance calculation
        let palette_int: Vec<[i32; 3]> = palette_vec
            .iter()
            .map(|p| [p[0] as i32, p[1] as i32, p[2] as i32])
            .collect();

        // Parallel processing for nearest neighbor mapping
        #[cfg(not(target_arch = "wasm32"))]
        let chunks = indexed.par_chunks_mut(width);
        #[cfg(target_arch = "wasm32")]
        let chunks = indexed.chunks_mut(width);

        chunks.enumerate().for_each(|(y, row_indices)| {
            for x in 0..width {
                let base = y * in_stride + x * 4;
                let a = in_samples[base + 3];
                if a > 0 {
                    let r = in_samples[base] as i32;
                    let g = in_samples[base + 1] as i32;
                    let b = in_samples[base + 2] as i32;

                    // Find nearest
                    let mut min_dist = i32::MAX;
                    let mut best_idx = 0;

                    for (i, p_color) in palette_int.iter().enumerate() {
                        // Integer squared Euclidean distance
                        let dr = r - p_color[0];
                        let dg = g - p_color[1];
                        let db = b - p_color[2];
                        let dist = dr * dr + dg * dg + db * db;
                        if dist < min_dist {
                            min_dist = dist;
                            best_idx = i;
                        }
                    }

                    row_indices[x] = best_idx as u8;
                }
            }
        });

        #[cfg(not(target_arch = "wasm32"))]
        let out_samples_iter = indexed.par_iter();
        #[cfg(target_arch = "wasm32")]
        let out_samples_iter = indexed.iter();

        // Write output pixels based on indexed
        let out_samples: Vec<u8> = out_samples_iter
            .map(|&idx| {
                if idx == 255 {
                    // Transparent
                    vec![0, 0, 0, 0]
                } else {
                    let p = palette_vec[idx as usize];
                    vec![p[0], p[1], p[2], 255]
                }
            })
            .flatten()
            .collect();

        return Ok(QuantizedImage {
            img: RgbaImage::from_raw(width as u32, height as u32, out_samples).unwrap(),
            palette: palette_vec,
            indexed,
        });
    }

    // Fast path: Build palette and indexed buffer directly for low-color images
    let mut color_to_idx = std::collections::HashMap::with_capacity(MAX_PALETTE);
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(MAX_PALETTE);
    let mut indexed = vec![255u8; width * height];
    let mut exceeded_k = false;

    for y in 0..height {
        for x in 0..width {
            let base = y * in_stride + x * 4;
            let a = in_samples[base + 3];
            if a > 0 {
                let pixel = [
                    in_samples[base],
                    in_samples[base + 1],
                    in_samples[base + 2],
                    a,
                ];
                let idx = if let Some(&existing) = color_to_idx.get(&pixel) {
                    existing
                } else {
                    if palette.len() >= MAX_PALETTE || palette.len() > k_target {
                        exceeded_k = true;
                        255
                    } else {
                        let new_idx = palette.len() as u8;
                        color_to_idx.insert(pixel, new_idx);
                        palette.push(pixel);
                        new_idx
                    }
                };
                indexed[y * width + x] = idx;
            }
        }
        if exceeded_k && palette.len() > k_target {
            break; // Need K-means, stop early
        }
    }

    // If unique colors ≤ k_target, we're done - no K-means needed
    if !exceeded_k || palette.len() <= k_target {
        return Ok(QuantizedImage {
            img: img.clone(),
            palette,
            indexed,
        });
    }

    // Hybrid Strategy with Reservoir Sampling:
    // 1. Single-pass reservoir sampling to collect training pixels (no opaque_indices allocation).
    // 2. Run K-Means on Sample to get Centroids.
    // 3. Parallel SIMD Map ALL pixels to Centroids.

    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.k_seed);

    let max_samples = 4096;
    let mut training_pixels: Vec<Srgb> = Vec::with_capacity(max_samples);
    let mut opaque_count: usize = 0;

    // Reservoir Sampling: Single pass, O(1) extra allocation beyond training_pixels
    for i in 0..(width * height) {
        let base = i * 4;
        if in_samples[base + 3] > 0 {
            // Opaque pixel
            opaque_count += 1;

            if training_pixels.len() < max_samples {
                // Fill the reservoir
                training_pixels.push(Srgb::new(
                    in_samples[base] as f32 / 255.0,
                    in_samples[base + 1] as f32 / 255.0,
                    in_samples[base + 2] as f32 / 255.0,
                ));
            } else {
                // Reservoir replacement with probability max_samples / opaque_count
                let j = rng.gen_range(0..opaque_count);
                if j < max_samples {
                    training_pixels[j] = Srgb::new(
                        in_samples[base] as f32 / 255.0,
                        in_samples[base + 1] as f32 / 255.0,
                        in_samples[base + 2] as f32 / 255.0,
                    );
                }
            }
        }
    }

    if training_pixels.is_empty() {
        return Ok(QuantizedImage {
            img: img.clone(),
            palette: vec![],
            indexed: vec![255u8; width * height],
        });
    }

    // println!("DEBUG: Training Pixels: {}, Total Opaque: {}", training_pixels.len(), opaque_indices.len());

    let k = k_target.min(training_pixels.len());
    // let start_kmeans = std::time::Instant::now();
    let result = kmeans_colors::get_kmeans_hamerly(
        k,
        config.max_kmeans_iterations,
        0.005,
        false,
        &training_pixels,
        config.k_seed,
    );
    // println!("DEBUG: K-Means Training took: {:?}", start_kmeans.elapsed());

    // Build palette from centroids
    let palette_vec: Vec<[u8; 4]> = result
        .centroids
        .iter()
        .map(|&c| {
            [
                (c.red * 255.0).round() as u8,
                (c.green * 255.0).round() as u8,
                (c.blue * 255.0).round() as u8,
                255,
            ]
        })
        .collect();

    // 2. Parallel SIMD Map
    // Prepare SIMD Palette (SoA) for integer dist calc
    let mut palette_r = Vec::with_capacity(palette_vec.len());
    let mut palette_g = Vec::with_capacity(palette_vec.len());
    let mut palette_b = Vec::with_capacity(palette_vec.len());

    for p in &palette_vec {
        palette_r.push(p[0] as i32);
        palette_g.push(p[1] as i32);
        palette_b.push(p[2] as i32);
    }

    // Pad to multiple of 8 for SIMD load safety
    while palette_r.len() % 8 != 0 {
        palette_r.push(10000); // Massive distance to ensure not chosen
        palette_g.push(10000);
        palette_b.push(10000);
    }

    // Pre-compute SIMD palette chunks
    let mut palette_chunks_r = Vec::new();
    let mut palette_chunks_g = Vec::new();
    let mut palette_chunks_b = Vec::new();

    let mut j = 0;
    while j < palette_r.len() {
        palette_chunks_r.push(i32x8::new([
            palette_r[j],
            palette_r[j + 1],
            palette_r[j + 2],
            palette_r[j + 3],
            palette_r[j + 4],
            palette_r[j + 5],
            palette_r[j + 6],
            palette_r[j + 7],
        ]));
        palette_chunks_g.push(i32x8::new([
            palette_g[j],
            palette_g[j + 1],
            palette_g[j + 2],
            palette_g[j + 3],
            palette_g[j + 4],
            palette_g[j + 5],
            palette_g[j + 6],
            palette_g[j + 7],
        ]));
        palette_chunks_b.push(i32x8::new([
            palette_b[j],
            palette_b[j + 1],
            palette_b[j + 2],
            palette_b[j + 3],
            palette_b[j + 4],
            palette_b[j + 5],
            palette_b[j + 6],
            palette_b[j + 7],
        ]));
        j += 8;
    }

    use wide::{i32x4, i32x8, CmpEq, CmpLt};
    let mut indexed = vec![255u8; width * height];

    // Use a fixed chunk size to ensure enough work per thread
    let chunk_size = 4096;
    #[cfg(not(target_arch = "wasm32"))]
    indexed
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, indices)| {
            let start_pixel = chunk_idx * chunk_size;
            let palette_len = palette_vec.len();

            // Process 4 pixels at a time for ILP
            let limit_4 = indices.len() - (indices.len() % 4);
            let mut i = 0;

            while i < limit_4 {
                let pixel_idx_base = start_pixel + i;
                if pixel_idx_base + 3 >= in_samples.len() / 4 {
                    break;
                }

                // Load 4 pixels (deinterleave RGBA -> R4, G4, B4, A4)
                // SAFETY: bounds checked above
                let (r4, g4, b4, a4) = unsafe {
                    let b0 = (pixel_idx_base) * 4;
                    let b1 = (pixel_idx_base + 1) * 4;
                    let b2 = (pixel_idx_base + 2) * 4;
                    let b3 = (pixel_idx_base + 3) * 4;
                    (
                        i32x4::new([
                            *in_samples.get_unchecked(b0) as i32,
                            *in_samples.get_unchecked(b1) as i32,
                            *in_samples.get_unchecked(b2) as i32,
                            *in_samples.get_unchecked(b3) as i32,
                        ]),
                        i32x4::new([
                            *in_samples.get_unchecked(b0 + 1) as i32,
                            *in_samples.get_unchecked(b1 + 1) as i32,
                            *in_samples.get_unchecked(b2 + 1) as i32,
                            *in_samples.get_unchecked(b3 + 1) as i32,
                        ]),
                        i32x4::new([
                            *in_samples.get_unchecked(b0 + 2) as i32,
                            *in_samples.get_unchecked(b1 + 2) as i32,
                            *in_samples.get_unchecked(b2 + 2) as i32,
                            *in_samples.get_unchecked(b3 + 2) as i32,
                        ]),
                        i32x4::new([
                            *in_samples.get_unchecked(b0 + 3) as i32,
                            *in_samples.get_unchecked(b1 + 3) as i32,
                            *in_samples.get_unchecked(b2 + 3) as i32,
                            *in_samples.get_unchecked(b3 + 3) as i32,
                        ]),
                    )
                };

                // Initialize min_dist and best_idx for all 4 pixels
                let mut min_dists = i32x4::splat(i32::MAX);
                let mut best_idxs = i32x4::splat(0);

                // Iterate through palette colors (scalar index, broadcast to 4 pixels)
                for p_idx in 0..palette_len {
                    // Broadcast palette color to all 4 lanes
                    let pr = i32x4::splat(palette_r[p_idx]);
                    let pg = i32x4::splat(palette_g[p_idx]);
                    let pb = i32x4::splat(palette_b[p_idx]);

                    let dr = r4 - pr;
                    let dg = g4 - pg;
                    let db = b4 - pb;

                    let dist = (dr * dr) + (dg * dg) + (db * db);

                    // Compare: is this distance smaller?
                    let mask = dist.cmp_lt(min_dists);

                    // Blend: update where mask is true
                    min_dists = mask.blend(dist, min_dists);
                    best_idxs = mask.blend(i32x4::splat(p_idx as i32), best_idxs);
                }

                // Branchless transparency: if alpha == 0, force index to 255
                let transparent_mask = a4.cmp_eq(i32x4::splat(0));
                let final_idxs: [i32; 4] =
                    transparent_mask.blend(i32x4::splat(255), best_idxs).into();

                // Write results
                for j in 0..4 {
                    let idx = final_idxs[j] as u8;
                    indices[i + j] = idx;
                }

                i += 4;
            }

            // Scalar fallback for remaining pixels (with RLE cache)
            let mut prev_pixel_val: u32 = u32::MAX;
            let mut prev_best_idx: u8 = 0;

            while i < indices.len() {
                let pixel_idx = start_pixel + i;
                if pixel_idx >= in_samples.len() / 4 {
                    break;
                }

                let base = pixel_idx * 4;

                let pixel_val = unsafe {
                    let ptr = in_samples.as_ptr().add(base);
                    *(ptr as *const u32)
                };

                let a = in_samples[base + 3];
                if a == 0 {
                    prev_pixel_val = pixel_val;
                    prev_best_idx = 255;
                    indices[i] = 255;
                    i += 1;
                    continue;
                }

                if pixel_val == prev_pixel_val {
                    indices[i] = prev_best_idx;
                    i += 1;
                    continue;
                }

                let r = in_samples[base] as i32;
                let g = in_samples[base + 1] as i32;
                let b = in_samples[base + 2] as i32;

                let r_cur = i32x8::splat(r);
                let g_cur = i32x8::splat(g);
                let b_cur = i32x8::splat(b);

                let mut min_dist = i32::MAX;
                let mut best_idx = 0;

                let mut chunk_idx_inner = 0;
                for k in 0..palette_chunks_r.len() {
                    let p_r = palette_chunks_r[k];
                    let p_g = palette_chunks_g[k];
                    let p_b = palette_chunks_b[k];

                    let dr = r_cur - p_r;
                    let dg = g_cur - p_g;
                    let db = b_cur - p_b;

                    let dist_vec = (dr * dr) + (dg * dg) + (db * db);
                    let dist_arr: [i32; 8] = dist_vec.into();

                    for (l, &d) in dist_arr.iter().enumerate() {
                        if d < min_dist {
                            min_dist = d;
                            best_idx = chunk_idx_inner + l;
                        }
                    }
                    chunk_idx_inner += 8;
                }

                if best_idx >= palette_len {
                    best_idx = 0;
                }

                prev_pixel_val = pixel_val;
                prev_best_idx = best_idx as u8;
                indices[i] = prev_best_idx;
                i += 1;
            }
        });

    #[cfg(target_arch = "wasm32")]
    indexed
        .chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, indices)| {
            let start_pixel = chunk_idx * chunk_size;
            let palette_len = palette_vec.len();

            // Process 4 pixels at a time for ILP
            let limit_4 = indices.len() - (indices.len() % 4);
            let mut i = 0;

            while i < limit_4 {
                let pixel_idx_base = start_pixel + i;
                if pixel_idx_base + 3 >= in_samples.len() / 4 {
                    break;
                }

                // Load 4 pixels (deinterleave RGBA -> R4, G4, B4, A4)
                // SAFETY: bounds checked above
                let (r4, g4, b4, a4) = unsafe {
                    let b0 = (pixel_idx_base) * 4;
                    let b1 = (pixel_idx_base + 1) * 4;
                    let b2 = (pixel_idx_base + 2) * 4;
                    let b3 = (pixel_idx_base + 3) * 4;
                    (
                        i32x4::new([
                            *in_samples.get_unchecked(b0) as i32,
                            *in_samples.get_unchecked(b1) as i32,
                            *in_samples.get_unchecked(b2) as i32,
                            *in_samples.get_unchecked(b3) as i32,
                        ]),
                        i32x4::new([
                            *in_samples.get_unchecked(b0 + 1) as i32,
                            *in_samples.get_unchecked(b1 + 1) as i32,
                            *in_samples.get_unchecked(b2 + 1) as i32,
                            *in_samples.get_unchecked(b3 + 1) as i32,
                        ]),
                        i32x4::new([
                            *in_samples.get_unchecked(b0 + 2) as i32,
                            *in_samples.get_unchecked(b1 + 2) as i32,
                            *in_samples.get_unchecked(b2 + 2) as i32,
                            *in_samples.get_unchecked(b3 + 2) as i32,
                        ]),
                        i32x4::new([
                            *in_samples.get_unchecked(b0 + 3) as i32,
                            *in_samples.get_unchecked(b1 + 3) as i32,
                            *in_samples.get_unchecked(b2 + 3) as i32,
                            *in_samples.get_unchecked(b3 + 3) as i32,
                        ]),
                    )
                };

                // Initialize min_dist and best_idx for all 4 pixels
                let mut min_dists = i32x4::splat(i32::MAX);
                let mut best_idxs = i32x4::splat(0);

                // Iterate through palette colors (scalar index, broadcast to 4 pixels)
                for p_idx in 0..palette_len {
                    // Broadcast palette color to all 4 lanes
                    let pr = i32x4::splat(palette_r[p_idx]);
                    let pg = i32x4::splat(palette_g[p_idx]);
                    let pb = i32x4::splat(palette_b[p_idx]);

                    let dr = r4 - pr;
                    let dg = g4 - pg;
                    let db = b4 - pb;

                    let dist = (dr * dr) + (dg * dg) + (db * db);

                    // Compare: is this distance smaller?
                    let mask = dist.cmp_lt(min_dists);

                    // Blend: update where mask is true
                    min_dists = mask.blend(dist, min_dists);
                    best_idxs = mask.blend(i32x4::splat(p_idx as i32), best_idxs);
                }

                // Branchless transparency: if alpha == 0, force index to 255
                let transparent_mask = a4.cmp_eq(i32x4::splat(0));
                let final_idxs: [i32; 4] =
                    transparent_mask.blend(i32x4::splat(255), best_idxs).into();

                // Write results
                for j in 0..4 {
                    let idx = final_idxs[j] as u8;
                    indices[i + j] = idx;
                }

                i += 4;
            }

            // Scalar fallback for remaining pixels (with RLE cache)
            let mut prev_pixel_val: u32 = u32::MAX;
            let mut prev_best_idx: u8 = 0;

            while i < indices.len() {
                let pixel_idx = start_pixel + i;
                if pixel_idx >= in_samples.len() / 4 {
                    break;
                }

                let base = pixel_idx * 4;

                let pixel_val = unsafe {
                    let ptr = in_samples.as_ptr().add(base);
                    *(ptr as *const u32)
                };

                let a = in_samples[base + 3];
                if a == 0 {
                    prev_pixel_val = pixel_val;
                    prev_best_idx = 255;
                    indices[i] = 255;
                    i += 1;
                    continue;
                }

                if pixel_val == prev_pixel_val {
                    indices[i] = prev_best_idx;
                    i += 1;
                    continue;
                }

                let r = in_samples[base] as i32;
                let g = in_samples[base + 1] as i32;
                let b = in_samples[base + 2] as i32;

                let r_cur = i32x8::splat(r);
                let g_cur = i32x8::splat(g);
                let b_cur = i32x8::splat(b);

                let mut min_dist = i32::MAX;
                let mut best_idx = 0;

                let mut chunk_idx_inner = 0;
                for k in 0..palette_chunks_r.len() {
                    let p_r = palette_chunks_r[k];
                    let p_g = palette_chunks_g[k];
                    let p_b = palette_chunks_b[k];

                    let dr = r_cur - p_r;
                    let dg = g_cur - p_g;
                    let db = b_cur - p_b;

                    let dist_vec = (dr * dr) + (dg * dg) + (db * db);
                    let dist_arr: [i32; 8] = dist_vec.into();

                    for (l, &d) in dist_arr.iter().enumerate() {
                        if d < min_dist {
                            min_dist = d;
                            best_idx = chunk_idx_inner + l;
                        }
                    }
                    chunk_idx_inner += 8;
                }

                if best_idx >= palette_len {
                    best_idx = 0;
                }

                prev_pixel_val = pixel_val;
                prev_best_idx = best_idx as u8;
                indices[i] = prev_best_idx;
                i += 1;
            }
        });

    #[cfg(not(target_arch = "wasm32"))]
    let out_samples_iter = indexed.par_iter();
    #[cfg(target_arch = "wasm32")]
    let out_samples_iter = indexed.iter();

    // Write output pixels based on indexed
    let out_samples: Vec<u8> = out_samples_iter
        .map(|&idx| {
            if idx == 255 {
                vec![0, 0, 0, 0]
            } else {
                let p = palette_vec[idx as usize];
                vec![p[0], p[1], p[2], 255]
            }
        })
        .flatten()
        .collect();

    let out_img =
        RgbaImage::from_raw(width as u32, height as u32, out_samples).ok_or_else(|| {
            crate::error::PixelSnapperError::ProcessingError(
                "Failed to create output image".to_string(),
            )
        })?;

    Ok(QuantizedImage {
        img: out_img,
        palette: palette_vec,
        indexed,
    })
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
        assert_eq!(result.img.dimensions(), (10, 10));
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
        let p = result.img.get_pixel(0, 0);
        assert!(p[0] > 250);
        assert_eq!(result.palette.len(), 1);
    }
}
