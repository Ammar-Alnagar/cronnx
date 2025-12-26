use crate::error::InferenceError;
use image::imageops::FilterType;
use ndarray::{Array, Array4, Axis};

// ImageNet Standards
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocesses an image file into an ONNX-compatible tensor.
/// Returns a tensor of shape [1, 3, 224, 224].
pub fn load_and_preprocess(path: &str) -> Result<Array4<f32>, InferenceError> {
    // 1. Load image
    let img = image::open(path)?;

    // 2. Resize to 224x224
    let resized = img.resize_exact(224, 224, FilterType::Triangle);

    // 3. Convert to ndarray and normalize
    let mut normalized_data = Vec::with_capacity(3 * 224 * 224);

    // Process pixels in channel-first order (RGB)
    for pixel in resized.to_rgb8().pixels() {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);

        // Normalize R
        normalized_data.push(((r as f32 / 255.0) - MEAN[0]) / STD[0]);
        // Normalize G
        normalized_data.push(((g as f32 / 255.0) - MEAN[1]) / STD[1]);
        // Normalize B
        normalized_data.push(((b as f32 / 255.0) - MEAN[2]) / STD[2]);
    }

    // Currently data is [R, G, B, R, G, B...] which is [Height, Width, Channels] format
    let array = Array::from_shape_vec((224, 224, 3), normalized_data)
        .map_err(|e| InferenceError::PreprocessingError(e.to_string()))?;

    // Permute to [Channels, Height, Width] -> (2, 0, 1)
    let array = array.permuted_axes([2, 0, 1]);

    // Add batch dimension [1, 3, 224, 224]
    let array = array.insert_axis(Axis(0));

    // Ensure standard layout (contiguous)
    let array = array.as_standard_layout().to_owned();

    Ok(array)
}

/// Preprocesses image from raw bytes into an ONNX-compatible tensor.
/// Returns a tensor of shape [1, 3, 224, 224].
pub fn process_bytes(buffer: &[u8]) -> Result<Array4<f32>, InferenceError> {
    // 1. Load image from bytes (guess format)
    let img = image::load_from_memory(buffer).map_err(InferenceError::ImageError)?;

    // 2. Resize
    let resized = img.resize_exact(224, 224, FilterType::Triangle);

    // 3. Normalize & Create Tensor (Same logic as file-based approach)
    let mut normalized_data = Vec::with_capacity(3 * 224 * 224);

    for pixel in resized.to_rgb8().pixels() {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        // Normalize (ImageNet stats)
        normalized_data.push(((r as f32 / 255.0) - MEAN[0]) / STD[0]);
        normalized_data.push(((g as f32 / 255.0) - MEAN[1]) / STD[1]);
        normalized_data.push(((b as f32 / 255.0) - MEAN[2]) / STD[2]);
    }

    // Shape: [H, W, C] -> Permute to [C, H, W] -> Add Batch [1, C, H, W]
    let array = Array::from_shape_vec((224, 224, 3), normalized_data)
        .map_err(|e| InferenceError::PreprocessingError(e.to_string()))?;

    let array = array.permuted_axes([2, 0, 1]);
    let array = array.insert_axis(Axis(0));

    Ok(array.as_standard_layout().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use std::io::Cursor;

    #[test]
    fn test_load_and_preprocess_shape() {
        // Test that the function is defined but we can't test with real files in unit tests
        // This test will fail since we don't have a real file, but it tests the error path
        let result = load_and_preprocess("nonexistent.jpg");
        assert!(result.is_err());

        match result.unwrap_err() {
            InferenceError::ImageError(_) => {} // Expected for nonexistent file
            _ => panic!("Expected ImageError for nonexistent file"),
        }
    }

    #[test]
    fn test_process_bytes_shape() {
        // Create a simple test image (10x10 RGB)
        let img = RgbImage::new(10, 10);
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

        let result = process_bytes(&buffer);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        // Should be [1, 3, 224, 224] after preprocessing
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_process_bytes_normalization() {
        // Test that normalization produces values in expected range
        let img = RgbImage::from_pixel(10, 10, image::Rgb([255, 255, 255])); // White image
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

        let tensor = process_bytes(&buffer).unwrap();
        let view = tensor.view();

        // Check that values are normalized (should be around 1-2 for white pixels with ImageNet stats)
        let first_pixel = view[[0, 0, 0, 0]]; // [batch, channel, height, width]
        assert!(first_pixel > 1.0 && first_pixel < 3.0); // Approximate range after normalization
    }

    #[test]
    fn test_process_bytes_different_sizes() {
        // Test with different input sizes - should all resize to 224x224
        let small_img = RgbImage::new(32, 32);
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        small_img
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();

        let tensor = process_bytes(&buffer).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_process_bytes_error_handling() {
        // Test with invalid image data
        let invalid_data = b"invalid image data";
        let result = process_bytes(invalid_data);
        assert!(result.is_err());

        match result.unwrap_err() {
            InferenceError::ImageError(_) => {} // Expected
            _ => panic!("Expected ImageError"),
        }
    }

    #[test]
    fn test_mean_std_constants() {
        // Verify ImageNet normalization constants
        assert_eq!(MEAN, [0.485, 0.456, 0.406]);
        assert_eq!(STD, [0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_tensor_dimensions() {
        // Create a test image and verify tensor dimensions
        let img = RgbImage::new(50, 50);
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

        let tensor = process_bytes(&buffer).unwrap();

        // Verify dimensions: [batch, channels, height, width]
        assert_eq!(tensor.shape()[0], 1); // batch size
        assert_eq!(tensor.shape()[1], 3); // channels (RGB)
        assert_eq!(tensor.shape()[2], 224); // height
        assert_eq!(tensor.shape()[3], 224); // width
    }

    #[test]
    fn test_preprocessing_consistency() {
        // Create a test image with known values
        let img = RgbImage::from_pixel(10, 10, image::Rgb([128, 128, 128])); // Gray image
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

        let tensor = process_bytes(&buffer).unwrap();
        let view = tensor.view();

        // For a gray image with all pixels [128, 128, 128], after normalization
        // the values should be approximately (128/255 - mean) / std for each channel
        let expected_channel_0 = ((128.0 / 255.0) - MEAN[0]) / STD[0];
        let actual_channel_0 = view[[0, 0, 0, 0]];

        // Allow small tolerance for floating point differences
        assert!((actual_channel_0 - expected_channel_0).abs() < 0.001);
    }
}
