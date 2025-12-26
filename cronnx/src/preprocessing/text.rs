use crate::error::InferenceError;
use ndarray::Array4;

/// Simple text preprocessing: convert text to a 3x224x224 image-like tensor
/// by mapping characters to RGB values
pub fn preprocess_text(text: &str) -> Result<Array4<f32>, InferenceError> {
    let mut data = vec![0.0f32; 3 * 224 * 224];

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len().min(224 * 224);

    for i in 0..len {
        let c = chars[i];
        let r = (c as u32 % 256) as f32 / 255.0;
        let g = ((c as u32 / 256) % 256) as f32 / 255.0;
        let b = ((c as u32 / 65536) % 256) as f32 / 255.0;

        // Normalize to ImageNet mean/std
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        data[i * 3] = (r - mean[0]) / std[0];
        data[i * 3 + 1] = (g - mean[1]) / std[1];
        data[i * 3 + 2] = (b - mean[2]) / std[2];
    }

    // Fill the rest with mean
    for i in len..(224 * 224) {
        data[i * 3] = -0.485 / 0.229;
        data[i * 3 + 1] = -0.456 / 0.224;
        data[i * 3 + 2] = -0.406 / 0.225;
    }

    let array = Array4::from_shape_vec((1, 3, 224, 224), data)
        .map_err(|e| InferenceError::PreprocessingError(format!("Shape error: {}", e)))?;

    Ok(array)
}
