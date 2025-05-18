# CNN Filter Playground

This project demonstrates how convolutional filters work in image processing using TensorFlow and Matplotlib. It allows you to experiment with custom convolution kernels (filters) and visualize their effects on any input image.

## Features

- **Custom Convolution Kernel:** Easily modify the kernel to see how different filters (e.g., sharpening, edge detection, blurring) affect your image.
- **Visualization:** The script displays the original image, the filtered image, and the activated (ReLU) output side by side for easy comparison.
- **Educational Purpose:** Great for learning and teaching how convolutional layers in CNNs process images.

## How It Works

1. **Image Loading:** Reads an image from a specified path.
2. **Kernel Definition:** Applies a 3x3 sharpening filter by default, but you can change the kernel values to try other effects.
3. **Convolution:** The kernel is applied to the image using TensorFlow's `conv2d` operation.
4. **Activation:** A ReLU activation highlights the detected features.
5. **Visualization:** The results are plotted for visual inspection.

## Usage

1. Replace `'your_image_path.jpg'` in the script with the path to your own image.
2. (Optional) Modify the `kernel` variable to try different filters.
3. Run the script:
    ```bash
    python CNN_Filter_Playground.py
    ```

## Example

The default kernel is a sharpening filter:
```python
kernel = tf.constant([
    [-1, -1, -1],
    [-1, 12, -1],
    [-1, -1, -1],
], dtype=tf.float32)
```
You can replace it with other kernels, such as edge detection or blur, to see different effects.

## Requirements

- Python 3.x
- TensorFlow
- Matplotlib

Install dependencies with:
```bash
pip install tensorflow matplotlib
```

## Notes

- The script works with RGB images.
- For best results, use clear and high-resolution images.

*This project is intended for educational and experimental purposes.*
