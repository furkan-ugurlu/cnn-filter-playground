import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

# Set Matplotlib defaults for better visualization
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # Suppress warnings for cleaner output

# Read the image from the specified path
image_path = 'your_image_path.jpg'  # Specify your image path here
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image)

# Define a convolution kernel (sharpening filter)
# This kernel is designed to enhance edges in the image you can change it to look how it works
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  12, -1],
    [-1, -1, -1],
], dtype=tf.float32)

# Expand the kernel to match RGB channels (3 input channels)
kernel = tf.reshape(kernel, [3, 3, 1, 1])  # Shape: 3x3, 1 input channel, 1 output channel
kernel = tf.tile(kernel, [1, 1, 3, 1])     # Tile for 3 input channels (RGB)

# Prepare the image for convolution (normalize and add batch dimension)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)  # Add batch dimension

# Apply the convolution filter to the image
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME'
)

# Apply ReLU activation to highlight detected features
image_detect = tf.nn.relu(image_filter)

# Visualize the original image, filtered image, and detected features
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title('Input')
plt.subplot(132)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Filter')
plt.subplot(133)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Detect')
plt.show()