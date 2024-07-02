import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess an example image
img_path = 'dog.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Define Grad-CAM function
def grad_cam(model, x, layer_name):
    input_model = model.input
    output_model = model.output
    last_conv_layer = model.get_layer(layer_name)
    grads = tf.gradients(output_model[:, np.argmax(output_model)], last_conv_layer.output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    iterate = tf.keras.backend.function([input_model], [last_conv_layer.output[0], pooled_grads, output_model])
    conv_output, pooled_grads_value, predictions = iterate([x])
    for i in range(pooled_grads_value.shape[0]):
        conv_output[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Specify the layer name for Grad-CAM calculation
layer_name = 'block5_conv3'

# Generate class activation heatmap
heatmap = grad_cam(model, x, layer_name)

# Resize heatmap to match the original image size
heatmap_resized = np.uint8(255 * heatmap)
heatmap_resized = tf.image.resize(heatmap_resized, (img.shape[1], img.shape[0])).numpy()

# Apply heatmap to original image
heatmap_resized = heatmap_resized / 255.0
superimposed_img = heatmap_resized[..., np.newaxis] * img

# Plot original image and overlayed heatmap
plt.imshow(superimposed_img.astype(np.uint8))
plt.axis('off')
plt.show()
