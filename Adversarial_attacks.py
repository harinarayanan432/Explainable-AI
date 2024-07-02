import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained model (for example, the MobileNetV2 model trained on ImageNet)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Function to preprocess an image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to deprocess an image (for visualization)
def deprocess_image(img):
    img = img.reshape((224, 224, 3))
    img /= 2.0
    img += 0.5
    img *= 255.0
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Function to generate adversarial examples using Fast Gradient Sign Method (FGSM)
def generate_adversarial_example(model, image, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        target_class = tf.argmax(prediction[0])
        loss = tf.keras.losses.sparse_categorical_crossentropy([target_class], prediction)
    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    adversarial_image = image + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, -1, 1)
    return adversarial_image.numpy()

# Load and preprocess the image
image_path = 'H:\\dog1.jpg' # Provide the path to your image
image = preprocess_image(image_path)

# Generate adversarial example
epsilon = 0.1  # Adjust epsilon as needed
adversarial_image = generate_adversarial_example(model, image, epsilon)

# Make predictions on original and adversarial images
original_prediction = model.predict(image)
adversarial_prediction = model.predict(adversarial_image)

# Extract confidence rates from predictions
original_confidence = original_prediction.max()
adversarial_confidence = adversarial_prediction.max()

# Display the original and adversarial images along with their predictions and confidence rates
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(deprocess_image(image))
plt.title('Original Image\nPrediction: {} (Confidence: {:.2%})'.format(tf.keras.applications.mobilenet_v2.decode_predictions(original_prediction)[0][0][1], original_confidence))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(deprocess_image(adversarial_image))
plt.title('Adversarial Image\nPrediction: {} (Confidence: {:.2%})'.format(tf.keras.applications.mobilenet_v2.decode_predictions(adversarial_prediction)[0][0][1], adversarial_confidence))
plt.axis('off')

plt.show()
