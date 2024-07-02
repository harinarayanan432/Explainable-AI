#GPU needed
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(input_images)

activations = model.predict(dataset)

# Flatten the activations
flat_activations = activations.reshape((-1, activations.shape[-1]))

# Apply k-means clustering
kmeans = KMeans(n_clusters=9)
kmeans.fit(flat_activations)


for i in range(10):
    # Get the activations for this cluster
    cluster_activations = flat_activations[kmeans.labels_ == i]
    cluster_images = []
    for j in range(num_images):
        # Access the single value from cluster_activations
        cluster_activation_value = cluster_activations[:1000]  # Adjust this based on your data structure

        # Generate an image that maximally activates this neuron
        img = tf.random.normal((224, 224, 3))

        for _ in range(100):
            # Assuming the layer you are interested in is Dense layer at index 1
            layer_output = model.layers[1](tf.expand_dims(img, axis=0))  # Adjust layer indexing based on your model

            # Use tf.GradientTape to compute the gradient
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = -cluster_activation_value * tf.reduce_sum(layer_output)

            # Compute gradient with respect to the input image
            grad = tape.gradient(loss, img)

            if grad is not None:
                # Normalize the gradient using TensorFlow operations
                grad = grad / (tf.norm(grad) + 1e-5)

                # Clip the gradient
                grad = tf.clip_by_value(grad, -1, 1)

                # Update the input image using TensorFlow operations
                img = img + grad * 0.1

        img = tf.clip_by_value(img, 0, 1)
        cluster_images.append(img.numpy())

    # Display the cluster images
    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(np.hstack(cluster_images))

plt.show()
""
