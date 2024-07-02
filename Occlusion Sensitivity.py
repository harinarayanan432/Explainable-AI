#occlusion sensitivity for single image
#pip install tf-explain

import tensorflow as tf
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
if __name__ == "__main__":
     model = tf.keras.applications.resnet50.ResNet50(weights="imagenet", include_top=True)
     
IMAGE_PATH = "C:\\Users\\ADMIN\\OneDrive\\Pictures\\Screenshots\\dog.png"

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
model.summary()

data = ([img], None)
tabby_cat_class_index = 281
dog = 189
explainer = OcclusionSensitivity()
grid = explainer.explain(data, model, tabby_cat_class_index, 10)
explainer.save(grid, ".", "D:\\occlusion_sensitivity_10_cat.png")
grid = explainer.explain(data, model, dog, 10)
explainer.save(grid, ".", "D:\\occlusion_sensitivity_10_dog.png")
