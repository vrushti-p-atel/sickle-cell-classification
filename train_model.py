import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
import os
import pickle

# Classes
classes = ["circular", "falciforme", "outras"]

# Dataset paths (assuming cloned)
train_dir = './train-test-erythrocytes/dataset/5-fold/round_1/train'
test_dir = './train-test-erythrocytes/dataset/5-fold/round_1/test'

def createModel():
    base_model = applications.ResNet50(weights='imagenet', include_top=True)
    vector = base_model.get_layer("avg_pool").output
    model = tf.keras.Model(base_model.input, vector)
    return model

def extract_features(path, model):
    x_list = []
    y_list = []
    for label in range(3):
        folder_path = os.path.join(path, classes[label])
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if not file.endswith(('.jpg', '.png', '.jpeg')):
                continue
            img = image.load_img(file_path, target_size=(224,224))
            img_arr = image.img_to_array(img)
            img_arr_b = np.expand_dims(img_arr, axis=0)
            input_img = preprocess_input(img_arr_b)
            features = model.predict(input_img)
            x_list.append(features.ravel())
            y_list.append(label)
    return np.array(x_list), np.array(y_list)

# Load model
model = createModel()

# Extract features
print("Extracting train features...")
X_train, y_train = extract_features(train_dir, model)
print("Extracting test features...")
X_test, y_test = extract_features(test_dir, model)

# Train SVM
print("Training SVM...")
clf = SVC(C=2.9, kernel='linear')
clf.fit(X_train, y_train)

# Save model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model saved as svm_model.pkl")