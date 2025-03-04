import os
import kagglehub
import pickle
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

# Load training data
def train_df(tr_path):
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                 for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                 for image in os.listdir(os.path.join(tr_path, label))])
    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df

# Load testing data
def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])
    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

# preprocessing data
def get_preprocessed_data(tr_df, valid_df, ts_df, batch_size=32, img_size=(299, 299)):
    """
    Creates and returns ImageDataGenerators for training, validation, and testing.

    Parameters:
        tr_df (DataFrame): Training data DataFrame.
        valid_df (DataFrame): Validation data DataFrame.
        ts_df (DataFrame): Testing data DataFrame.
        batch_size (int): Batch size for training and validation generators. Default is 32.
        img_size (tuple): Target image size. Default is (299, 299).

    Returns:
        tr_gen: Training data generator with augmentation.
        valid_gen: Validation data generator.
        ts_gen: Test data generator.
    """
    
    # Train data augmentation
    train_gen = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))
    test_gen = ImageDataGenerator(rescale=1/255)

    # Load images from DataFrame
    tr_gen = train_gen.flow_from_dataframe(tr_df, x_col='Class Path', y_col='Class', 
                                           batch_size=batch_size, target_size=img_size)
    
    valid_gen = train_gen.flow_from_dataframe(valid_df, x_col='Class Path', y_col='Class', 
                                              batch_size=batch_size, target_size=img_size)
    
    ts_gen = test_gen.flow_from_dataframe(ts_df, x_col='Class Path', y_col='Class', 
                                          batch_size=16, target_size=img_size, shuffle=False)

    return tr_gen, valid_gen, ts_gen

# Plot image from dataset
def plot_one_image_per_class(tr_gen):
    """
    Displays one image per class from the test generator (tr_gen), ensuring 'notumor' appears first.
    
    Parameters:
        tr_gen: ImageDataGenerator instance for the train dataset.
    """
    # Get class indices and names
    class_dict = tr_gen.class_indices
    classes = list(class_dict.keys())

    # Fetch a batch of images and labels
    images, labels = next(tr_gen)

    # Convert one-hot labels back to class names
    label_indices = np.argmax(labels, axis=1)  # Get index of the highest probability class
    label_names = [classes[idx] for idx in label_indices]  # Convert indices to class names

    # Dictionary to store first occurrence of each class
    class_images = {}

    # Ensure "notumor" appears first
    if "notumor" in classes:
        classes.remove("notumor")
        classes.insert(0, "notumor")

    # Find the first image for each class
    for img, label_name in zip(images, label_names):
        if label_name not in class_images:  # Store only the first occurrence
            class_images[label_name] = img
        if len(class_images) == len(classes):  # Stop if we have all classes
            break

    # Plot one image per class
    plt.figure(figsize=(15, 10))
    for i, (class_name, img) in enumerate(class_images.items()):
        plt.subplot(1, len(class_images), i + 1)
        plt.imshow(img)
        plt.title(class_name, color='k', fontsize=15)
        plt.axis("off")

    plt.show()


# Build a model
def build_model(img_shape=(299, 299, 3), learning_rate=0.001):
    """
    Builds and returns a ResNet50-based model for brain tumor classification.

    Parameters:
        img_shape (tuple): Input image shape, default (299, 299, 3).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (tf.keras.Model): Compiled ResNet50-based model.
    """
    
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation='relu'),
        Dropout(rate=0.25),
        Dense(4, activation='softmax')  # 4 output classes
    ])

    # Compile the model
    model.compile(
        Adamax(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )

    return model

# Predict the tumor 
def predict(img_path, class_dict, model):

    label = list(class_dict.keys())
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    predictions = model.predict(img)
    probs = list(predictions[0])

    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.subplot(2, 1, 2)
    bars = plt.barh(label, probs)
    plt.xlabel('Probability', fontsize=15)
    plt.show()
