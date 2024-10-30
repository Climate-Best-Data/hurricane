# Import necessary libraries for data manipulation, image processing, machine learning, and visualization
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.io import imread
from skimage.util import montage
from tqdm import tqdm
from sklearn.decomposition import PCA
from keras.applications import resnet50
from keras import models, layers
import ssl

# Configure SSL to avoid issues with secure connections
ssl._create_default_https_context = ssl._create_unverified_context

# Set global parameters for visualization and progress bars
plt.rcParams.update({"figure.figsize": (7, 7), "figure.dpi": 200, "font.size": 13, 'font.family': 'Calibri'})

# Enable progress bars for pandas methods
tqdm.pandas()


# Function to create a montage of RGB images for easier visualization
def create_montage_rgb(images):
    return np.stack([montage(images[:, :, :, i]) for i in range(images.shape[3])], -1)


# Define source directory and load image paths into a DataFrame
src_dir = Path('./lib')
img_data = pd.DataFrame({'location': list(src_dir.glob('**/*.jp*g'))})

# Extract metadata from image paths: label (damage type), split (data partition), and coordinates (lat/lon)
img_data['label'] = img_data['location'].apply(lambda x: x.parent.stem)
img_data['split'] = img_data['location'].apply(lambda x: x.parent.parent.stem)
img_data['coords'] = img_data['location'].apply(lambda x: x.stem)
img_data['latitude'] = img_data['coords'].apply(lambda x: float(x.split('_')[0]))
img_data['longitude'] = img_data['coords'].apply(lambda x: float(x.split('_')[-1]))

# Visualize data distribution by damage label and split category
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for label, group in img_data.groupby('label'):
    ax1.scatter(group['latitude'], group['longitude'], label=label, alpha=0.6)
ax1.set_title('Data Split by Label')
ax1.legend()

for group, subset in img_data.groupby('split'):
    ax2.scatter(subset['latitude'], subset['longitude'], label=group, alpha=0.6)
ax2.set_title('Data Grouped by Split')
ax2.legend()

# Create montage plots by label and split to visually inspect the images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, (label, group) in zip(axes, img_data.groupby('label')):
    image_stack = np.stack(group.sample(100)['location'].apply(imread), 0)
    ax.imshow(create_montage_rgb(image_stack))
    ax.axis('off')
    ax.set_title(label)

fig, axes = plt.subplots(2, 2, figsize=(18, 18))
for ax, (split, subset) in zip(axes.flatten(), img_data.groupby('split')):
    img_stack = np.stack(subset.sample(100)['location'].apply(imread), 0)
    ax.imshow(create_montage_rgb(img_stack))
    ax.set_title(split)
    ax.axis('off')

# Load a sample image, reduce its color palette, and compare original vs. reduced color versions
last_img = Image.open(img_data['location'].iloc[-1])
web_palette_img = last_img.convert('P', palette='WEB', dither=None).convert('RGB')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(last_img)
ax2.imshow(web_palette_img)
ax1.set_title("Original Image")
ax2.set_title("Reduced Colors")


# Function to normalize color histograms for each image, returning the color count as features
def normalize_color_histogram(img_path):
    img_raw = Image.open(img_path).convert('P', palette='WEB', dither=None)
    counts, _ = np.histogram(np.array(img_raw).ravel(), bins=np.arange(256))
    return counts.astype(float) / np.prod(img_raw.size)


# Apply color normalization function to all images
img_data['color_data'] = img_data['location'].progress_apply(normalize_color_histogram)

# Stack color features for dimensionality reduction using PCA
color_data = np.stack(img_data['color_data'])
pca_model = PCA(n_components=2)
pca_coords = pca_model.fit_transform(color_data)
img_data['x'] = pca_coords[:, 0]
img_data['y'] = pca_coords[:, 1]

# Visualize PCA projection by damage type
fig, ax = plt.subplots(figsize=(10, 10))
for label, group in img_data.groupby('label'):
    ax.scatter(group['x'], group['y'], label=label)
ax.legend()
ax.set_title("PCA Projection by Damage Type")


# Function to overlay images on the PCA plot, enhancing visual analysis
def display_pca_images(df, img_zoom=1.2):
    fig, ax = plt.subplots(figsize=(10, 10))
    for _, row in df.iterrows():
        img = Image.open(row['location']).resize((60, 60))
        box_img = OffsetImage(img, zoom=img_zoom)
        ab = AnnotationBbox(box_img, (row['x'], row['y']), frameon=False)
        ax.add_artist(ab)
    ax.autoscale()
    ax.axis('off')
    plt.show()


# Display a subset of images on the PCA plot for inspection
display_pca_images(img_data.sample(200))

# Save DataFrame with processed information to JSON
img_data.to_json('processed_data.json')

# Initialize ResNet50 model for feature extraction, removing the final layers and adding global pooling
resnet_model = resnet50.ResNet50(include_top=False, weights='imagenet')
feature_extractor = models.Sequential([
    layers.Lambda(lambda x: x - tf.constant([103.9, 116.78, 123.68])), 
    resnet_model, 
    layers.GlobalAveragePooling2D()
])

# Save the feature extraction model
feature_extractor.save('saved_feature_model.h5')

# Extract features from each image using the pretrained model and save them to the DataFrame
img_data['extracted_features'] = img_data['location'].progress_apply(lambda x: feature_extractor.predict(np.expand_dims(imread(x), 0))[0])

# Save DataFrame with extracted features to JSON
img_data.to_json('extracted_features.json')