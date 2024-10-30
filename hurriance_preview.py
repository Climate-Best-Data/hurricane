import matplotlib.pyplot as plt
import seaborn as sns
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
ssl._create_default_https_context = ssl._create_unverified_context

plt.rcParams.update({"figure.figsize": (7, 7), "figure.dpi": 180, "font.size": 13, 'font.family': 'serif'})
sns.set_theme(style="darkgrid", rc={'axes.grid': False})
tqdm.pandas()

def create_montage_rgb(images):
    return np.stack([montage(images[:, :, :, i]) for i in range(images.shape[3])], -1)

src_dir = Path('./lib')
img_data = pd.DataFrame({'location': list(src_dir.glob('**/*.jp*g'))})
img_data['label'] = img_data['location'].apply(lambda x: x.parent.stem)
img_data['split'] = img_data['location'].apply(lambda x: x.parent.parent.stem)
img_data['coords'] = img_data['location'].apply(lambda x: x.stem)
img_data['latitude'] = img_data['coords'].apply(lambda x: float(x.split('_')[0]))
img_data['longitude'] = img_data['coords'].apply(lambda x: float(x.split('_')[-1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for label, group in img_data.groupby('label'):
    ax1.scatter(group['latitude'], group['longitude'], label=label, alpha=0.6)
ax1.set_title('Data Split by Label')
ax1.legend()

for group, subset in img_data.groupby('split'):
    ax2.scatter(subset['latitude'], subset['longitude'], label=group, alpha=0.6)
ax2.set_title('Data Grouped by Split')
ax2.legend()

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
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

last_img = Image.open(img_data['location'].iloc[-1])
web_palette_img = last_img.convert('P', palette='WEB', dither=None).convert('RGB')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(last_img)
ax2.imshow(web_palette_img)
ax1.set_title("Original Image")
ax2.set_title("Reduced Colors")

def normalize_color_histogram(img_path):
    img_raw = Image.open(img_path).convert('P', palette='WEB', dither=None)
    counts, _ = np.histogram(np.array(img_raw).ravel(), bins=np.arange(256))
    return counts.astype(float) / np.prod(img_raw.size)

img_data['color_data'] = img_data['location'].progress_apply(normalize_color_histogram)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
color_data = np.stack(img_data['color_data'])
ax1.imshow(color_data)
ax1.set_title('Raw Color Count per Image')

norm_avg = np.tile(color_data.mean(axis=0), (color_data.shape[0], 1))
ax2.imshow(color_data / norm_avg.clip(1e-4), vmin=0.1, vmax=10, cmap='plasma')
ax2.set_title('Normalized Color Counts')

pca_model = PCA(n_components=2)
pca_coords = pca_model.fit_transform(color_data)
img_data['x'] = pca_coords[:, 0]
img_data['y'] = pca_coords[:, 1]

fig, ax = plt.subplots(figsize=(15, 15))
for label, group in img_data.groupby('label'):
    ax.scatter(group['x'], group['y'], label=label)
ax.legend()
ax.set_title("PCA Projection by Damage Type")

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

display_pca_images(img_data.sample(200))

img_data.to_json('processed_data.json')

resnet_model = resnet50.ResNet50(include_top=False, weights='imagenet')
feature_extractor = models.Sequential([layers.Lambda(lambda x: x - tf.constant([103.9, 116.78, 123.68])), resnet_model, layers.GlobalAveragePooling2D()])
feature_extractor.save('saved_feature_model.h5')

img_data['extracted_features'] = img_data['location'].progress_apply(lambda x: feature_extractor.predict(np.expand_dims(imread(x), 0))[0])

img_data.to_json('extracted_features.json')