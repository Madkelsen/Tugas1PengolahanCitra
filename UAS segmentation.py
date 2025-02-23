# Nama : Armada Satya Permana
# NIM : 220401010131
# Kelas : IFD51
# Mata Kuliah : Pengolahan Citra

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_path = "image.png"
image = iio.imread(image_path, pilmode="L") 

pixels = image.reshape(-1, 1)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(pixels)

segmented_pixels = kmeans.labels_

segmented_image = segmented_pixels.reshape(image.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Gambar Asli")
axes[0].axis("off")

axes[1].imshow(segmented_image, cmap="viridis") 
axes[1].set_title(f"Segmentasi K-Means (k={k})")
axes[1].axis("off")

plt.show()

