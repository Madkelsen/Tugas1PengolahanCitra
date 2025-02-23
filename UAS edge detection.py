# Nama : Armada Satya Permana
# NIM : 220401010131
# Kelas : IFD51
# Mata Kuliah : Pengolahan Citra

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

image_path = "image.png" 
image = iio.imread(image_path, pilmode="L") 

roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

edge_roberts_x = convolve(image, roberts_x)
edge_roberts_y = convolve(image, roberts_y)
edge_roberts = np.sqrt(edge_roberts_x**2 + edge_roberts_y**2)

edge_sobel_x = convolve(image, sobel_x)
edge_sobel_y = convolve(image, sobel_y)
edge_sobel = np.sqrt(edge_sobel_x**2 + edge_sobel_y**2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Gambar Asli")
axes[0].axis("off")

axes[1].imshow(edge_roberts, cmap="gray")
axes[1].set_title("Deteksi Tepi - Roberts")
axes[1].axis("off")

axes[2].imshow(edge_sobel, cmap="gray")
axes[2].set_title("Deteksi Tepi - Sobel")
axes[2].axis("off")

plt.show()
