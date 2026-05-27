import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from openslide import OpenSlide
from helper_scripts.compressed_sampler import SampleManager
from tqdm import tqdm


def create_filelist(directory, fileextension: str = ""):
    filelist = list()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(fileextension.lower()):
                filelist.append(Path(root) / file)
    filelist.sort()
    return filelist


def get_directory_size_gb(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size / (1024**3)  # Convert bytes to gigabytes


def reconstruct_image_from_bbox(nuclei_list, display=True, dict_key="bbox"):
    max_y = max(d[dict_key][1][0] for d in nuclei_list)  # y2
    max_x = max(d[dict_key][1][1] for d in nuclei_list)  # x2
    image_shape = (int(max_y), int(max_x))

    # Create blank canvas
    image = np.zeros(image_shape, dtype=np.uint8)

    # Place each nucleus bitmap into the canvas
    for d in nuclei_list:
        (y1, x1), (y2, x2) = d[dict_key]
        y1, x1, y2, x2 = map(int, [y1, x1, y2, x2])
        bitmap = np.array(d["bbox_bitmap"], dtype=np.uint8)
        # Ensure the bitmap size matches bbox
        bitmap = bitmap[: y2 - y1, : x2 - x1]
        h, w = bitmap.shape
        h_bbox, w_bbox = y2 - y1, x2 - x1

        # Pad if needed --> bcs of rotation etc, the bbox coords might give a bigger box than bitmap actually is.
        if h < h_bbox or w < w_bbox:
            padded = np.zeros((h_bbox, w_bbox), dtype=bitmap.dtype)
            padded[:h, :w] = bitmap
            bitmap = padded

        try:
            image[y1:y2, x1:x2] = np.maximum(image[y1:y2, x1:x2], bitmap)
        except Exception as e:
            print(e)
            pass

    if display:
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()

    return image


def check_coords_key_centr_bbox(sampler: SampleManager, margin=5):
    outside_margin = list()
    # Initialize min/max for centroids
    minx_centr = np.inf
    miny_centr = np.inf
    maxx_centr = -np.inf
    maxy_centr = -np.inf

    # Initialize min/max for bbox
    minx_bbox = np.inf
    miny_bbox = np.inf
    maxx_bbox = -np.inf
    maxy_bbox = -np.inf

    for i, key in enumerate(sampler.sample_keys):
        centroids = sampler.sample_xs[i]["centroid"]
        centroids = (int(centroids[0]), int(centroids[1]))
        bbox = sampler.sample_xs[i]["bbox"]
        key_centroids = key.split("_")[-2:]

        (y0, x0), (y1, x1) = bbox
        bbox_cy = (y0 + y1) / 2
        bbox_cx = (x0 + x1) / 2
        bbox_centroids = (int(bbox_cx), int(bbox_cy))

        parts = key.split("_")
        key_centroids = (int(parts[-2]), int(parts[-1]))

        # Convert everything to NumPy arrays
        centroids_arr = np.array(centroids)
        bbox_arr = np.array(bbox_centroids)
        key_arr = np.array(key_centroids)

        # Check if differences are within the margin
        all_same = np.all(np.abs(centroids_arr - bbox_arr) <= margin) and np.all(
            np.abs(centroids_arr - key_arr) <= margin
        )

        if not all_same:
            outside_margin.append([bbox_centroids, centroids, key_centroids, key, i])

        # Update centroid min/max
        cx, cy = centroids
        minx_centr = min(minx_centr, cx)
        maxx_centr = max(maxx_centr, cx)
        miny_centr = min(miny_centr, cy)
        maxy_centr = max(maxy_centr, cy)

        # Update bbox min/max
        minx_bbox = min(minx_bbox, bbox_cx)
        maxx_bbox = max(maxx_bbox, bbox_cx)
        miny_bbox = min(miny_bbox, bbox_cy)
        maxy_bbox = max(maxy_bbox, bbox_cy)
    print("Centroid min max:", minx_centr, miny_centr, maxx_centr, maxy_centr)
    print("BBox min max:", minx_bbox, miny_bbox, maxx_bbox, maxy_bbox)
    return outside_margin
