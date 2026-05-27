import torch
from torch import nn
from torchvision import transforms
import imageio
from skimage.transform import resize

import pandas as pd
import csv
import numpy as np
import os
from pathlib import Path

import utils

from enum import Enum

import sys
import matplotlib.pyplot as plt

from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import SubsetRandomSampler

from PIL import Image
from sampler import SampleManager

from train import DATASET


class StainAugment:
    """Randomly perturb H&E stain channels."""

    def __init__(self, sigma1=0.2, sigma2=0.2):
        self.sigma1 = sigma1  # perturbation for channel intensities
        self.sigma2 = sigma2  # perturbation for channel colors

    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)
        # Random color perturbation per channel
        for c in range(3):
            scale = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            shift = np.random.uniform(-self.sigma2, self.sigma2) * 255
            img_array[:, :, c] = np.clip(img_array[:, :, c] * scale + shift, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, keys, images, labels, transform=None):
        self.images = images

        if labels is None:
            labels = [0] * len(images)
        self.labels = torch.tensor(labels, dtype=torch.long)  # crossentropy

        self.keys = keys
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # make (80,80) to (80,80,1)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Convert the image to 3 channels by repeating the single channel 3 times
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # Convert the image (numpy array) to PIL image for transformations
        # image = Image.fromarray((image * 255).astype(np.uint8))  # Assuming image is in range [0, 1]
        image = Image.fromarray(image)  # Assuming image is 8b
        if self.transform:
            image = self.transform(image)  # Apply the transform (augmentations)

        return {"image": image, "key": self.keys[idx], "label": self.labels[idx]}

    def get_labels(self):
        return self.labels


def get_training_sampler(dataset, subset_size, balance_samples):
    if balance_samples:
        labs = [0 if v[0] == 1 else 1 for v in dataset.get_labels().numpy()]
        unique_values, counts = np.unique(labs, return_counts=True)
        clean_counts = {int(k): int(v) for k, v in zip(unique_values, counts)}
        print("Rebalancing label counts:", clean_counts)

        return ImbalancedDatasetSampler(dataset, labels=labs, num_samples=subset_size)

    else:
        indices = np.random.choice(len(dataset), size=subset_size, replace=False)
        return SubsetRandomSampler(indices)


def prep_dataset(sampler, transformer, dev_mode, training=True, sen_model=None):
    xs, ys, keys = sampler.get()

    if training:
        print(f"TRAINING {sen_model}")
        labels = sampler.sample_ys

        # filter out samples with no label
        filtered_data = [
            (x, label, key)
            for x, label, key in zip(xs, labels, keys)
            if label is not None
        ]
        xs, labels, keys = zip(*filtered_data)

        # show distribution
        count_0 = sum(1 for lbl in labels if lbl == 0)
        count_1 = sum(1 for lbl in labels if lbl == 1)
        print(f"Number of samples: {count_0} vs {count_1}")

        print(labels[0:20])
    else:
        labels = None

    if dev_mode:
        print("DEV MODE")
        xs = xs[0:100]
        labels = labels[0:100]
        keys = keys[0:100]

    return CustomImageDataset(keys, xs, labels=labels, transform=transformer)


# size for centering.  in case of non-padded images
# disabling sizing or centering... diff methods lead to mistakes.  clearer to require pre-normalized input
import cv2
import numpy as np
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from skimage import morphology


def sharpen(img):
    thresh_value = threshold_otsu(img)
    smoothed_mask = (img > thresh_value).astype(int)
    return smoothed_mask


def smooth(img, radius=3):
    selem = morphology.disk(radius)
    if len(img.shape) == 3:
        img = img[:, :, 0]

    opened = morphology.opening(img, selem)
    closed = morphology.closing(opened, selem)
    smoothed = (closed * 255).astype(np.uint8)
    smoothed_3c = np.repeat(smoothed[..., np.newaxis], 3, axis=-1)
    return smoothed_3c


class FastGaussianBlur:
    def __init__(self, kernel_size=9, sigma=4.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        np_img = np.array(img)  # Convert PIL image to NumPy array
        blurred_img = cv2.GaussianBlur(
            np_img, (self.kernel_size, self.kernel_size), self.sigma
        )
        return Image.fromarray(blurred_img)


def val_transforms(cfg, size, data_mods=None):
    """
    Validation transforms:
    - Uses only the currently toggled train transforms (no stochastic flips/rotations)
    - Applies optional data_mods (blur, smooth, etc.)
    """
    t = []

    # Optional padding / resizing (if enabled in train)
    if cfg.get("resize_and_pad", False):
        t.append(ResizeAndPad((size)))

    # Optional Gaussian blur
    if cfg.get("gaussian_blur", False):
        kernel = cfg.get("gaussian_blur_kernel", 9)
        sigma = cfg.get("gaussian_blur_sigma", (0.1, 4.0))
        t.append(transforms.GaussianBlur(kernel_size=kernel, sigma=tuple(sigma)))

    # Optional custom FastGaussianBlur
    if cfg.get("fast_gaussian_blur", False):
        t.append(
            FastGaussianBlur(
                cfg["fast_gaussian_blur_kernel"], cfg["fast_gaussian_blur_sigma"]
            )
        )

    # Optional sharpening via lambda
    if cfg.get("sharpen", False):
        t.append(
            transforms.Lambda(
                lambda x: Image.fromarray((sharpen(np.array(x)) * 255).astype(np.uint8))
            )
        )

    # Optional smooth/blur mods from data_mods (custom preprocessing)
    if data_mods is not None:
        data_mods = sorted(data_mods, key=lambda x: 0 if x[0] == "blur" else 1)
        for data_mod in data_mods:
            key, val = data_mod
            if key == "blur":
                t.append(FastGaussianBlur(val, 4))
            elif key == "smooth":
                t.append(
                    transforms.Lambda(
                        lambda x: Image.fromarray(smooth(np.array(x), radius=val))
                    )
                )
            else:
                raise ValueError(f"Unknown data_mod key: {key}")

    # Mandatory tensor conversion
    t.append(transforms.ToTensor())

    # Optional normalization
    if cfg.get("normalize", False):
        t.append(
            transforms.Normalize(
                mean=cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
                std=cfg.get("normalize_std", [0.229, 0.224, 0.225]),
            )
        )

    return transforms.Compose(t)


def train_transforms(cfg, size):
    t = []

    # Resize options
    if cfg.get("resize", False):
        t.append(transforms.Resize(size))

    if cfg.get("center_crop", False):
        t.append(transforms.CenterCrop(size))

    if cfg.get("random_resized_crop", False):
        t.append(
            transforms.RandomResizedCrop(
                size, scale=cfg.get("random_resized_crop_scale", (0.8, 1.0))
            )
        )

    # custom transform - if images are not preprocessed - fptr
    if cfg.get("resize_and_pad", False):
        t.append(ResizeAndPad(size))

    # Geometric augmentations
    if cfg.get("random_rotation", False):
        t.append(transforms.RandomRotation(cfg["random_rotation"]))

    if cfg.get("horizontal_flip", False):
        t.append(transforms.RandomHorizontalFlip())

    if cfg.get("vertical_flip", False):
        t.append(transforms.RandomVerticalFlip())

    # Color / blur
    if cfg.get("color_jitter", False):
        t.append(transforms.ColorJitter(**cfg["color_jitter_params"]))

    if cfg.get("gaussian_blur", False):
        kernel = cfg.get("gaussian_blur_kernel", 9)
        sigma = cfg.get("gaussian_blur_sigma", (0.1, 4.0))
        t.append(transforms.GaussianBlur(kernel_size=kernel, sigma=tuple(sigma)))

    if cfg.get("fast_gaussian_blur", False):
        t.append(
            FastGaussianBlur(
                cfg["fast_gaussian_blur_kernel"], cfg["fast_gaussian_blur_sigma"]
            )
        )  # custom

    # Sharpening via lambda
    if cfg.get("sharpen", False):
        t.append(
            transforms.Lambda(
                lambda x: Image.fromarray((sharpen(np.array(x)) * 255).astype(np.uint8))
            )
        )

    # RandomApply / RandomChoice examples
    if cfg.get("random_apply_gaussian", False):
        t.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=cfg.get("random_apply_p", 0.5),
            )
        )

    if cfg.get("random_choice", False):
        t.append(
            transforms.RandomChoice(
                [
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                ]
            )
        )

    # Mandatory
    t.append(transforms.ToTensor())

    # Normalize
    if cfg.get("normalize", False):
        t.append(
            transforms.Normalize(
                mean=cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
                std=cfg.get("normalize_std", [0.229, 0.224, 0.225]),
            )
        )

    return transforms.Compose(t)


def _create_filelist(directory, fileextension=""):
    directory = Path(directory)
    pickle_filelist = list()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(fileextension):
                pickle_filelist.append(Path(root) / file)

    return pickle_filelist


def _load_all_samplers(
    input_dir,
    output_dir,
    train_val_ratio=0.8,
    fileextension="",
    celldict_key="bbox_pixels",
    label_key="sample_ys",
    use_subset=False,
    # Filtering parameters
    dab_key="dab_mean",
    positive_threshold=0.25,
    # Exclusion zone filtering (used when extreme_sampling=False)
    exclusion_n_std=2.0,
    # Extreme sampling (used when extreme_sampling=True)
    extreme_sampling=False,
    positive_percentile_lower=30,  # keep top X% of positives (50 = top half)
    negative_percentile_range=(10, 60),  # keep negatives in this percentile range
    # Balancing
    balance_classes=True,
    undersample_ratio=1,  # negatives per positive (1 = equal, 3 = 3:1)
    model_key=None,
):
    """
    Load all nuclei, split by WSI, filter training data, balance classes.

    Flow:
        1. Load all nuclei grouped by WSI (with dab values)
        2. Split WSIs into train/val (deterministic seed)
        3. Build val sampler — UNFILTERED (real-world distribution)
        4. Filter training data (extreme sampling OR exclusion zone)
        5. Balance training classes if requested
        6. Save and return both samplers
    """
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Step 0: If both files already exist, no need to create new.
    ## !! Remove, if starting from scratch
    train_sampler_name = f"train_sampler_{model_key}"
    if os.path.isfile(train_sampler_name):
        train_sampler = SampleManager(filename=train_sampler_name)
        val_sampler_name = f"val_sampler_{model_key}"
        if os.path.isfile(val_sampler_name):
            val_sampler = SampleManager(filename=val_sampler_name)
            print("Loaded train and val sampler - not loading all files again. ")
            return train_sampler, val_sampler

    # STEP 1: Load all nuclei grouped by WSI
    sampler_paths_list = _create_filelist(input_dir, fileextension)
    if use_subset:
        csv_path = input_dir + "/" + "ok_slides_path.csv"
        if os.path.isfile(csv_path):
            print("Subsetting to OK files.")
            subset_df = pd.read_csv(csv_path)
            wsi_names = (
                subset_df.results_filepath.str.split("/")
                .str[-1]
                .str.split(" ")
                .str[0]
                .to_list()
            )
        else:
            raise FileNotFoundError("use_subset used, but no subset found.")

    wsi_data = {}  # {wsi_name: [(s_x, s_y, s_key, dab_value), ...]}

    for path in sampler_paths_list:
        wsi_name = path.stem.split(" ")[0]
        if use_subset and wsi_name not in wsi_names:
            print(f"Skipping {wsi_name}, as quality doesnt suffice.")
            continue

        if wsi_name not in wsi_data:
            wsi_data[wsi_name] = []

        nuc_samples = SampleManager(path)
        nuc_samples.load_samples()

        for i, cell_dict in enumerate(nuc_samples.sample_xs):
            s_key = nuc_samples.sample_keys[i]

            # --- Extract label ---
            if label_key == "sample_ys":
                s_y = nuc_samples.sample_ys[i]
            elif label_key in cell_dict.keys():
                s_y = cell_dict[label_key]
                if type(s_y) == int:
                    if s_y not in (0, 1):
                        continue
                elif type(s_y) == list:
                    if len(s_y) != 1:
                        raise ValueError(
                            f"label is list with more than one element: {s_y}"
                        )
                    else:
                        s_y = s_y[0]
                else:
                    raise ValueError(
                        f"label is in wrong format {s_y} with type: {type(s_y)}"
                    )
            else:
                raise KeyError(
                    f"Couldnt find key for labels {label_key} in {path.stem} for nucleus {s_key}"
                )

            # --- Extract image data ---
            if celldict_key in cell_dict.keys():
                arr = np.asarray(cell_dict[celldict_key])
                if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
                    raise ValueError(
                        f"Bad sample at index {i}, key={s_key}, shape={arr.shape}"
                    )
                if arr.max() == 1:
                    arr = (arr.astype(np.uint8)) * 255
                s_x = arr
            elif celldict_key == "bbox_pixels":
                raise NotImplementedError(
                    "Adding raw pixels still needs to be implemented"
                )
            else:
                raise KeyError(
                    f"Couldnt find key for bbox {celldict_key} in {path.stem} for nucleus {s_key}"
                )

            # --- Extract dab intensity ---
            dab_value = None
            if "dab_intensity" in cell_dict and dab_key in cell_dict["dab_intensity"]:
                dab_value = cell_dict["dab_intensity"][dab_key]
                if np.isnan(dab_value):
                    dab_value = None

            wsi_data[wsi_name].append((s_x, s_y, s_key, dab_value))

    # Print summary of loaded data
    total_nuclei = sum(len(v) for v in wsi_data.values())
    total_pos = sum(
        1 for nuclei in wsi_data.values() for _, y, _, _ in nuclei if y == 1
    )
    print(
        f"\nLoaded {total_nuclei} nuclei from {len(wsi_data)} WSIs "
        f"({total_pos} positive, {total_nuclei - total_pos} negative)"
    )

    # STEP 2: Split WSIs into train/val (deterministic)
    all_wsi_names = sorted(wsi_data.keys())
    rng = np.random.RandomState(123)  # !! fixed seed for reproducible splits
    rng.shuffle(all_wsi_names)

    num_train_wsi = int(round(len(all_wsi_names) * train_val_ratio))
    train_wsi_names = all_wsi_names[:num_train_wsi]
    val_wsi_names = all_wsi_names[num_train_wsi:]

    print(f"\nWSI split: {len(train_wsi_names)} train, {len(val_wsi_names)} val")
    print(f"  Train WSIs: {sorted(train_wsi_names)}")
    print(f"  Val WSIs:   {sorted(val_wsi_names)}")

    # STEP 3: Build val sampler — UNFILTERED
    val_sampler_name = f"val_sampler_{model_key}"
    if os.path.isfile(val_sampler_name):
        os.remove(val_sampler_name)
        print("removed val")

    # VAL_MAX_NEGATIVES = 50000 / len(val_wsi_names)
    val_sampler = SampleManager(filename=val_sampler_name)
    n_negatives = 0
    for wsi_name in val_wsi_names:
        n_negatives = 0
        for s_x, s_y, s_key, dab_value in wsi_data[wsi_name]:
            # skip nuclei with no dab data
            if dab_value is None:
                continue

            # if s_y == 0:
            #     n_negatives += 1
            #     if n_negatives > VAL_MAX_NEGATIVES:
            #         break

            val_sampler.add(x=s_x, y=s_y, key=s_key)

    val_sampler.shuffle()

    val_pos = sum(1 for y in val_sampler.sample_ys if y == 1)
    val_neg = sum(1 for y in val_sampler.sample_ys if y == 0)

    # if len(val_sampler.sample_ys) > VAL_MAX_NEGATIVES:
    #     raise ValueError("Too many samples.")
    print(
        f"\n  Val (unfiltered): {len(val_sampler.sample_ys)} nuclei "
        f"({val_pos} pos, {val_neg} neg)"
    )

    # STEP 4: Filter TRAINING data only
    # Collect all training nuclei with dab values
    train_nuclei_all = []  # [(s_x, s_y, s_key, dab_value)]
    for wsi_name in train_wsi_names:
        for item in wsi_data[wsi_name]:
            s_x, s_y, s_key, dab_value = item
            if dab_value is None:
                continue
            train_nuclei_all.append(item)

    train_pos_all = [(x, y, k, d) for x, y, k, d in train_nuclei_all if y == 1]
    train_neg_all = [(x, y, k, d) for x, y, k, d in train_nuclei_all if y == 0]

    print(
        f"\n  Train before filtering: {len(train_nuclei_all)} nuclei "
        f"({len(train_pos_all)} pos, {len(train_neg_all)} neg)"
    )

    if extreme_sampling:

        # EXTREME SAMPLING: keep strongest positives + clear mid-range negatives
        print(f"\n=== Extreme sampling mode ===")

        # Positives: keep top N%
        pos_dab_values = np.array([d for _, _, _, d in train_pos_all])
        pos_cutoff = np.percentile(pos_dab_values, positive_percentile_lower)
        train_pos_filtered = [
            (x, y, k) for x, y, k, d in train_pos_all if d >= pos_cutoff
        ]

        print(
            f"  Positives: keeping DAB >= {pos_cutoff:.4f} "
            f"(top {100 - positive_percentile_lower}%): "
            f"{len(train_pos_filtered)} of {len(train_pos_all)}"
        )

        # Negatives: keep percentile range
        neg_dab_values = np.array([d for _, _, _, d in train_neg_all])
        neg_lower = np.percentile(neg_dab_values, negative_percentile_range[0])
        neg_upper = np.percentile(neg_dab_values, negative_percentile_range[1])
        train_neg_filtered = [
            (x, y, k) for x, y, k, d in train_neg_all if neg_lower <= d <= neg_upper
        ]

        print(
            f"  Negatives: keeping DAB in [{neg_lower:.4f}, {neg_upper:.4f}] "
            f"(p{negative_percentile_range[0]}-p{negative_percentile_range[1]}): "
            f"{len(train_neg_filtered)} of {len(train_neg_all)}"
        )

        train_pos_final = train_pos_filtered
        train_neg_final = train_neg_filtered

    else:

        # EXCLUSION ZONE: per-WSI, remove ambiguous nuclei between
        # (neg_mean + N*std) and positive_threshold

        print(f"\n=== Exclusion zone filtering (per-WSI, {exclusion_n_std}sigma) ===")

        train_pos_final = []
        train_neg_final = []
        total_excluded = 0

        for wsi_name in train_wsi_names:
            nuclei = wsi_data[wsi_name]

            # compute exclusion boundary from this WSI's negatives
            neg_dab = [
                d
                for _, _, _, d in nuclei
                if d is not None and not np.isnan(d) and d < positive_threshold
            ]

            if len(neg_dab) > 10:
                neg_mean = np.mean(neg_dab)
                neg_std = np.std(neg_dab)
                exclusion_lower = neg_mean + exclusion_n_std * neg_std
            else:
                exclusion_lower = positive_threshold

            excluded = 0
            for s_x, s_y, s_key, dab_value in nuclei:
                if dab_value is None:
                    continue
                # keep if clearly negative or clearly positive
                if dab_value <= exclusion_lower or dab_value >= positive_threshold:
                    if s_y == 1:
                        train_pos_final.append((s_x, s_y, s_key))
                    else:
                        train_neg_final.append((s_x, s_y, s_key))
                else:
                    excluded += 1

            total_excluded += excluded
            n_pos = sum(1 for _, y, _ in train_pos_final)  # running total
            print(
                f"  {wsi_name}: excl_zone=[{exclusion_lower:.4f}, {positive_threshold}], "
                f"excluded={excluded}"
            )

        print(f"  Total excluded: {total_excluded}")
        print(
            f"  After filtering: {len(train_pos_final)} pos, {len(train_neg_final)} neg"
        )

    # STEP 5: Balance training classes

    if balance_classes:
        n_pos = len(train_pos_final)
        n_neg = len(train_neg_final)
        target_neg = min(n_pos * undersample_ratio, n_neg)

        if n_neg > target_neg and n_pos > 0:
            indices = np.random.choice(n_neg, size=target_neg, replace=False)
            train_neg_final = [train_neg_final[i] for i in indices]
            print(
                f"\n  Train balanced: {n_neg} neg -> {target_neg} neg "
                f"(ratio {undersample_ratio}:1 with {n_pos} pos)"
            )
        else:
            print(f"\n  Train: no undersampling needed ({n_neg} neg, {n_pos} pos)")

    # STEP 6: Build train sampler and save

    train_sampler_name = f"train_sampler_{model_key}"
    if os.path.isfile(train_sampler_name):
        os.remove(train_sampler_name)
        print("removed train")

    train_sampler = SampleManager(filename=train_sampler_name)

    for s_x, s_y, s_key in train_pos_final:
        train_sampler.add(x=s_x, y=s_y, key=s_key)
    for s_x, s_y, s_key in train_neg_final:
        train_sampler.add(x=s_x, y=s_y, key=s_key)

    train_sampler.shuffle()

    train_pos = sum(1 for y in train_sampler.sample_ys if y == 1)
    train_neg = sum(1 for y in train_sampler.sample_ys if y == 0)
    print(f"\n=== Final dataset ===")
    print(
        f"  Train: {len(train_sampler.sample_ys)} nuclei ({train_pos} pos, {train_neg} neg)"
    )
    print(
        f"  Val:   {len(val_sampler.sample_ys)} nuclei ({val_pos} pos, {val_neg} neg)"
    )

    train_sampler.save_samples()
    val_sampler.save_samples()
    return train_sampler, val_sampler


class ResizeAndPad:
    """Resize so the diagonal fits within `size`, then pad to make square."""

    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        # This ensures the full image is preserved even at 45° rotation
        diagonal = (w**2 + h**2) ** 0.5
        scale = self.size / diagonal
        # -
        new_w, new_h = int(w * scale), int(h * scale)
        img = transforms.functional.resize(img, (new_h, new_w))

        # Pad to make it square
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        img = transforms.functional.pad(img, padding, fill=self.fill)

        return img


def pad_collate(batch):
    # Find max height and width in this batch
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    images = []
    for item in batch:
        img = item["image"]  # (C, H, W) after ToTensor
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        # pad order: (left, right, top, bottom)
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
        images.append(img)

    return {
        "image": torch.stack(images),
        "label": torch.stack([item["label"] for item in batch]),
        "key": [item["key"] for item in batch],
    }
