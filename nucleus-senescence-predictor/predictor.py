import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

import numpy as np
from PIL import Image

from sampler import SampleManager

import models as models
import data_nuclei as data_nuclei
import os, glob
from pathlib import Path
import copy
from openslide import OpenSlide


from data_nuclei import _create_filelist, pad_collate
import sys

#
# Settings
#

DEV_MODE = 0
GPU = 1  # not number of GPU, but if used or not.

#
#

MODEL = "xception"
SIZE = 299
ENSEMBLE = 1
BATCH_SIZE = 256
DROPOUT_RATE = 0.6
OUT_BINS = 2


ENSEMBLE_START = 0
ENSEMBLE = 1
########


def _add_raw_pixels(
    sampler: SampleManager,
    wsi_path: Path,
    padding: int = 10,  # in pixels around the bbox
):

    # load files
    wsi = OpenSlide(wsi_path)
    sampler_raw = SampleManager(".../temp/temp_raw.pkl")

    for i, cell in enumerate(sampler.sample_xs):
        inst_bbox = cell["bbox"]  # Shape (2,2) with [[y1,x1],[y2,x2]]
        y1, x1, y2, x2 = (
            int(inst_bbox[0][0] - padding),
            int(inst_bbox[0][1] - padding),
            int(inst_bbox[1][0] + padding),
            int(inst_bbox[1][1] + padding),
        )
        width, height = int(x2 - x1), int(y2 - y1)
        region = wsi.read_region((x1, y1), 0, (width, height))
        pixel_array = np.array(region)[:, :, :3]
        cell["bbox_pixels"] = pixel_array

        # Add information to new samplemanager object
        cur_y = sampler.sample_ys[i]
        cur_key = sampler.sample_keys[i]
        sampler_raw.add(x=cell, y=cur_y, key=cur_key)
    wsi.close()
    return sampler_raw


####################################################################


def predict_imgs(model, img_dataset, device):
    criterion = torch.nn.BCEWithLogitsLoss()

    model.eval()

    y_pred, y_key = [], []

    img_loader = DataLoader(
        img_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=pad_collate,
    )

    with torch.no_grad():

        progress_bar = tqdm(img_loader, unit="batch", leave=False, desc="Batches")

        for batch in progress_bar:
            images = batch["image"].to(device, non_blocking=True)
            keys = batch["key"]

            outputs = model(images)

            # take 2nd as sen score
            preds = torch.softmax(outputs, dim=1)[:, 1]
            y_pred.extend(preds.cpu().tolist())
            y_key.extend(keys)

            progress_bar.set_postfix(
                predicted=len(y_pred), batch_mean=f"{preds.mean().item():.3f}"
            )

    return y_pred, y_key


def predict(
    key,
    extra_key,
    model_base_path,
    all_img_paths,
    data_mod,
    out_path,
    val_conf=None,
    celldict_key="bbox_pixels",
    wsi_path=None,
):
    model_weights_path = f"{model_base_path}/model_weights-KEY.pth"
    out_scores_path = (
        f"{out_path}/nusp-KEY-{extra_key}_p15.csv"  # !! potentially hardcoded
    )
    print("output goes to:", out_scores_path.replace("KEY", key))

    for ens_idx in range(ENSEMBLE_START, ENSEMBLE):
        model = models.get_model(
            MODEL, OUT_BINS, DROPOUT_RATE, custom=True
        )  # !! change back to true
        if GPU:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        enspath = f"-e{ens_idx}" if ENSEMBLE > 1 else ""
        mwpath = model_weights_path.replace("KEY", f"{key}-best{enspath}")
        print("Loading model weights from", mwpath)
        model.load_state_dict(torch.load(mwpath, map_location=device))
        model.to(device)

        img_paths = _create_filelist(all_img_paths, "_results")
        print(f"Processing: {len(img_paths)} image files from {all_img_paths}")
        if len(img_paths) == 0:
            raise ValueError(f"No images found at {all_img_paths}")

        ospath = out_scores_path.replace("KEY", f"{key}{enspath}")

        already_predicted = set()
        header_written = False
        total_written = 0

        if os.path.isfile(ospath):
            existing_df = pd.read_csv(ospath)
            total_written = len(existing_df)
            header_written = True

            # Extract slide identifiers from existing keys
            for k in existing_df["key"].values:
                # !! TODO: Confirm that its universally correct
                # Split off the nucleus coordinates at the end
                # Keep everything up to the slide identifier
                parts = k.rsplit("_", 2)  # split from right, max 2 splits
                if len(parts) >= 3:
                    slide_id = parts[0]
                else:
                    slide_id = k
                already_predicted.add(slide_id)

            print(
                f"Resuming: found {total_written} existing predictions from {len(already_predicted)} slides in {ospath}"
            )

        for img_path in tqdm(img_paths):
            # skip already predicted slides
            filename_stem = Path(img_path).stem
            # Remove suffixes like "_rawpix_results" or "_results"
            slide_id = filename_stem.replace("_rawpix_results", "").replace(
                "_results", ""
            )

            if slide_id in already_predicted:
                # tqdm.write(f"Skipping (already predicted): {slide_id}")
                continue

            print(f"Predicting image: {img_path}")

            sampler = _load_sampler(
                str(img_path), cell_dict_key=celldict_key, wsi_directory=wsi_path
            )

            if len(sampler.sample_xs) == 0:
                print("**** Empty samples:", img_path)
                continue

            img_dataset = data_nuclei.prep_dataset(
                sampler,
                data_nuclei.val_transforms(val_conf, SIZE, data_mod),
                DEV_MODE,
                training=False,
            )

            preds, keys = predict_imgs(model, img_dataset, device)
            # --- relative parts if necessary ---
            # rel_path = Path(img_path).relative_to(all_img_paths)
            # rel_parts = "_".join(rel_path.parent.parts)
            # if rel_parts != "":
            #     keys = [rel_parts + "@" + key for key in keys]
            # ---

            # save incrementally after each image
            results = pd.DataFrame({"key": keys, "prediction": preds})
            if not header_written:
                results.to_csv(ospath, index=False, mode="w")
                header_written = True
            else:
                results.to_csv(ospath, index=False, mode="a", header=False)

            total_written += len(preds)
            print(f"  Saved {len(preds)} predictions (total: {total_written})")
            # -

            if DEV_MODE:
                break

        print(f"Done. Total predictions written to {ospath}: {total_written}")


def _load_sampler(img_path, wsi_directory=None, cell_dict_key="bbox_pixels"):
    sampler_full = SampleManager(str(img_path))
    sampler_full.load_samples()
    if os.path.isfile("temp.pkl"):
        os.remove("temp.pkl")
    sampler = SampleManager(filename="temp.pkl")

    if cell_dict_key in sampler_full.sample_xs[0]:
        sampler_raw = copy.deepcopy(sampler_full)
        pass

    elif cell_dict_key == "bbox_pixels":
        # !! TODO: Make this more universally functional!
        if wsi_directory is None:
            raise KeyError(
                "Wsi directory needs to be specified when wanting to use raw pixels."
            )
        wsi_name = Path(img_path).stem.removesuffix("_results") + ".mrxs"
        wsi_path = Path(wsi_directory) / wsi_name
        if not os.path.isfile(wsi_path):
            raise FileNotFoundError(f"{wsi_path} doesnt exist!")
        print(f"Adding raw pixel data from wsi directory: {wsi_path}")
        sampler_raw = _add_raw_pixels(sampler_full, wsi_path=wsi_path)

    else:
        raise KeyError(f"Key {cell_dict_key} not found and can't be added.")

    for i, cell_dict in enumerate(sampler_raw.sample_xs):
        s_key = sampler_raw.sample_keys[i]
        s_y = sampler_raw.sample_ys[i]

        arr = np.asarray(cell_dict[cell_dict_key])

        if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
            raise ValueError(f"Bad sample at index {i}, key={s_key}, shape={arr.shape}")
        if arr.max() == 1:  # bitmap input
            arr = (arr.astype(np.uint8)) * 255

        s_x = arr

        sampler.add(x=s_x, y=s_y, key=s_key)

    return sampler
