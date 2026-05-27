from helper_scripts.compressed_sampler import SampleManager
from pathlib import Path
from openslide import OpenSlide
from tqdm import tqdm
import numpy as np
import os


def add_raw_pixels(
    wsi_path: Path,
    filepath: Path,
    filename: str = None,  # Defaults to f"{wsi_path.stem}_rawpix_results"
    padding: int = 10,  # in pixels around the bbox
):
    if filename == None:
        filename = f"{wsi_path.stem}_rawpix_results"

    nuc_samples = SampleManager(filepath)
    nuc_samples.load_samples()

    # load files
    wsi = OpenSlide(wsi_path)
    new_nuc_samples = SampleManager(filename=f"{wsi_path.stem}_rawpix_results")

    for i, cell in tqdm(enumerate(nuc_samples.sample_xs)):
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
        cur_y = nuc_samples.sample_ys[i]
        cur_key = nuc_samples.sample_keys[i]
        new_nuc_samples.add(x=cell, y=cur_y, key=cur_key)
    wsi.close()
    return new_nuc_samples


def _create_filelist(directory, fileextension=""):
    pickle_filelist = list()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(fileextension):
                pickle_filelist.append(Path(root) / file)

    return pickle_filelist


def export_list_txt(export_list: list, output_path):
    os.chdir(output_path)
    with open("failed_files.txt", "w") as f:
        for line in export_list:
            f.write(f"{line}\n")
    return True


def main(img_dir, pickle_dir, output_dir, fileextension=""):

    pickle_files = _create_filelist(pickle_dir)
    # loop through filelist, load
    failed_files = list()
    for pickle_path in pickle_files:
        try:
            wsi_name = pickle_path.name.removesuffix("_results") + fileextension
            wsi_path = img_dir / wsi_name
            new_nuc_samples = add_raw_pixels(wsi_path, pickle_path)

            # Save outside function.
            os.chdir(output_dir)
            new_nuc_samples.save_samples()
            print("WSI ", wsi_path.stem, " done.")
        except:
            failed_files.append(pickle_path)

    # Save txt to store failed attempts
    export_list_txt(failed_files, output_dir)
    return True
