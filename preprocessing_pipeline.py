"""
Gets raw pixels for HE and intenstity from corresponding pX image.

1. Load SampleManager file
2. Load HE file using OpenSlide
3. Get raw pixels from original image (with 10x10 margin around it)
4. Run 'VALIS' on HE and pX image --> Aligns the two slides - outputs coordinates
5. Add aligned centroids, bbox and contours to each the nuclei
6. Load pX image
7. Deconvolute image --> Normalizes DAB intensity? !!
8. Load regions for each nucleus --> Evaluate intensity and add to dict (intensity only based on actual area of nucleus.)
9. Save file to disk
"""

import os

# limit BLAS/OpenMP threads
# !! otherwise this pipeline takes over a whole server.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

## Set number of available cores
N = 30

# detect available CPUs
total_cores = os.cpu_count()

# select last N cores
cpu_ids = set(range(total_cores - (N * 2), total_cores - N))

# apply affinity
os.sched_setaffinity(0, cpu_ids)

# GPU turned off - VALIS complains otherwise and we dont need it for this pipeline
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### NOTE: All above done deliberately before loading the packages!

from pathlib import Path
from openslide import OpenSlide


import shutil
import numpy as np
from tqdm import tqdm

# custom helpers
from helper_scripts.compressed_sampler import SampleManager
from helper_scripts.tools import create_filelist
from helper_scripts.add_raw_pixel_bbox import add_raw_pixels


import pyvips
from valis import registration

from color_deconvolution import (
    run_deconvolution_pipeline,
    export_labels_to_qupath_geojson,
)


def _get_pX_bbox(HE_slide, pX_slide, xy_coords):
    """Warp bbox coordinates from HE to pX space."""
    # bbox format is [[y1, x1], [y2, x2]] — VALIS expects (x, y)
    xy_flat = xy_coords.reshape(-1, 2)[:, ::-1]  # flip to (x, y)
    pX_bbox_flat = HE_slide.warp_xy_from_to(xy_flat, to_slide_obj=pX_slide)
    pX_bbox = pX_bbox_flat[:, ::-1].reshape(xy_coords.shape)  # flip back to (y, x)
    return pX_bbox


def _match_slides(registrar, HE_img_path: Path, pX_img_path: Path):
    """
    Match VALIS slide keys to HE and pX by filename stem.
    Falls back to index order [0]=HE, [1]=pX if matching fails.
    """
    slide_keys = list(registrar.slide_dict.keys())

    he_stem = HE_img_path.stem
    px_stem = pX_img_path.stem

    HE_slide, pX_slide = None, None
    for key in slide_keys:
        if he_stem in key or key in he_stem:
            HE_slide = registrar.get_slide(key)
        elif px_stem in key or key in px_stem:
            pX_slide = registrar.get_slide(key)

    # Fallback to original index-based behavior
    if HE_slide is None or pX_slide is None:
        print(f"  [VALIS] WARNING: Could not match by name, using index order.")
        print(f"    Keys: {slide_keys}")
        print(f"    HE stem: {he_stem}")
        print(f"    pX stem: {px_stem}")
        HE_slide = registrar.get_slide(slide_keys[0])
        pX_slide = registrar.get_slide(slide_keys[1])
    else:
        print(f"  [VALIS] Matched HE: {HE_slide.name}")
        print(f"  [VALIS] Matched pX: {pX_slide.name}")

    return HE_slide, pX_slide


def add_pX_coords(
    nuc_samples: SampleManager,
    HE_img_path: Path,
    pX_img_path: Path,
    xy_col_names: list = ["bbox", "centroid"],
):
    # Create a temp dir, that is deleted afterwards
    valis_temp_dir = Path("/lovelace/fabior/valis_temp_p21")
    valis_temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created {valis_temp_dir}")
    shutil.copy2(HE_img_path, valis_temp_dir)
    shutil.copy2(pX_img_path, valis_temp_dir)

    # Run VALIS with original defaults
    registrar = registration.Valis(
        str(valis_temp_dir),
        str(valis_temp_dir / "output"),
        create_masks=True,
        crop_for_rigid_reg=True,
    )
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    print(f"[VALIS] Registration error:\n{error_df}")

    slide_keys = list(registrar.slide_dict.keys())
    print(slide_keys)

    if len(slide_keys) != 2:
        raise ValueError(f"Expected 2 slides, got {len(slide_keys)}")

    # Name-safe slide matching
    HE_slide, pX_slide = _match_slides(registrar, HE_img_path, pX_img_path)

    # Define coords to add to HE_SampleManager file
    print("Adding new values to SampleManager file.")
    for column_name in tqdm(xy_col_names):
        pX_col_name = column_name + "_pX"

        # get coords from SampleManager file
        current_coords = list()
        for i, cell in enumerate(nuc_samples.sample_xs):
            current_coords.append(cell[column_name])
        xy_coords = np.array(current_coords)

        if "bbox" in column_name:  # as format is [[y1, x1], [y2, x2]]
            xy_in_pX = _get_pX_bbox(HE_slide, pX_slide, xy_coords)
        else:  # if format is just [[x,y],[x,y],[...]] - e.g. centroids.
            xy_in_pX = HE_slide.warp_xy_from_to(xy_coords, to_slide_obj=pX_slide)

        # finally add new coords to SampleManager file.
        for i, cell in enumerate(nuc_samples.sample_xs):
            if xy_coords[i].tolist() != cell[column_name]:
                print(xy_coords[i], cell[column_name])
                raise ValueError
            cell[pX_col_name] = xy_in_pX[i].tolist()

    # delete temp dir afterwards
    shutil.rmtree(valis_temp_dir)

    return nuc_samples


def main(
    NUSP_input: Path,
    HE_img_dir: Path,
    output_dir: Path,
    config: dict,
    pX_dir=Path(""),
    special_I0_dict: dict = {},
):
    sample_files = create_filelist(NUSP_input, fileextension="_results")
    print(len(sample_files))
    output_dir.mkdir(parents=True, exist_ok=True)

    error_files = list()
    for filepath in sample_files:
        # check if already exists.
        results_file = filepath.name.removesuffix("results") + "rawpix_results"
        if os.path.isfile(output_dir / results_file):
            print(f"Already analyzed {filepath.name} - continuing.")
            continue

        try:
            ### --- Search for HE and pX slide in directories --- ###
            wsi_prefix = filepath.name.split(" ")[0]
            print(wsi_prefix)
            wsi_prefix = wsi_prefix + "*"
            HE_wsi_path = next(HE_img_dir.glob(wsi_prefix))
            px_full_dir = HE_img_dir.parent / pX_dir
            pX_img_path = next(px_full_dir.glob(wsi_prefix))

            print(HE_wsi_path, pX_img_path)

            ### --- Add raw pixels from HE image --- ###
            new_nuc_samples = add_raw_pixels(
                HE_wsi_path, filepath, padding=10
            )  # not saving the file yet

            ### --- run VALIS --- ###
            new_nuc_samples = add_pX_coords(
                new_nuc_samples, HE_wsi_path, pX_img_path, ["bbox", "centroid"]
            )

            ### --- remove nuclei outside of pX bounds --- ###
            pX_reader = OpenSlide(str(pX_img_path))
            px_w, px_h = pX_reader.dimensions
            pX_reader.close()
            margin = 50

            rem_keys = list()
            before = len(new_nuc_samples.sample_xs)
            for i, nuc in enumerate(new_nuc_samples.sample_xs):
                if (
                    margin < nuc["centroid_pX"][0] < px_w - margin
                    and margin < nuc["centroid_pX"][1] < px_h - margin
                ):
                    pass
                else:
                    rem_keys.append(new_nuc_samples.sample_keys[i])

            for key in rem_keys:
                new_nuc_samples.remove(key)

            removed = before - len(new_nuc_samples.sample_xs)
            if removed > 0:
                print(f"Removed {removed}/{before} nuclei outside pX bounds")

            ### Same with bbox check - just to make sure.
            rem_keys_bbox = []
            for i, nuc in enumerate(new_nuc_samples.sample_xs):
                y1, x1 = nuc["bbox_pX"][0]
                y2, x2 = nuc["bbox_pX"][1]
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0 or x1 < 0 or y1 < 0:
                    rem_keys_bbox.append(new_nuc_samples.sample_keys[i])

            for key in rem_keys_bbox:
                new_nuc_samples.remove(key)

            if rem_keys_bbox:
                print(f"Removed {len(rem_keys_bbox)} nuclei with invalid bbox_pX")

            ### --- deconvolution pipeline --- ###
            # Read from config
            manual_threshold = config.get("manual_threshold")
            labeling_method = config.get("labeling_method")
            labeling_sigma = config.get("labeling_sigma")

            # Default to "negative_fit" if None - autodetermination of threshold.
            ## Best fallbackmethod, although others have been tried out (see 'run_decovolution_pipeline')
            if labeling_method is None:
                labeling_method = "negative_fit"
            if labeling_sigma is None:
                labeling_sigma = 5.0
            # !! Be aware that CUDA devices have been turned off!
            ## Load pX image, Deconvolute, load regions, evaluate intensity, add to dict.
            result = run_deconvolution_pipeline(
                pX_img_path,
                new_nuc_samples,
                key_bbox="bbox_pX",
                dab_od_threshold=config[
                    "dab_od_threshold"
                ],  # adapt for DAB - lower = broader
                refinement_strength=config[
                    "refinement_strength"
                ],  # blend NMF (0) vs angular (1)
                verbose=True,  # print output?
                labeling_sigma=labeling_sigma,
                I0=None,
                special_I0_dict=special_I0_dict,  # config
                labeling_method=labeling_method,
                manual_threshold=manual_threshold,
            )

            ### --- POSTPROCESSING --- ###

            labels = result["labels"]  # 0=negative, 1=senescent, -1=invalid - removed
            dab_results = result["dab_results"]

            # Inject labels back into nuclei
            for nuc, label, dab in zip(new_nuc_samples.sample_xs, labels, dab_results):
                nuc["senescence_label"] = int(
                    label
                )  # this is label for training (not sample_ys)
                nuc["dab_intensity"] = dab  # 5 different metrics

            # Remove invalid nuclei (label=-1) from the dataset
            invalid_keys = [
                new_nuc_samples.sample_keys[i]
                for i, label in enumerate(labels)
                if label == -1
            ]
            for key in invalid_keys:
                new_nuc_samples.remove(key)
            if invalid_keys:
                print(f"Removed {len(invalid_keys)} invalid nuclei (label=-1)")

            # Save as train input files.
            cwd = os.getcwd()
            os.chdir(output_dir)
            new_nuc_samples.save_samples()
            os.chdir(cwd)

            # optional to view in QuPath - better to do in jupyter notebook 'nuc_seg_check.ipynb'
            if 0:
                export_labels_to_qupath_geojson(
                    new_nuc_samples.sample_xs,
                    labels,
                    dab_results,
                    output_path=str(output_dir / "test.geojson"),
                )

        except Exception as e:
            error_files.append((filepath, e))
            print(f"ERROR: {e}")
    print(error_files)
    return True


if __name__ == "__main__":
    NUSP_input = Path("Outputfiles from CellVit++")
    HE_img_dir = NUSP_input.parent / "HE DIR"
    output_dir = NUSP_input.parent / ""
    pX_dir = Path("IHC stained slides")
    """
    Some images have a weird upper end - so for some we manually change the I0 background field
    """
    from config import p15_special_I0, p16_special_I0, p21_special_I0
    from config import p16_config, p15_config, p21_config

    main(
        NUSP_input,
        HE_img_dir,
        output_dir,
        pX_dir=pX_dir,
        config=p21_config,
        special_I0_dict=p21_special_I0,
    )
    print("Done")
