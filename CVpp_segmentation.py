"""
CellViT++ Batch Processor for mrxs, svs and ndpi (and more probably) wsi
Runs CellViT++ segmentation tracking progress to allow restarts without reprocessing.
Requires custom CellViT++ fork.
"""

import os
import subprocess
import csv
import pandas as pd
from pathlib import Path
from openslide import OpenSlide

# !! Define GPU and ray directory
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["RAY_TMPDIR"] = (
    ".../ray_temp/"  # set manually as default one kept overflowing
)

### ------- CONFIG ------- ###
# dir of custom CellViT++
CVPP_DIR = Path("/.../CellViT-plus-plus")
CHECKPOINT = CVPP_DIR / "checkpoints" / "CellViT-Virchow-x40-AMP.pth"
FILE_EXTENSION = ""

INPUT_DIR = Path("")
OUTPUT_DIR = Path("")
# below: directory where CV++ json files are dumped - I never used them for anything.
CELLVIT_OUTDIR = OUTPUT_DIR / "CVpp-output"
ANALYSED_CSV = OUTPUT_DIR / "csv-Virchow_analysed.csv"

READ_ALL_METADATA = False
## Set to True to read metadata from every file individually
#### --> (a lot slower - only needed if files differ in magnification and resolution)
## Set to False to read one file and apply metadata to all files. (default)


### ------- Helper functions ------- ###
def get_analysed_paths() -> set:
    """Load set of already-analysed file paths from tracking CSV."""
    if not ANALYSED_CSV.exists():
        return set()
    df = pd.read_csv(ANALYSED_CSV)
    return set(df["path"].values)


def mark_as_analysed(file_paths: list):
    """Append newly analysed file paths to tracking CSV."""
    if not file_paths:
        return
    if ANALYSED_CSV.exists():
        df = pd.read_csv(ANALYSED_CSV)
    else:
        df = pd.DataFrame(columns=["path"])

    new = [p for p in file_paths if p not in df["path"].values]
    if new:
        df = pd.concat([df, pd.DataFrame({"path": new})], ignore_index=True)
        df.to_csv(ANALYSED_CSV, index=False)
        print(f"Marked {len(new)} files as analysed.")


def has_output_pickle(file_path: str) -> bool:
    """Check if a specific file already has a CellViT++ output pickle."""
    rel_path = Path(file_path).relative_to(INPUT_DIR)
    rel_parts = "_".join(rel_path.parent.parts)
    filename = Path(file_path).stem
    if rel_parts:
        expected_pickle = OUTPUT_DIR / f"{rel_parts}@{filename}_results"
    else:
        expected_pickle = OUTPUT_DIR / f"{filename}_results"
    return expected_pickle.exists()


def find_unprocessed_files() -> list:
    """Walk input directory and return sorted list of unprocessed image files."""
    analysed = get_analysed_paths()
    unprocessed = []
    if os.path.isfile(INPUT_DIR):
        return [INPUT_DIR]

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.lower().endswith(FILE_EXTENSION):
                continue

            full_path = str(Path(root) / file)

            # Skip if already in tracking CSV
            if full_path in analysed:
                continue

            # Skip if output pickle already exists (exact match)
            if has_output_pickle(full_path):
                continue

            unprocessed.append(full_path)

    unprocessed.sort()
    return unprocessed


def mark_completed_files(file_paths: list):
    """Check which files from the batch actually produced output,
    and mark those as analysed. This way, even if the batch crashes
    on file 15/50, files 1-14 get tracked."""
    completed = [p for p in file_paths if has_output_pickle(p)]
    if completed:
        mark_as_analysed(completed)
    failed = len(file_paths) - len(completed)
    if failed > 0:
        print(f"Warning: {failed} files did not produce output.")


### ------- Read metadata ------- ###
# implemented different versions for testing - openslide is default
# matches my implemented versions in PathoPatch. - .vsi slides dont work with OpenSlide.
def _read_metadata_openslide(path: str) -> tuple:
    slide = OpenSlide(path)
    mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    mag = float(slide.properties.get("openslide.objective-power", 40.0))
    slide.close()
    return mag, mpp


def _read_metadata_slideio(path: str) -> tuple:
    import slideio

    slide = slideio.open_slide(path, "AUTO")
    scene = slide.get_scene(0)
    res_x = scene.resolution[0]
    mpp = res_x * 1e6 if res_x > 0 else 0.25  # meters → microns
    mag = scene.magnification if scene.magnification > 0 else 40.0
    return mag, mpp


def _read_metadata(path: str) -> tuple:
    """Read metadata using the appropriate backend based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".vsi":  # vsi doesnt work with OpenSlide
        return _read_metadata_slideio(path)
    else:
        return _read_metadata_openslide(path)


def read_metadata_from_one(file_paths: list) -> list:
    first = file_paths[0]
    mag, mpp = _read_metadata(first)
    print(f"Metadata from {Path(first).name}: mag={mag}, mpp={mpp:.6f}")
    return [(p, mag, mpp) for p in file_paths]


def read_metadata_from_all(file_paths: list) -> list:
    results = []
    for p in file_paths:
        mag, mpp = _read_metadata(p)
        results.append((p, mag, mpp))
        print(f"  {Path(p).name}: mag={mag}, mpp={mpp:.6f}")
    return results


def create_filelist_csv(file_metadata: list, csv_path: Path):
    """Write a CellViT-compatible CSV filelist."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "magnification", "slide_mpp"])
        for path, mag, mpp in file_metadata:
            writer.writerow([path, mag, mpp])
    print(f"Created filelist with {len(file_metadata)} files: {csv_path}")


def run_cellvit(csv_path: Path) -> bool:
    """
    Execute CellViT++ on the filelist CSV. Returns True on success.
    Runs as a subprocess - this is the command that worked the best for me.
    Theoretically possible to subset csv and run in multiprocessing - probably not worth to implement.
    """
    command = [
        "python3",
        str(CVPP_DIR / "cellvit" / "detect_cells.py"),
        "--model",
        str(CHECKPOINT),
        "--outdir",
        str(CELLVIT_OUTDIR),
        "--reference_dir",
        str(INPUT_DIR),
        "process_dataset",
        "--filelist",
        str(csv_path),
        "--wsi_extension",
        FILE_EXTENSION.lstrip("."),
    ]

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"CellViT++ error: {e}")
        return False


### ------- Main execution loop ------- ###
def main():
    CELLVIT_OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "filelist_virchow.csv"  # !! hardcoded.
    unprocessed = find_unprocessed_files()

    if not unprocessed:
        print(f"No unprocessed files found")
        return

    print(f"Found {len(unprocessed)} unprocessed files.")

    # Read metadata from wsi
    if READ_ALL_METADATA:
        file_metadata = read_metadata_from_all(unprocessed)
    else:
        file_metadata = read_metadata_from_one(unprocessed)

    create_filelist_csv(file_metadata, csv_path)
    run_cellvit(csv_path)

    # Always check what actually completed (handles partial failures)
    mark_completed_files(unprocessed)
    print("Done")


if __name__ == "__main__":
    main()
