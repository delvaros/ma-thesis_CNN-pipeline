import predictor
import os
import yaml

#
# Settings
#

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#### CONFIG ####
DATASET = "p15"
IMG_PATH = f"SamplerDir"
OUT_PATH = f""
MODELS_PATH = f"/ktb_ihc_{DATASET}/model_training/models"
KEYS = ["v1"]
celldict_key = "bbox_pixels"  # or bbox_bitmap
WSI_PATH = ""


# default
varias = [None]
BASE_KEY = "virchow"


os.makedirs(OUT_PATH, exist_ok=True)

for vs in varias:
    if vs is None:
        EXTRA_KEY = BASE_KEY
    else:
        vs = list(vs.items())
        if len(vs) == 1:
            vs1 = vs[0]
            EXTRA_KEY = f"{BASE_KEY}-{vs1[0]}{vs1[1]}"
        else:
            vs1, vs2 = vs
            EXTRA_KEY = f"{BASE_KEY}-{vs1[0]}{vs1[1]}-{vs2[0]}{vs2[1]}"

    for key in KEYS:
        # !! hardcoded load config
        with open(
            f"/ktb_ihc_{DATASET}/model_training/{DATASET}_model_config.yaml"
        ) as f:
            config = yaml.safe_load(f)[key]

        val_conf = config["val_augmentations"]

        model_base_path = f"{MODELS_PATH}/{key}"
        predictor.predict(
            key,
            EXTRA_KEY,
            model_base_path,
            IMG_PATH,
            vs,
            OUT_PATH,
            val_conf,
            celldict_key=celldict_key,
            wsi_path=WSI_PATH,
        )
