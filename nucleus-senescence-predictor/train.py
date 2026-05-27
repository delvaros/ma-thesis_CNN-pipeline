import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr


import pandas as pd
import numpy as np
from PIL import Image
import os
from collections import defaultdict
from pathlib import Path
import yaml

from sampler import SampleManager

import models as models
import data_nuclei as data_nuclei
import utils


import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

#
# Settings
#
DEV_MODE = 0


# detect available CPUs
# N = 60
# N_offset = 0
# total_cores = os.cpu_count()
# cpu_ids = set(range(total_cores - (N + N_offset), total_cores - N_offset))
# os.sched_setaffinity(0, cpu_ids)
# -
# os.environ["OMP_NUM_THREADS"] = str(N)
# os.environ["OPENBLAS_NUM_THREADS"] = str(N)
# os.environ["MKL_NUM_THREADS"] = str(N)
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(N)
# os.environ["NUMEXPR_NUM_THREADS"] = str(N)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATASET = "p15"

SEN_MODEL = f""
label_key = "senescence_label"

KEYS = ["v1"]

for KEY in KEYS:
    # Load config
    with open(f"/ktb_ihc_{DATASET}/model_training/{DATASET}_model_config.yaml") as f:
        config = yaml.safe_load(f)[KEY]

    train_trans_cfg = config["train_augmentations"]
    val_trans_cfg = config["val_augmentations"]
    SIZE = int(config["SIZE"])
    MODEL = config["MODEL"]
    LEARNING_RATE = float(config["LEARNING_RATE"])
    WEIGHT_DECAY = float(config["WEIGHT_DECAY"])
    BALANCE_CLASS_WEIGHT = config["BALANCE_CLASS_WEIGHT"]
    BALANCE_SAMPLES = config["BALANCE_SAMPLES"]
    BALANCE_CLASSES = config["BALANCE_CLASSES"]
    DROPOUT_RATE = config["DROPOUT_RATE"]
    OUT_BINS = config["OUT_BINS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    TRAIN_SUBSET = config["TRAIN_SUBSET"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    ENSEMBLE = config["ENSEMBLE"]
    EARLY_STOPPING_PATIENCE = 30
    FREEZE_BACKBONE = config["FREEZE_BACKBONE"]
    BBOX_TYPE = config["BBOX_TYPE"]
    # -

    IMPORT_DIR = f"/ktb_ihc_{DATASET}/results_Apr26_v2/"
    OUT_BASE = f"/ktb_ihc_{DATASET}/model_training/models/{KEY}"

    MODEL_WEIGHTS_PATH = f"{OUT_BASE}/model_weights-KEY.pth"

    if BBOX_TYPE == "RAW":
        cell_dict_key = "bbox_pixels"
    elif BBOX_TYPE == "BITMAP":
        cell_dict_key = "bbox_bitmap"
    else:
        raise KeyError("BBOX Type not found - check config. ")

    ####################################################################

    def train_phase(model, train_dataset, device, optimizer, criterion):
        running_loss = 0.0

        y_true, y_pred, y_probs, y_key = [], [], [], []

        subset_size = int(TRAIN_SUBSET * len(train_dataset))
        subset_sampler = data_nuclei.get_training_sampler(
            train_dataset, subset_size, BALANCE_SAMPLES
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=subset_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=data_nuclei.pad_collate,
        )

        progress_bar = tqdm(train_loader, desc=f"Train", unit="batch", leave=False)
        model.train()

        for batch in progress_bar:
            optimizer.zero_grad()

            images = batch["image"]
            keys = batch["key"]

            images = images.to(device)
            labels_tensor = batch["label"]

            outputs = model(images)

            # labels_tensor = labels_tensor.unsqueeze(1)
            loss = criterion(outputs, labels_tensor.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            outputs = outputs.detach()
            preds = outputs.cpu()

            probs = torch.softmax(preds, dim=1)[:, 1]
            y_probs.extend(probs.numpy())
            preds = torch.argmax(preds, dim=1)

            progress_bar.set_postfix(loss=loss.item())

            y_true.extend(labels_tensor.numpy())
            y_pred.extend(preds.numpy())
            y_key.extend(keys)

        train_loss = running_loss / len(train_loader.dataset)

        return train_loss, y_true, y_pred, y_probs, y_key

    #
    # validation phase
    #
    def validate_phase(model, val_dataset, device, criterion):
        model.eval()
        val_loss = 0.0

        y_true, y_pred, y_probs, y_key = [], [], [], []

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=data_nuclei.pad_collate,
        )

        with torch.no_grad():

            progress_bar = tqdm(
                val_loader, desc=f"Validation", unit="batch", leave=False
            )

            for batch in progress_bar:
                images = batch["image"]
                keys = batch["key"]

                images = images.to(device)
                labels_tensor = batch["label"]

                outputs = model(images)

                loss = criterion(outputs, labels_tensor.to(device))

                val_loss += loss.item() * images.size(0)

                outputs = outputs.detach()
                preds = outputs.cpu()
                probs = torch.softmax(preds, dim=1)[:, 1]
                y_probs.extend(probs.numpy())

                preds = torch.argmax(preds, dim=1)

                progress_bar.set_postfix(loss=loss.item())

                y_true.extend(labels_tensor.numpy())
                y_pred.extend(preds.numpy())
                y_key.extend(keys)

        val_loss /= len(val_loader.dataset)

        return val_loss, y_true, y_pred, y_probs, y_key

    #
    #
    #
    #

    if __name__ == "__main__":
        train_sampler, val_sampler = data_nuclei._load_all_samplers(
            IMPORT_DIR,
            Path(OUT_BASE).parent / "train_datasets",
            fileextension="_results",
            celldict_key=cell_dict_key,
            label_key=label_key,
            use_subset=False,
            dab_key="dab_mean",
            positive_threshold=0.25,
            exclusion_n_std=2.0,
            extreme_sampling=True,
            positive_percentile_lower=60,
            negative_percentile_range=(10, 60),
            balance_classes=True,
            undersample_ratio=3,
            model_key=KEY,
        )

        train_dataset = data_nuclei.prep_dataset(
            train_sampler,
            data_nuclei.train_transforms(train_trans_cfg, SIZE),
            DEV_MODE,
            sen_model=SEN_MODEL,
        )
        val_dataset = data_nuclei.prep_dataset(
            val_sampler,
            data_nuclei.val_transforms(val_trans_cfg, SIZE),
            DEV_MODE,
            sen_model=SEN_MODEL,
        )

        print("Training: ", MODEL_WEIGHTS_PATH.replace("KEY", KEY))

        key = KEY

        top_save_metrics, final_save_metrics = [], []

        for ens_idx in range(ENSEMBLE):
            model = models.get_model(MODEL, OUT_BINS, DROPOUT_RATE, FREEZE_BACKBONE)

            device = torch.device("cuda")
            model.to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            #### !! Different schedulers - decide which one works best for you
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
            # )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer,
            #     mode="max",
            #     factor=0.5,
            #     patience=5,
            # )
            # Warm up scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-7
            )
            # -

            if BALANCE_CLASS_WEIGHT:
                train_labels = np.array(train_dataset.labels)
                class_weights = compute_class_weight(
                    "balanced", classes=np.unique(train_labels), y=train_labels
                )
                class_weights[1] *= BALANCE_CLASS_WEIGHT  # multiplier from config

                print(f"Class weights: {class_weights}")
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
                criterion = torch.nn.CrossEntropyLoss(
                    weight=class_weights_tensor.to(device),
                    label_smoothing=0.1,  # !! added for preventing overconfidence.
                )
            else:
                criterion = torch.nn.CrossEntropyLoss()

            val_criterion = torch.nn.CrossEntropyLoss()  # unweighted for val
            # -

            save_metric, last_save_metric = 0, 0

            # Early stopping
            patience = EARLY_STOPPING_PATIENCE
            epochs_without_improvement = 0
            # -
            # TensorBoard logging
            os.makedirs(OUT_BASE, exist_ok=True)
            writer = SummaryWriter(log_dir=f"{OUT_BASE}/tensorboard")
            # -

            for epoch in range(NUM_EPOCHS):

                for param_group in optimizer.param_groups:
                    for p in param_group["params"]:
                        state = optimizer.state[p]
                        if state:
                            print(
                                "Momentum buffer sum:",
                                state["exp_avg"].abs().sum().item(),
                            )
                            print(
                                "Variance buffer sum:",
                                state["exp_avg_sq"].abs().sum().item(),
                            )
                            break
                    break

                train_loss, train_true, train_pred, train_probs, train_key = (
                    train_phase(model, train_dataset, device, optimizer, criterion)
                )
                val_loss, val_true, val_pred, val_probs, val_key = validate_phase(
                    model, val_dataset, device, val_criterion
                )

                ensinfo = f"Ensemble {ens_idx} | " if ENSEMBLE > 1 else ""

                print(
                    f"\033[1m*** {ensinfo} Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f} [{key}]\033[0m"
                )

                # auc_score = roc_auc_score(val_true, val_pred)
                try:
                    auc_score = roc_auc_score(val_true, val_probs)
                except ValueError:
                    auc_score = 0.0
                # -
                precision = precision_score(val_true, val_pred)
                recall = recall_score(val_true, val_pred)
                f1 = f1_score(val_true, val_pred)

                # sensitivity, specificity, confusion matrix
                cm = confusion_matrix(val_true, val_pred)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # = recall
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Add (weighted) youdens j
                youden_j = sensitivity + specificity - 1
                spec_weight = 3.0  # how much more you value specificity
                weighted_j = (sensitivity + spec_weight * specificity) / (
                    1 + spec_weight
                ) - 0.5

                # Target-specific save metric
                save_metric = specificity + 0.3 * sensitivity

                writer.add_scalar("Metrics/ConstrainedMetric", save_metric, epoch)

                print(
                    f"=== Val AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )

                # log confusion matrix as image to TensorBoard
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                im = ax_cm.imshow(cm, cmap="Blues")
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                ax_cm.set_xticklabels(["Pred Neg", "Pred Pos"])
                ax_cm.set_yticklabels(["True Neg", "True Pos"])
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title(f"Epoch {epoch+1}")

                # Add numbers to cells
                for i in range(2):
                    for j in range(2):
                        color = "white" if cm[i, j] > cm.max() / 2 else "black"
                        ax_cm.text(
                            j,
                            i,
                            f"{cm[i, j]}",
                            ha="center",
                            va="center",
                            fontsize=14,
                            color=color,
                        )

                fig_cm.tight_layout()
                writer.add_figure("ConfusionMatrix/val", fig_cm, epoch)
                plt.close(fig_cm)
                # -

                # class-level accuracy for monitoring
                train_pos_acc = sum(
                    1 for t, p in zip(train_true, train_pred) if t == 1 and p == 1
                ) / max(sum(1 for t in train_true if t == 1), 1)
                train_neg_acc = sum(
                    1 for t, p in zip(train_true, train_pred) if t == 0 and p == 0
                ) / max(sum(1 for t in train_true if t == 0), 1)
                val_pos_acc = sum(
                    1 for t, p in zip(val_true, val_pred) if t == 1 and p == 1
                ) / max(sum(1 for t in val_true if t == 1), 1)
                val_neg_acc = sum(
                    1 for t, p in zip(val_true, val_pred) if t == 0 and p == 0
                ) / max(sum(1 for t in val_true if t == 0), 1)

                # Slide level evaluation
                wsi_stats = defaultdict(lambda: {"true": [], "probs": []})
                for k, t, p in zip(val_key, val_true, val_probs):
                    wsi = k.split("_")[0]  # adjust parsing
                    wsi_stats[wsi]["true"].append(t)
                    wsi_stats[wsi]["probs"].append(p)

                true_rates = []
                pred_rates = []
                for wsi in sorted(wsi_stats.keys()):
                    true_rates.append(np.mean(wsi_stats[wsi]["true"]))
                    pred_rates.append(
                        # np.mean(np.array(wsi_stats[wsi]["probs"]) >= best_threshold)
                        np.mean(np.array(wsi_stats[wsi]["probs"]) >= 0.5)
                    )

                if len(true_rates) > 2:
                    slide_corr, _ = spearmanr(true_rates, pred_rates)
                else:
                    slide_corr = 0.0
                # -
                train_acc = sum(
                    1 for t, p in zip(train_true, train_pred) if t == p
                ) / len(train_true)
                val_acc = sum(1 for t, p in zip(val_true, val_pred) if t == p) / len(
                    val_true
                )

                bal_acc = balanced_accuracy_score(val_true, val_pred)

                # log to TensorBoard
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Metrics/AUC", auc_score, epoch)
                writer.add_scalar("Metrics/Precision", precision, epoch)
                writer.add_scalar("Metrics/Recall", recall, epoch)
                writer.add_scalar("Metrics/F1", f1, epoch)
                writer.add_scalar("Metrics/Sensitivity", sensitivity, epoch)
                writer.add_scalar("Metrics/Specificity", specificity, epoch)
                writer.add_scalar("Metrics/YoudenJ", youden_j, epoch)
                writer.add_scalar("Metrics/WeightedJ", weighted_j, epoch)
                writer.add_scalar("Accuracy/train_positive", train_pos_acc, epoch)
                writer.add_scalar("Accuracy/train_negative", train_neg_acc, epoch)
                writer.add_scalar("Accuracy/val_positive", val_pos_acc, epoch)
                writer.add_scalar("Accuracy/val_negative", val_neg_acc, epoch)
                writer.add_scalar("Accuracy/train_overall", train_acc, epoch)
                writer.add_scalar("Accuracy/val_overall", val_acc, epoch)
                writer.add_scalar("Metrics/BalancedAccuracy", bal_acc, epoch)
                writer.add_scalar("Metrics/SlideCorrelation", slide_corr, epoch)
                # -

                o = utils.cm_breakdown(train_true, train_pred)
                print(f".. Train: {o}")
                o = utils.cm_breakdown(val_true, val_pred)
                print(f".. Valid: {o}")

                # !! Decide the save-metric for best epoch eval
                # save_metric = auc_score
                # save_metric = youden_j
                # save_metric = weighted_j
                # save_metric = specificity

                enspath = f"-e{ens_idx}" if ENSEMBLE > 1 else ""
                torch.save(
                    model.state_dict(),
                    MODEL_WEIGHTS_PATH.replace("KEY", f"{key}-last{enspath}"),
                )

                if save_metric > last_save_metric:
                    last_save_metric = save_metric
                    epochs_without_improvement = 0
                    print(f">>>>>> Model improved - saving weights: {save_metric:.4f}")
                    enspath = f"-e{ens_idx}" if ENSEMBLE > 1 else ""
                    torch.save(
                        model.state_dict(),
                        MODEL_WEIGHTS_PATH.replace("KEY", f"{key}-best{enspath}"),
                    )

                    output_file = f"{OUT_BASE}/{key}{enspath}-val_out.csv"
                    utils.dump_predictions(output_file, val_true, val_pred, val_key)

                    # mark best epoch in TensorBoard
                    writer.add_scalar("BestModel/saved", 1, epoch)
                    writer.add_scalar("BestModel/best_AUC", save_metric, epoch)
                else:
                    epochs_without_improvement += 1
                    # log 0 so the line is continuous
                    writer.add_scalar("BestModel/saved", 0, epoch)
                    # -
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                # !! Depending on scheduler, choose the stepping metric --> You can also set to 'save_metric'
                scheduler.step()
                # scheduler.step(auc_score)
                # !! added for warmup training.
                # warmup_epochs = 5
                # if epoch < warmup_epochs:
                #     warmup_lr = LEARNING_RATE * (epoch + 1) / warmup_epochs
                #     for param_group in optimizer.param_groups:
                #         param_group["lr"] = warmup_lr
                # else:
                #     scheduler.step()

                # scheduler.step(youden_j)
                # scheduler.step(weighted_j)

                # Log current LR in TensorBoard
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("Training/LearningRate", current_lr, epoch)

                print()

            top_save_metrics.append(last_save_metric)
            final_save_metrics.append(save_metric)
            writer.close()
            # -

            print("Top save metrics:", top_save_metrics)
            print("Final save metrics:", final_save_metrics)
