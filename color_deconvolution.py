"""
Color Deconvolution for H&E + DAB Stained WSI
=============================================================

Automatic stain matrix estimation and deconvolution for slides
restained with senescence markers (p15/p16/p21) after H&E.

Strategy:
---------
1. Extract pixels from nuclei regions only (avoids background/fat/artifacts)
2. Convert to Optical Density (OD) space
3. Multi-strategy stain matrix estimation:
   a) Reference-initialized NMF(3) — uses known H/E/DAB vectors as starting
      point so NMF converges correctly even when eosin is faint
   b) Angular extrema refinement — finds pure-stain pixels by direction in OD space
   c) Spectral DAB targeting — uses blue-channel OD ratio to find brown pixels
4. Full 3-channel deconvolution with the estimated matrix
5. Per-nucleus DAB intensity quantification

Key design decisions:
- Uses nuclei-only sampling to avoid the silvery wash artifact in background
- Reference-guided NMF(3) handles the weak-eosin problem on restained slides
- Multiple independent DAB estimation strategies with cross-validation
- Quality checks prevent vector collapse (two stains becoming identical)

Additional NOTE:
- Some things in this file are hardcoded for the thesis - they are marked with comments; search for: '# !!'
"""

import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from typing import Optional
import pickle
import warnings

import matplotlib.pyplot as plt
from openslide import OpenSlide

from helper_scripts.compressed_sampler import SampleManager

# =============================================================================
# Reference stain vectors (Ruifrok & Johnston, normalized OD vectors)
# Used for matching, NOT as the final deconvolution matrix
# =============================================================================
REFERENCE_VECTORS = {
    "hematoxylin": np.array([0.6500286, 0.704031, 0.2860126]),
    "eosin": np.array([0.07200160, 0.9900224, 0.10502519]),
    "dab": np.array([0.268, 0.570, 0.776]),
}


# =============================================================================
# Core OD / RGB conversion utilities
# =============================================================================


def rgb_to_od(rgb: np.ndarray, I0: float = 240.0, clip_min: float = 1e-6) -> np.ndarray:
    """
    Convert RGB image (uint8 or float [0,255]) to Optical Density.
    OD = -log10(I / I0)
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    ratio = np.clip(rgb / I0, clip_min, 1.0)
    return -np.log10(ratio)


def od_to_rgb(od: np.ndarray, I0: float = 240.0) -> np.ndarray:
    """Convert OD back to RGB [0, 255] uint8."""
    rgb = I0 * np.power(10, -od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def normalize_stain_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a stain vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


# =============================================================================
# Illumination estimation
# =============================================================================


def estimate_I0_from_background(
    slide_reader,
    sample_coords: list[tuple[int, int]],
    patch_size: int = 512,
    percentile: float = 99.0,
) -> np.ndarray:
    """
    Estimate per-channel illumination I0 from bright background regions.

    Parameters
    ----------
    slide_reader : openslide.OpenSlide
    sample_coords : list of (x, y) tuples
        Coordinates of known background regions (or random sample).
    patch_size : int
    percentile : float
        Use high percentile of bright pixels as I0.
    """
    all_pixels = []
    for x, y in sample_coords:
        patch = np.array(slide_reader.read_region((x, y), 0, (patch_size, patch_size)))[
            ..., :3
        ]  # drop alpha
        all_pixels.append(patch.reshape(-1, 3))

    all_pixels = np.concatenate(all_pixels, axis=0).astype(np.float64)

    # Use bright pixels (likely background)
    brightness = all_pixels.mean(axis=1)
    bright_mask = brightness > np.percentile(brightness, 80)
    bright_pixels = all_pixels[bright_mask]

    I0 = np.percentile(bright_pixels, percentile, axis=0)
    return I0


# =============================================================================
# Pixel extraction from nuclei
# =============================================================================


def extract_nuclei_pixels_from_slide(
    slide_reader,
    nuclei: list[dict],
    margin: int = 10,
    key_bbox: str = "bbox_pX",
    max_nuclei: Optional[int] = None,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    Extract RGB pixels from the pX slide for all detected nuclei.

    Parameters
    ----------
    slide_reader : openslide.OpenSlide - pX slide.
    nuclei : list of dict
    margin : int
        Extra pixels around each bounding box to capture peri-nuclear stain.
    key_bbox : str
        Key in nuclei dict for the bounding box on the pX slide.
    max_nuclei : int or None
        If set, randomly subsample this many nuclei - 200K enough for assessment.
    rng_seed : int

    Returns
    -------
    pixels : array of shape (N, 3), uint8 RGB values
    """
    rng = np.random.default_rng(rng_seed)

    if max_nuclei is not None and len(nuclei) > max_nuclei:
        indices = rng.choice(len(nuclei), max_nuclei, replace=False)
        nuclei_subset = [nuclei[i] for i in indices]
    else:
        nuclei_subset = nuclei

    all_pixels = []
    print(f"nuclei {len(nuclei)}")

    for nuc in nuclei_subset:
        bbox = nuc[key_bbox]  # [[y1, x1], [y2, x2]]
        y1, x1 = bbox[0]
        y2, x2 = bbox[1]

        # Add margin
        x1_m = max(0, x1 - margin)
        y1_m = max(0, y1 - margin)
        w = (x2 + margin) - x1_m
        h = (y2 + margin) - y1_m

        w = int(round(w))
        h = int(round(h))
        if w <= 0 or h <= 0:
            continue

        try:
            patch = np.array(
                slide_reader.read_region((int(x1_m), int(y1_m)), 0, (int(w), int(h)))
            )[..., :3]
            all_pixels.append(patch.reshape(-1, 3))
        except Exception as e:
            raise Exception(e)
            continue

    if not all_pixels:
        raise ValueError("No pixels could be extracted from nuclei regions.")

    return np.concatenate(all_pixels, axis=0)


def flag_low_tissue_nuclei(
    slide_reader,
    nuclei: list[dict],
    key_bbox: str = "bbox_pX",
    I0: float | np.ndarray = 240.0,
    min_tissue_fraction: float = 0.3,
    verbose: bool = True,
) -> np.ndarray:
    """
    Flag nuclei where the pX bbox has mostly background.

    Aimed to catch these cases:
    - VALIS misalignment (bbox lands on wrong region)
    - bbox on fatty tissue gap (no actual cell there)
    - Missing tissue regions after washing and restaining

    Returns bool mask: True = keep, False = discard.
    """
    from color_deconvolution_v2 import rgb_to_od

    n = len(nuclei)
    keep = np.ones(n, dtype=bool)

    n_flagged = 0
    for i, nuc in enumerate(nuclei):
        if key_bbox not in nuc:
            keep[i] = False
            n_flagged += 1
            continue

        bbox = nuc[key_bbox]
        y1, x1 = bbox[0]
        y2, x2 = bbox[1]
        w = int(x2 - x1)
        h = int(y2 - y1)

        if w <= 0 or h <= 0:
            keep[i] = False
            n_flagged += 1
            continue

        try:
            patch = np.array(
                slide_reader.read_region((int(x1), int(y1)), 0, (int(w), int(h)))
            )[..., :3]
            od = rgb_to_od(patch, I0=I0)
            od_total = np.linalg.norm(od.reshape(-1, 3), axis=1)
            tissue_frac = float((od_total > 0.05).mean())
            if tissue_frac < min_tissue_fraction:
                keep[i] = False
                n_flagged += 1
        except Exception:
            keep[i] = False
            n_flagged += 1

    if verbose:
        print(
            f"[TissueFilter] Flagged {n_flagged}/{n} nuclei "
            f"({n_flagged/n*100:.1f}%) with <{min_tissue_fraction*100:.0f}% tissue"
        )

    return keep


# =============================================================================
# Stain matrix estimation
# =============================================================================


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def _match_vectors_to_stains(
    vectors: np.ndarray,
    references: dict[str, np.ndarray],
) -> dict[str, int]:
    """
    Match estimated NMF components to known stain types using cosine similarity.

    Parameters
    ----------
    vectors : array of shape (n_components, 3)
        Each row is an estimated stain vector.
    references : dict
        stain_name -> reference OD vector

    Returns
    -------
    mapping : dict of stain_name
    """
    n = vectors.shape[0]
    stain_names = list(references.keys())

    # Build similarity matrix
    sim_matrix = np.zeros((len(stain_names), n))
    for i, name in enumerate(stain_names):
        for j in range(n):
            sim_matrix[i, j] = _cosine_similarity(references[name], vectors[j])

    mapping = {}
    used_components = set()
    used_stains = set()

    for _ in range(min(len(stain_names), n)):
        best_val = -1
        best_stain = None
        best_comp = None
        for i, name in enumerate(stain_names):
            if name in used_stains:
                continue
            for j in range(n):
                if j in used_components:
                    continue
                if sim_matrix[i, j] > best_val:
                    best_val = sim_matrix[i, j]
                    best_stain = name
                    best_comp = j
        if best_stain is not None:
            mapping[best_stain] = best_comp
            used_components.add(best_comp)
            used_stains.add(best_stain)

    return mapping


def _angular_extreme_pixels(
    od: np.ndarray,
    reference: np.ndarray,
    angle_tolerance_deg: float = 30.0,
    od_min: float = 0.1,
) -> np.ndarray:
    """
    Find pixels whose OD direction is close to a reference stain vector
    AND that have strong staining (high OD magnitude).

    Returns the indices of candidate pixels for that stain.
    """
    od_mag = np.linalg.norm(od, axis=1)
    # Normalize each pixel to unit direction
    od_unit = od / (od_mag[:, None] + 1e-10)
    ref_unit = reference / (np.linalg.norm(reference) + 1e-10)

    # Cosine similarity to reference
    cos_sim = od_unit @ ref_unit
    angle_rad = np.arccos(np.clip(cos_sim, -1, 1))
    angle_deg = np.degrees(angle_rad)

    # Select pixels that are (a) close in angle AND (b) have decent OD
    mask = (angle_deg < angle_tolerance_deg) & (od_mag > od_min)

    if mask.sum() < 5:
        # Relax angle tolerance
        mask = (angle_deg < angle_tolerance_deg * 2) & (od_mag > od_min * 0.5)

    return np.where(mask)[0]


def estimate_stain_matrix(
    pixels_rgb: np.ndarray,
    I0: float | np.ndarray = 240.0,
    od_threshold: float = 0.05,
    dab_od_threshold: float = 0.15,
    nmf_max_iter: int = 1000,
    refinement_strength: float = 0.5,
    rng_seed: int = 42,
    verbose: bool = True,
):
    """
    Stain matrix estimation for H&E+DAB on restained slides.
    Containing 3 strategies with different weighing.

    Strategy 1: Reference-initialized NMF
        - Uses known H/E/DAB vectors as initialization so NMF converges
          to the correct solution even when one stain is very weak.

    Strategy 2: Angular extrema refinement
        - For each stain, find pixels whose OD direction is closest to
          the expected direction AND have high staining intensity.
        - The mean OD direction of these extreme pixels refines each vector.

    Strategy 3: Targeted DAB estimation from brown pixels
        - Specifically selects pixels with high B-channel OD, independent of NMF.
        - Cross-validates against the NMF-estimated DAB vector.

    Parameters
    ----------
    pixels_rgb : array (N, 3), uint8 (output of extract function)
    I0 : float or (3,)
    od_threshold : float
        Min OD magnitude for tissue pixels.
    dab_od_threshold : float
        Min OD magnitude for DAB candidate pixels (higher than general
        tissue threshold since we want well-stained pixels).
    nmf_max_iter : int
    refinement_strength : float (from config)
        How much to blend NMF result with angular-refind result.
        0 = pure NMF, 1 = pure angular.
    rng_seed : int
    verbose : bool
    """
    # --- Convert to OD and filter ---
    od = rgb_to_od(pixels_rgb, I0=I0)
    od_magnitude = np.linalg.norm(od, axis=1)

    tissue_mask = od_magnitude > od_threshold
    od_tissue = od[tissue_mask]
    od_tissue = np.clip(od_tissue, 0, None)

    if verbose:
        print(f"[StainMatrix] Total pixels: {len(od):,}")
        print(f"[StainMatrix] Tissue pixels (OD > {od_threshold}): {len(od_tissue):,}")

    if len(od_tissue) < 100:
        raise ValueError(f"Only {len(od_tissue)} tissue pixels. Check extraction.")

    # Subsample for NMF speed
    max_pixels = 1_000_000
    rng = np.random.default_rng(rng_seed)
    if len(od_tissue) > max_pixels:
        idx = rng.choice(len(od_tissue), max_pixels, replace=False)
        od_for_nmf = od_tissue[idx]
    else:
        od_for_nmf = od_tissue

    # =========================================================================
    # Strategy 1: Refreence-initialized NMF
    # =========================================================================
    if verbose:
        print("[StainMatrix] Strategy 1: Reference-initialized NMF(3)...")

    # Build initial H_init (components matrix, shape 3x3: components x features)
    ref_h = normalize_stain_vector(REFERENCE_VECTORS["hematoxylin"])
    ref_e = normalize_stain_vector(REFERENCE_VECTORS["eosin"])
    ref_d = normalize_stain_vector(REFERENCE_VECTORS["dab"])

    H_init = np.stack([ref_h, ref_e, ref_d], axis=0)  # (3, 3)

    # Scale H_init so magnitudes roughly match the data
    od_scale = np.percentile(np.linalg.norm(od_for_nmf, axis=1), 90)
    H_init = H_init * od_scale

    # Initial W via non-negative projection onto reference basis
    H_init_pinv = np.linalg.pinv(H_init.T)  # (3, 3)
    W_init = np.clip(od_for_nmf @ H_init_pinv.T, 0, None)  # (N, 3)

    # Ensure no zeros (NMF requires strictly positive init for custom)
    H_init = np.clip(H_init, 1e-6, None)
    W_init = np.clip(W_init, 1e-6, None)

    nmf = NMF(
        n_components=3,
        init="custom",
        max_iter=nmf_max_iter,
        random_state=rng_seed,
        beta_loss="frobenius",
        solver="mu",  # multiplicative update, works with custom init
        l1_ratio=0.05,
    )

    try:
        W_nmf = nmf.fit_transform(od_for_nmf, W=W_init, H=H_init)
        H_nmf = nmf.components_  # (3, 3)
        nmf_converged = nmf.n_iter_ < nmf_max_iter
    except Exception as e:
        warnings.warn(f"NMF failed: {e}. Using reference vectors.", RuntimeWarning)
        H_nmf = H_init.copy()
        nmf_converged = False

    # Normalize NMF components
    nmf_vectors = np.array([normalize_stain_vector(H_nmf[i]) for i in range(3)])

    # Match NMF components to stains
    mapping = _match_vectors_to_stains(nmf_vectors, REFERENCE_VECTORS)
    h_nmf = nmf_vectors[mapping["hematoxylin"]]
    e_nmf = nmf_vectors[mapping["eosin"]]
    d_nmf = nmf_vectors[mapping["dab"]]

    if verbose:
        print(f"  NMF converged: {nmf_converged} ({nmf.n_iter_} iters)")
        for name, vec in [("H", h_nmf), ("E", e_nmf), ("DAB", d_nmf)]:
            ref = REFERENCE_VECTORS[
                {"H": "hematoxylin", "E": "eosin", "DAB": "dab"}[name]
            ]
            sim = _cosine_similarity(vec, ref)
            print(f"  {name}_nmf: {vec} (cos sim: {sim:.4f})")

    # =========================================================================
    # Strategy 2: Angular extrema refinement
    # =========================================================================
    if verbose:
        print("[StainMatrix] Strategy 2: Angular extrema refinement...")

    # For each stain, find pixels that are angularly close to the NMF-estimated
    # direction AND have high staining. Use these "pure" pixels to refine.
    refined = {}
    for name, nmf_vec, ref_vec in [
        ("hematoxylin", h_nmf, ref_h),
        ("eosin", e_nmf, ref_e),
        ("dab", d_nmf, ref_d),
    ]:
        # Use the NMF vector as the search direction (it's closer to truth
        # than the reference for H and E; for DAB use a wider search)
        if name == "dab":
            search_vec = ref_d  # DAB might be poorly estimated by NMF
            angle_tol = 35.0
            min_od = dab_od_threshold
        else:
            search_vec = nmf_vec
            angle_tol = 25.0
            min_od = 0.1

        candidates_idx = _angular_extreme_pixels(
            od_tissue,
            search_vec,
            angle_tolerance_deg=angle_tol,
            od_min=min_od,
        )

        if len(candidates_idx) >= 10:
            # Take the top staining-intensity pixels within the angular cone
            candidate_od = od_tissue[candidates_idx]
            candidate_mag = np.linalg.norm(candidate_od, axis=1)
            top_mask = candidate_mag > np.percentile(candidate_mag, 80)
            top_od = candidate_od[top_mask]

            # Mean direction = refined stain vector
            mean_dir = top_od.mean(axis=0)
            refined[name] = normalize_stain_vector(np.clip(mean_dir, 0, None))

            if verbose:
                sim = _cosine_similarity(refined[name], ref_vec)
                print(
                    f"  {name}: {len(candidates_idx)} candidates, "
                    f"refined vec: {refined[name]} (cos sim to ref: {sim:.4f})"
                )
        else:
            refined[name] = nmf_vec
            if verbose:
                print(
                    f"  {name}: too few candidates ({len(candidates_idx)}), "
                    f"keeping NMF result"
                )

    # =========================================================================
    # Strategy 3: Targeted DAB from spectral signature
    # =========================================================================
    if verbose:
        print("[StainMatrix] Strategy 3: Targeted DAB from spectral signature...")

    # DAB absorbs most in blue channel (OD index 2) relative to red (OD index 0)
    # In OD space: DAB has high OD_blue / OD_red ratio
    # H has relatively balanced OD across R and G with low B
    # E has high G, low R and B
    od_strong = od_tissue[np.linalg.norm(od_tissue, axis=1) > dab_od_threshold]

    if len(od_strong) > 0:
        # Ratio: blue_od / (red_od + green_od + eps)
        blue_ratio = od_strong[:, 2] / (od_strong[:, 0] + od_strong[:, 1] + 1e-6)

        # DAB pixels should have high blue ratio (brown in RGB = high blue OD)
        # Also check that it's not just noise by requiring decent total OD
        high_blue = blue_ratio > np.percentile(blue_ratio, 95)
        dab_spectral_candidates = od_strong[high_blue]

        if len(dab_spectral_candidates) >= 5:
            # Weight by OD magnitude (prefer strongly stained pixels)
            weights = np.linalg.norm(dab_spectral_candidates, axis=1)
            dab_spectral_vec = normalize_stain_vector(
                np.clip(
                    np.average(dab_spectral_candidates, axis=0, weights=weights),
                    0,
                    None,
                )
            )
            sim_spectral = _cosine_similarity(dab_spectral_vec, ref_d)

            if verbose:
                print(
                    f"  Spectral DAB: {dab_spectral_vec} "
                    f"(cos sim to ref: {sim_spectral:.4f}), "
                    f"from {len(dab_spectral_candidates)} pixels"
                )

            # Use spectral estimate if it's better than angular refined
            sim_angular_dab = _cosine_similarity(refined["dab"], ref_d)
            if sim_spectral > sim_angular_dab:
                if verbose:
                    print(
                        f"  -> Spectral DAB is better ({sim_spectral:.4f} vs "
                        f"{sim_angular_dab:.4f}), using spectral"
                    )
                refined["dab"] = dab_spectral_vec
        else:
            if verbose:
                print(
                    f"  Too few spectral DAB candidates ({len(dab_spectral_candidates)})"
                )
    else:
        if verbose:
            print(f"  No strong-OD pixels found for spectral DAB analysis")

    # Blend NMF and refined estimates
    alpha = refinement_strength

    h_final = normalize_stain_vector(
        np.clip((1 - alpha) * h_nmf + alpha * refined["hematoxylin"], 0, None)
    )
    e_final = normalize_stain_vector(
        np.clip((1 - alpha) * e_nmf + alpha * refined["eosin"], 0, None)
    )
    d_final = normalize_stain_vector(
        np.clip((1 - alpha) * d_nmf + alpha * refined["dab"], 0, None)
    )

    # Check that vectors are sufficiently different from each other
    sim_he = _cosine_similarity(h_final, e_final)
    sim_hd = _cosine_similarity(h_final, d_final)
    sim_ed = _cosine_similarity(e_final, d_final)

    if verbose:
        print(
            f"\n[StainMatrix] Inter-stain similarities: "
            f"H·E={sim_he:.3f}, H·DAB={sim_hd:.3f}, E·DAB={sim_ed:.3f}"
        )

    # If any pair is too similar, fall back to reference for the weaker stain
    for pair_sim, name_a, name_b, vec_a_ref, vec_b_ref in [
        (sim_he, "H", "E", ref_h, ref_e),
        (sim_hd, "H", "DAB", ref_h, ref_d),
        (sim_ed, "E", "DAB", ref_e, ref_d),
    ]:
        if pair_sim > 0.95:
            warnings.warn(
                f"{name_a} and {name_b} vectors are nearly identical "
                f"(cos={pair_sim:.3f}). Falling back to references for both.",
                RuntimeWarning,
            )
            if name_a == "H":
                h_final = ref_h
            elif name_a == "E":
                e_final = ref_e
            if name_b == "E":
                e_final = ref_e
            elif name_b == "DAB":
                d_final = ref_d

    ### --- final matrix --- ###
    stain_matrix = np.stack([h_final, e_final, d_final], axis=1)  # (3, 3)
    channel_map = {"hematoxylin": 0, "eosin": 1, "dab": 2}

    if verbose:
        print(f"\n[StainMatrix] Final stain matrix (columns = H, E, DAB):")
        print(stain_matrix)
        for name, vec in [("H", h_final), ("E", e_final), ("DAB", d_final)]:
            ref = REFERENCE_VECTORS[
                {"H": "hematoxylin", "E": "eosin", "DAB": "dab"}[name]
            ]
            sim = _cosine_similarity(vec, ref)
            print(f"  {name} final cos sim to reference: {sim:.4f}")
        cond = np.linalg.cond(stain_matrix)
        print(f"[StainMatrix] Condition number: {cond:.2f}")
        if cond > 20:
            warnings.warn(
                "High condition number — deconvolution may be unstable.",
                RuntimeWarning,
            )

    return stain_matrix, channel_map


# =============================================================================
# Deconvolution
# =============================================================================
def deconvolve_pixels(
    pixels_rgb: np.ndarray,
    stain_matrix: np.ndarray,
    I0: float | np.ndarray = 240.0,
    method: str = "lstsq",
) -> np.ndarray:
    """
    Perform color deconvolution on RGB pixels.

    Parameters
    ----------
    pixels_rgb : array (N, 3), uint8 RGB
    stain_matrix : array (3, 3)
        Columns are stain vectors (normalized OD vectors).
    I0 : float or array
    method : str
        "lstsq" — fast least-squares (can produce small negative values)
        "nnls"  — non-negative least squares (slower, always >= 0)

    Returns
    -------
    concentrations : array (N, 3)
        Stain concentrations per pixel. Column order matches stain_matrix.
    """
    original_shape = pixels_rgb.shape[:-1]
    pixels_flat = pixels_rgb.reshape(-1, 3)

    od = rgb_to_od(pixels_flat, I0=I0)
    od = np.clip(od, 0, None)

    if method == "lstsq":
        # Fast: c = M_inv @ od^T
        M_inv = np.linalg.pinv(stain_matrix)  # (3, 3)
        concentrations = (M_inv @ od.T).T  # (N, 3)
        concentrations = np.clip(concentrations, 0, None)

    elif method == "nnls":
        # Slower but strictly non-negative
        concentrations = np.zeros((len(od), 3))
        for i in range(len(od)):
            concentrations[i], _ = nnls(stain_matrix, od[i])
    else:
        raise ValueError(f"Unknown method: {method}")

    return concentrations.reshape(*original_shape, 3)


# =============================================================================
# Per-nucleus DAB quantification
# =============================================================================
def quantify_nuclei_dab(
    slide_reader,
    nuclei: list[dict],
    stain_matrix: np.ndarray,
    channel_map: dict[str, int],
    I0: float | np.ndarray = 240.0,
    margin: int = 2,
    key_bbox: str = "bbox_pX",
    method: str = "lstsq",
    min_tissue_fraction: float = 0.2,
    use_nucleus_mask: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Quantify DAB staining intensity for each nucleus.

    Parameters
    ----------
    slide_reader : openslide.OpenSlide
    nuclei : list of dict
    stain_matrix : (3, 3) array
    channel_map : dict with "dab" key
    I0 : float or (3,) array
    margin : int — extra pixels around bbox (only used when no bitmap)
    key_bbox : str
    method : str, "lstsq" or "nnls"
    min_tissue_fraction : float — only used when no bitmap (old version)
    use_nucleus_mask : bool — if True, use bbox_bitmap when available (new version)
    verbose : bool

    Returns
    -------
    results : list of dict with different dab values.
    """
    dab_idx = channel_map["dab"]
    h_idx = channel_map["hematoxylin"]

    results = []
    n_failed = 0
    n_masked = 0

    for i, nuc in enumerate(nuclei):
        bbox = nuc[key_bbox]
        y1, x1 = bbox[0]
        y2, x2 = bbox[1]

        # !! Hardcoded, as always same output name from CellViT++
        has_bitmap = use_nucleus_mask and "bbox_bitmap" in nuc

        # No margin when using bitmap — it defines the exact boundary
        m = 0 if has_bitmap else margin

        x1_m = max(0, x1 - m)
        y1_m = max(0, y1 - m)
        w = (x2 + m) - x1_m
        h = (y2 + m) - y1_m

        w = int(round(w))
        h = int(round(h))
        if w <= 0 or h <= 0:
            results.append(_empty_result())
            n_failed += 1
            continue

        try:
            patch = np.array(
                slide_reader.read_region((int(x1_m), int(y1_m)), 0, (int(w), int(h)))
            )[..., :3]
        except Exception:
            results.append(_empty_result())
            n_failed += 1
            continue

        # Deconvolve
        conc = deconvolve_pixels(patch, stain_matrix, I0=I0, method=method)

        dab_values = conc[..., dab_idx].ravel()
        h_values = conc[..., h_idx].ravel()

        if has_bitmap:
            # Use the segmentation mask
            ## !! Hardcoded
            mask = np.array(nuc["bbox_bitmap"], dtype=bool)

            # Resize if patch dimensions don't match bitmap
            # (can differ by a few pixels due to float bbox_pX coordinates or slight misalignment)
            if mask.shape != patch.shape[:2]:
                from PIL import Image as PILImage

                mask = np.array(
                    PILImage.fromarray(mask.astype(np.uint8)).resize(
                        (patch.shape[1], patch.shape[0]),  # (width, height)
                        PILImage.NEAREST,
                    )
                ).astype(bool)

            if mask.sum() < 3:
                results.append(_empty_result())
                n_failed += 1
                continue

            # QC Check that the pX patch has actual tissue under the mask - set very low to keep fatty tissue regions.
            od_total = rgb_to_od(patch, I0=I0).reshape(-1, 3).sum(axis=1)
            tissue_in_mask = od_total[mask.ravel()] > 0.05
            if tissue_in_mask.mean() < min_tissue_fraction:
                results.append(_empty_result())
                n_failed += 1
                continue

            dab_tissue = dab_values[mask.ravel()]
            h_tissue = h_values[mask.ravel()]
            n_masked += 1

        else:
            # Old version: OD-based tissue mask
            od_total = rgb_to_od(patch, I0=I0).reshape(-1, 3).sum(axis=1)
            tissue_mask = od_total > 0.05
            tissue_fraction = tissue_mask.sum() / max(len(od_total), 1)
            if tissue_mask.sum() < 3 or tissue_fraction < min_tissue_fraction:
                results.append(_empty_result())
                n_failed += 1
                continue

            dab_tissue = dab_values[tissue_mask.ravel()]
            h_tissue = h_values[tissue_mask.ravel()]

        results.append(
            {
                "dab_mean": float(np.mean(dab_tissue)),
                "dab_median": float(np.median(dab_tissue)),
                "dab_max": float(np.max(dab_tissue)),
                "dab_p90": float(np.percentile(dab_tissue, 90)),
                "dab_positive_fraction": float(np.mean(dab_tissue > 0.1)),
                "h_mean": float(np.mean(h_tissue)),
            }
        )

    if verbose:
        if n_failed > 0:
            print(
                f"[DAB Quantification] {n_failed}/{len(nuclei)} nuclei "
                f"failed extraction."
            )
        if use_nucleus_mask:
            print(
                f"[DAB Quantification] Used nucleus mask for {n_masked}/{len(nuclei)} nuclei."
            )

    return results


def _empty_result() -> dict:
    return {
        "dab_mean": np.nan,
        "dab_median": np.nan,
        "dab_max": np.nan,
        "dab_p90": np.nan,
        "dab_positive_fraction": np.nan,
        "h_mean": np.nan,
    }


# =============================================================================
# Thresholding / labeling
# =============================================================================


def assign_senescence_labels(
    dab_results: list[dict],
    method: str = "negative_fit",
    metric: str = "dab_mean",
    sigma: float = 5.0,  # used for negative fit
    expected_positive_rate: float = 0.01,  # only for percentile method
    manual_threshold: Optional[
        float
    ] = None,  # default for me - chosen by manual inspection.
    verbose: bool = True,
):
    """
    Assign senescent/non-senescent labels based on DAB intensity.

    Parameters
    ----------
    dab_results : list of dict from quantify_nuclei_dab
    method : str
        "negative_fit" — (RECOMMENDED) Fits the negative population as a
            half-normal distribution, then thresholds at 'sigma' standard
            deviations above the mode.
        "manual"
        (and more - not removed if wanting to use in future - methods didnt work for me.)
    metric : str
        Which DAB metric to threshold on.
        - "dab_mean": mean DAB concentration across the nucleus. Good default.
        - "dab_p90": 90th percentile. More sensitive to focal staining
          (e.g., nucleus with partial DAB).
        - "dab_max": peak DAB pixel. Most sensitive but most noisy.
    sigma : float
        For "negative_fit": number of standard deviations above the
        negative population center. Higher = stricter = fewer false positives.
        Default 5.0 yielded good results for me.
    expected_positive_rate : float
        For "percentile": expected fraction of positive nuclei (e.g. 0.01
        for 1%, 0.05 for 5%). Threshold is placed at the (1 - expected_positive_rate) percentile.
    manual_threshold : float or None
        Required if method="manual".
    verbose : bool

    Returns
    -------
    labels : array of shape (n_nuclei,), int
        0 = non-senescent, 1 = senescent, -1 = invalid/failed
    threshold : float
        The threshold used - important output when not using manual.
    """
    values = np.array([r[metric] for r in dab_results])
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    if len(valid_values) < 30:
        raise ValueError(
            "Too few valid nuclei for thresholding - please check pipeline."
        )

    if method == "manual":
        if manual_threshold is None:
            raise ValueError("manual_threshold required for method='manual'")
        threshold = manual_threshold

    elif method == "negative_fit":  # best fallback if not manual
        threshold = _negative_population_threshold(
            valid_values,
            sigma=sigma,
            verbose=verbose,
        )

    elif method == "mean_std":  # not used
        global_mean = np.mean(valid_values)
        global_std = np.std(valid_values)
        threshold = global_mean + sigma * global_std
        if verbose:
            print(
                f"[Labeling] Mean+Std method: μ={global_mean:.4f}, "
                f"sigma_data={global_std:.4f}, threshold={threshold:.4f}"
            )

    elif method == "percentile":  # not used
        pct = (1.0 - expected_positive_rate) * 100.0
        threshold = float(np.percentile(valid_values, pct))
        if verbose:
            print(f"[Labeling] Percentile method: p{pct:.1f} = {threshold:.4f}")

    elif method == "gmm":  # not used
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=2,
            random_state=42,
            max_iter=300,
        )
        gmm.fit(valid_values.reshape(-1, 1))

        # The component with the lower mean is the negative population
        means = gmm.means_.ravel()
        stds = np.sqrt(gmm.covariances_.ravel())
        neg_idx = np.argmin(means)
        pos_idx = 1 - neg_idx

        # Threshold at the crossover point, or at neg_mean + sigma * neg_std
        neg_mean = means[neg_idx]
        neg_std = stds[neg_idx]
        threshold = neg_mean + sigma * neg_std

        if verbose:
            weights = gmm.weights_
            print(
                f"[Labeling/GMM] Negative pop: μ={neg_mean:.4f}, "
                f"sigma={neg_std:.4f}, weight={weights[neg_idx]:.3f}"
            )
            print(
                f"[Labeling/GMM] Positive pop: μ={means[pos_idx]:.4f}, "
                f"sigma={stds[pos_idx]:.4f}, weight={weights[pos_idx]:.3f}"
            )
            print(
                f"[Labeling/GMM] Threshold = {neg_mean:.4f} + "
                f"{sigma} x {neg_std:.4f} = {threshold:.4f}"
            )

    elif method == "mode_mirror":  # not used
        threshold = _mode_mirror_threshold(
            valid_values,
            sigma=sigma,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    labels = np.full(len(values), -1, dtype=int)
    labels[valid_mask & (values <= threshold)] = 0
    labels[valid_mask & (values > threshold)] = 1

    if verbose:
        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        n_inv = (labels == -1).sum()
        pct_pos = n_pos / max(n_pos + n_neg, 1) * 100
        print(f"[Labeling] Method: {method}, threshold: {threshold:.4f}")
        print(
            f"[Labeling] Senescent: {n_pos}, Non-senescent: {n_neg}, "
            f"Invalid: {n_inv} ({pct_pos:.1f}% positive)"
        )

        # Sanity warnings
        if pct_pos > 20:
            warnings.warn(
                f"Positive rate is {pct_pos:.1f}%, which is unusually high. "
                f"If you expect ~1%, try increasing sigma (currently {sigma}) "
                f"or check the DAB channel in QuPath.",
                RuntimeWarning,
            )
        if pct_pos < 0.1:
            warnings.warn(
                f"Positive rate is {pct_pos:.2f}%, which may be too strict. "
                f"Try decreasing sigma (currently {sigma}).",
                RuntimeWarning,
            )

    return labels, threshold


def _negative_population_threshold(
    values: np.ndarray,
    sigma: float = 5.0,
    verbose: bool = True,
):
    """
    Estimate the negative population distribution and threshold at sigma SDs.

    Method:
    1. Uses median as the center of the negative population.
    2. Estimate the spraed using only values BELOW the median (guaranteed negatives).
       The MAD (median absolute deviation) of the lower half gives a robust scale estimate.
    3. Threshold = median + sigma * estimated_std
    """
    median_val = np.median(values)

    # Use only values <= median
    lower_half = values[values <= median_val]

    # MAD of lower half, scaled to approximate std for a half-normal
    mad = np.median(np.abs(lower_half - median_val))
    estimated_std = mad / 0.6745 if mad > 1e-10 else np.std(lower_half)

    # Alternative: directly compute std of lower half, then correct for the fact that we're only seeing half the distribution
    std_lower = np.std(lower_half)
    estimated_std_alt = std_lower / 0.8  # correction for truncation

    # Use the more conservative (larger) estimate
    estimated_std = max(estimated_std, estimated_std_alt)

    threshold = median_val + sigma * estimated_std

    if verbose:
        print(
            f"[Labeling/NegFit] Negative population: "
            f"center={median_val:.4f}, std≈{estimated_std:.4f}"
        )
        print(
            f"[Labeling/NegFit] Threshold = {median_val:.4f} + "
            f"{sigma:.1f} x {estimated_std:.4f} = {threshold:.4f}"
        )

        # Show what different sigma values would give
        for s in [3, 4, 5, 6, 7]:
            t = median_val + s * estimated_std
            n_above = (values > t).sum()
            pct = n_above / len(values) * 100
            marker = " <<<" if abs(s - sigma) < 0.01 else ""
            print(
                f"  signma={s}: threshold={t:.4f}, "
                f"positive={n_above} ({pct:.2f}%){marker}"
            )

    return float(threshold)


def _mode_mirror_threshold(
    values: np.ndarray,
    sigma: float = 5.0,
    kde_bandwidth: str = "silverman",
    verbose: bool = True,
):
    """
    Estimate negative population threshold using mode-mirror method.

    Parameters
    ----------
    values : array - DAB metric values.
    sigma : float
        Number of std devs above the mode.
    verbose : bool

    Returns
    -------
    threshold : float
    """
    from scipy.stats import gaussian_kde

    # Step 1: Find the mode via KDE
    kde = gaussian_kde(values, bw_method=kde_bandwidth)
    # Evaluate on a fine grid
    x_grid = np.linspace(
        np.percentile(values, 0.5),
        np.percentile(values, 99.5),
        2000,
    )
    density = kde(x_grid)
    mode_val = x_grid[np.argmax(density)]

    # Step 2: Use only the LEFT flank (values <= mode) to estimate std
    left_flank = values[values <= mode_val]

    if len(left_flank) < 10:
        # Fallback: use lower quartile
        left_flank = values[values <= np.percentile(values, 25)]

    # The left flank is a half-distribution. Its std relative to the mode
    # gives us the negative population's true spread.
    # Method A: RMS deviation from mode (direct std estimate for half-normal)
    rms_left = np.sqrt(np.mean((left_flank - mode_val) ** 2))

    # Method B: MAD of left flank, scaled
    mad_left = np.median(np.abs(left_flank - mode_val))
    # For a half-normal, median(|X|) = 0.6745 * sigma
    std_from_mad = mad_left / 0.6745 if mad_left > 1e-10 else rms_left

    # Use the average of both estimates for robustness
    estimated_std = (rms_left + std_from_mad) / 2

    # Step 3: Threshold
    threshold = mode_val + sigma * estimated_std

    if verbose:
        print(f"[Labeling/ModeMirror] Mode: {mode_val:.4f}")
        print(f"[Labeling/ModeMirror] Left-flank pixels: {len(left_flank)}")
        print(
            f"[Labeling/ModeMirror] Estimated σ: {estimated_std:.4f} "
            f"(RMS={rms_left:.4f}, MAD-based={std_from_mad:.4f})"
        )
        print(
            f"[Labeling/ModeMirror] Threshold = {mode_val:.4f} + "
            f"{sigma} × {estimated_std:.4f} = {threshold:.4f}"
        )

        for s in [3, 4, 5, 6, 7]:
            t = mode_val + s * estimated_std
            n_above = (values > t).sum()
            pct = n_above / len(values) * 100
            marker = " <<<" if abs(s - sigma) < 0.01 else ""
            print(
                f"  sigma={s}: threshold={t:.4f}, "
                f"positive={n_above} ({pct:.2f}%){marker}"
            )

    return float(threshold)


# =============================================================================
# Visualization helpers - optional use in Notebooks
# =============================================================================


def visualize_deconvolution(
    patch_rgb: np.ndarray,
    stain_matrix: np.ndarray,
    I0: float = 240.0,
    channel_names: tuple[str, ...] = ("Hematoxylin", "Eosin", "DAB"),
):
    """
    Visualize original patch and individual deconvolved channels.
    """
    import matplotlib.pyplot as plt

    conc = deconvolve_pixels(patch_rgb, stain_matrix, I0=I0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(patch_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, name in enumerate(channel_names):
        channel = conc[..., i]
        axes[i + 1].imshow(channel, cmap="gray_r", vmin=0, vmax=np.percentile(conc, 99))
        axes[i + 1].set_title(name)
        axes[i + 1].axis("off")

    plt.tight_layout()
    return fig


def plot_dab_distribution(
    dab_results: list[dict],
    metric: str = "dab_mean",
    threshold: Optional[float] = None,
    sigma: float = 5.0,
    show_negative_fit: bool = True,
    log_x: bool = False,
    log_y: bool = False,
):
    """
    Plot DAB intensity distribution.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import numpy as np

    values = np.array([r[metric] for r in dab_results if not np.isnan(r[metric])])
    if log_x:
        # Remove non-positive values for log scale
        values = values[values > 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Left panel: full distribution ---
    ax = axes[0]
    n_bins = min(200, max(50, len(values) // 20))
    counts, bin_edges, patches = ax.hist(
        values,
        bins=n_bins,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.7,
        color="steelblue",
        density=True,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if show_negative_fit and len(values) > 30:
        median_val = np.median(values)
        lower_half = values[values <= median_val]
        mad = np.median(np.abs(lower_half - median_val))
        est_std = mad / 0.6745 if mad > 1e-10 else np.std(lower_half)
        std_lower = np.std(lower_half)
        est_std = max(est_std, std_lower / 0.8)

        x_fit = np.linspace(values.min(), values.max(), 500)
        y_fit = norm.pdf(x_fit, loc=median_val, scale=est_std)
        ax.plot(
            x_fit,
            y_fit,
            color="green",
            linewidth=2,
            linestyle="-",
            label=f"Negative pop. fit (median={median_val:.4f}, signma={est_std:.4f})",
        )

        for s in [3, 5, 7]:
            t = median_val + s * est_std
            style = "--" if s == sigma else ":"
            lw = 2 if s == sigma else 1
            ax.axvline(
                t,
                color="orange",
                linestyle=style,
                linewidth=lw,
                alpha=0.7,
                label=f"signa={s}: {t:.4f}",
            )

    if threshold is not None:
        ax.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Threshold: {threshold:.4f}",
        )

    ax.set_xlabel(f"DAB intensity ({metric})")
    ax.set_ylabel("Density")
    ax.set_title("DAB Intensity Distribution (all nuclei)")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)

    # --- Right panel: zoomed into the tail ---
    ax2 = axes[1]
    if threshold is not None:
        zoom_min = (
            max(1e-3, threshold - (threshold * 2))
            if log_x
            else threshold - (threshold * 2)
        )
    else:
        zoom_min = (
            max(1e-3, np.percentile(values, 90)) if log_x else np.percentile(values, 90)
        )
    zoom_max = np.percentile(values, 99.9)

    tail_values = values[(values >= zoom_min) & (values <= zoom_max)]
    if len(tail_values) > 10:
        ax2.hist(
            tail_values,
            bins=80,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.7,
            color="steelblue",
        )
        if threshold is not None:
            ax2.axvline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=2.5,
                label=f"Threshold: {threshold:.4f}",
            )
        n_above = (values > threshold).sum() if threshold else 0
        ax2.set_title(f"Tail zoom ({n_above} nuclei above threshold)")
    else:
        ax2.text(
            0.5,
            0.5,
            "No tail to show",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Tail zoom")

    ax2.set_xlabel(f"DAB intensity ({metric})")
    ax2.set_ylabel("Count")
    if log_x:
        ax2.set_xscale("log")
    if log_y:
        ax2.set_yscale("log")
    if threshold is not None:
        ax2.legend()

    plt.tight_layout()
    return fig


def export_labels_to_qupath_geojson(
    nuclei: list[dict],
    labels: np.ndarray,
    dab_results: list[dict],
    output_path: str,
    key_centroid: str = "centroid_pX",
    metric: str = "dab_mean",
    export_negatives: bool = True,
) -> None:
    """
    Export labeled nuclei as QuPath-compatible GeoJSON annotations.

    Parameters
    ----------
    nuclei : list of dict
    labels : array from assign_senescence_labels (0/1/-1)
    dab_results : list of dict from quantify_nuclei_dab
    output_path : str
        Path to write the .geojson file.
    key_centroid : str
        set to centroid or centroid_pX depending on which slide is overlayed with annotations.
    metric : str
        DAB metric to include in measurements.
    export_negatives : bool
        If True, also exports label=0 nuclei.
    """
    import json

    def rgb_to_qupath_color(r: int, g: int, b: int) -> int:
        """Convert RGB to QuPath's signed 32-bit ARGB integer."""
        argb = (255 << 24) | (r << 16) | (g << 8) | b
        if argb >= 0x80000000:
            argb -= 0x100000000
        return argb

    features = []
    for i, (nuc, label, dab) in enumerate(zip(nuclei, labels, dab_results)):
        if label == -1:  # remove invalid
            continue
        if label == 0 and not export_negatives:
            continue

        cx, cy = nuc[key_centroid]  # [x, y]
        cx, cy = float(cx), float(cy)

        if label == 1:
            class_name = "Senescent"
            color_rgb = rgb_to_qupath_color(200, 50, 50)
        else:
            class_name = "Non-senescent"
            color_rgb = rgb_to_qupath_color(50, 50, 200)

        measurements = []
        for key in [metric, "dab_p90", "dab_max", "h_mean"]:
            val = dab.get(key, float("nan"))
            if not np.isnan(val):
                measurements.append({"name": key, "value": float(val)})

        # QuPath geojson format.
        feature = {
            "type": "Feature",
            "id": "PathAnnotationObject",
            "geometry": {
                "type": "Point",
                "coordinates": [cx, cy],
            },
            "properties": {
                "name": f"nucleus_{i}",
                "classification": {
                    "name": class_name,
                    "colorRGB": color_rgb,
                },
                "isLocked": False,
                "measurements": measurements,
            },
        }
        features.append(feature)

    with open(output_path, "w") as f:
        json.dump(features, f)

    print(f"[QuPath Export] Wrote {len(features)} annotations to {output_path}")


# =============================================================================
# Main pipeline function
# =============================================================================


def run_deconvolution_pipeline(
    slide_path,
    samples: SampleManager,
    key_bbox: str = "bbox_pX",
    margin_matrix: int = 5,
    margin_quantify: int = 2,
    max_nuclei_for_matrix: int = 200_000,
    I0: float | np.ndarray = 240.0,
    special_I0_dict: dict = {},
    dab_od_threshold: float = 0.15,
    refinement_strength: float = 0.5,
    labeling_method: str = "negative_fit",
    labeling_metric: str = "dab_mean",
    manual_threshold: Optional[float] = None,
    labeling_sigma: float = 5.0,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: estimate stain matrix → deconvolve → quantify → label.

    Parameters
    ----------
    slide_path: str or Path() - pX slide
    samples : SampleManager object of HE and pX coordinates.
    key_bbox : str
        Key in nuclei dict - coordinates of pX slide.
    margin_matrix : int
        Margin for pixel extraction during matrix estimation.
    margin_quantify : int
        Margin for per-nucleus quantification.
    max_nuclei_for_matrix : int
        Max nuclei to sample for stain matrix estimation.
    I0 : float or (3,) array - None for estimate (default in my pipeline).
    dab_od_threshold : float
        Min OD magnitude for DAB candidate pixels during estimation.
    refinement_strength : float
        Blend between NMF (0) and angular refinement (1).
    labeling_method : str
    labeling_metric : str
    labeling_sigma : float
        For negative_fit: SDs above negative population center.
    verbose : bool

    Returns
    -------
    result : dict with keys:
        "stain_matrix": (3, 3) array
        "channel_map": dict
        "dab_results": list of per-nucleus dicts
        "labels": array of int (0/1/-1)
        "threshold": float
    """
    # Step 0: Load all files and put them into format
    nuclei = samples.sample_xs
    slide_reader = OpenSlide(slide_path)

    if verbose:
        print("=" * 60)
        print("Color Deconvolution Pipeline for H&E + DAB")
        print("=" * 60)

    # Step 1: Extract pixels from nuclei
    if verbose:
        print(f"\n[Step 1] Extracting pixels from {len(nuclei)} nuclei...")

    pixels = extract_nuclei_pixels_from_slide(
        slide_reader,
        nuclei,
        margin=margin_matrix,
        key_bbox=key_bbox,
        max_nuclei=max_nuclei_for_matrix,
    )
    if verbose:
        print(f"  Extracted {len(pixels):,} pixels")

    # Step 2: Estimate stain matrix
    if verbose:
        print(f"\n[Step 2] Estimating stain matrix...")

    if I0 is None:
        patch_size = 512  # !! hardcoded for thesis.
        if slide_path.name in special_I0_dict.keys():
            # Chooses which corner it takes background patch from - default top left.
            # some corners have brown residue, thats why other parts are chosen
            I0_start = special_I0_dict[slide_path.name]
            x_start = slide_reader.dimensions[0] - patch_size
            y_start = slide_reader.dimensions[1] - patch_size
            if I0_start == "TR":  # toprigght
                y_start = 0
            elif I0_start == "BL":  # bottom left
                x_start = 0
            start_coords = (x_start, y_start)
        else:
            start_coords = (0, 0)
        I0 = estimate_I0_from_background(
            slide_reader, [start_coords], patch_size=patch_size
        )

    if verbose:
        print(f"\nI0: {I0}")

    stain_matrix, channel_map = estimate_stain_matrix(
        pixels,
        I0=I0,
        dab_od_threshold=dab_od_threshold,
        refinement_strength=refinement_strength,
        verbose=verbose,
    )

    # Step 3: Quantify DAB per nucleus
    if verbose:
        print(f"\n[Step 3] Quantifying DAB per nucleus...")
    dab_results = quantify_nuclei_dab(
        slide_reader,
        nuclei,
        stain_matrix=stain_matrix,
        channel_map=channel_map,
        I0=I0,
        margin=margin_quantify,
        key_bbox=key_bbox,
        verbose=verbose,
    )

    # Step 4: Assign labels
    if verbose:
        print(f"\n[Step 4] Assigning senescence labels...")
    labels, threshold = assign_senescence_labels(
        dab_results,
        method=labeling_method,
        metric=labeling_metric,
        sigma=labeling_sigma,
        verbose=verbose,
        manual_threshold=manual_threshold,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)

    return {
        "stain_matrix": stain_matrix,
        "channel_map": channel_map,
        "dab_results": dab_results,
        "labels": labels,
        "threshold": threshold,
    }
