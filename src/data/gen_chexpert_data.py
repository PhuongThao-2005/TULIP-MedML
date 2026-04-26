"""
File: gen_chexpert_data.py
Description:
    One-time data preparation script for CheXpert auxiliary files.
    Generates label word embeddings and a co-occurrence adjacency matrix
    from the training CSV.
Main Components:
    - build_word_vectors: Create (14, 300) GloVe-based label embeddings.
    - gen_biomedclip_embeddings: Create (14, 512) BioMedCLIP text embeddings.
    - verify_embeddings: Print cosine similarity diagnostics for embeddings.
    - build_adj_matrix: Build (14, 14) label co-occurrence adjacency matrix.
Inputs:
    CheXpert train.csv and (optionally) a GloVe 300d text file.
Outputs:
    data/chexpert_glove_word2vec.npy  — word embeddings (14, 300)
    data/chexpert_biomedclip_vec.npy  — BioMedCLIP embeddings (14, 512)
    data/chexpert_adj.pkl             — co-occurrence adjacency matrix
Notes:
    - Run once before training: python gen_chexpert_data.py --csv <path>
    - GloVe is optional: random vectors (seed-reproducible) used as fallback.
    - BioMedCLIP requires open_clip and network access to download weights.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


CHEXPERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

NUM_CLASSES = len(CHEXPERT_CLASSES)

# Radiology-grounded descriptions for BioMedCLIP encoding.
# Each description summarizes the visual appearance and clinical relevance
# of the corresponding CheXpert class on a frontal chest X-ray.
CHEXPERT_DESCRIPTIONS = {
    "No Finding":
        "chest X-ray with no significant cardiopulmonary abnormality detected",

    "Enlarged Cardiomediastinum":
        "widening of the mediastinum or enlarged cardiac silhouette "
        "on frontal chest X-ray, may indicate vascular or lymph node pathology",

    "Cardiomegaly":
        "abnormal enlargement of the heart shadow on chest X-ray, "
        "cardiothoracic ratio greater than 0.5, suggesting cardiac disease",

    "Lung Opacity":
        "increased density in the lung parenchyma on chest X-ray, "
        "presenting as consolidation, ground-glass opacity, or airspace disease",

    "Lung Lesion":
        "focal abnormality in the lung including nodule, mass, or cavitary "
        "lesion visible on chest X-ray, may represent neoplasm or infection",

    "Edema":
        "pulmonary edema with bilateral airspace opacities, perihilar haziness, "
        "and vascular congestion on chest X-ray due to fluid in lung interstitium",

    "Consolidation":
        "homogeneous opacification of lung parenchyma replacing air with fluid "
        "or tissue on chest X-ray, consistent with pneumonia or hemorrhage",

    "Pneumonia":
        "infectious or inflammatory consolidation of lung parenchyma on chest "
        "X-ray, often presenting as lobar or segmental airspace opacity with fever",

    "Atelectasis":
        "partial or complete collapse of lung tissue on chest X-ray, "
        "causing volume loss and increased density due to airway obstruction",

    "Pneumothorax":
        "presence of air in the pleural space on chest X-ray, "
        "visible as pleural line with absence of lung markings beyond the margin",

    "Pleural Effusion":
        "abnormal accumulation of fluid in the pleural cavity on chest X-ray, "
        "appearing as blunting of costophrenic angle or meniscus sign",

    "Pleural Other":
        "pleural abnormality other than effusion on chest X-ray, "
        "including pleural thickening, calcification, or fibrothorax",

    "Fracture":
        "cortical disruption or break in bony structures visible on chest X-ray, "
        "commonly involving ribs or clavicle due to trauma",

    "Support Devices":
        "medical support devices visible on chest X-ray including endotracheal "
        "tube, central venous catheter, pacemaker lead, or nasogastric tube",
}


# Word Embeddings (GloVe)

def _load_glove(glove_path: str) -> dict:
    """
    Load a GloVe text file into a word → vector dictionary.

    Args:
        glove_path (str): Path to the GloVe 300d .txt file.

    Returns:
        dict: Maps lowercase word strings to np.ndarray of shape (300,).

    Notes:
        - Lines with an unexpected number of tokens are skipped silently.
        - Expects exactly 301 space-separated tokens per line (word + 300 floats).
    """
    print(f"Loading GloVe from {glove_path} ...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != 301:
                continue
            word = parts[0].lower()
            glove[word] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(glove)} tokens")
    return glove


def build_word_vectors(
    glove_path: str = None,
    out_path: str = "data/chexpert_glove_word2vec.npy",
) -> np.ndarray:
    """
    Build GloVe word embeddings for the 14 CheXpert class names.

    Each class name is split into words; per-word GloVe vectors are averaged.
    Falls back to reproducible random vectors when GloVe is unavailable.

    Args:
        glove_path (str, optional): Path to a GloVe 300d embedding file.
        out_path (str): Output path for the saved .npy file.

    Returns:
        np.ndarray: Label embedding matrix of shape (NUM_CLASSES, 300).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if glove_path and os.path.exists(glove_path):
        glove = _load_glove(glove_path)
        label_embeddings = np.zeros((NUM_CLASSES, 300), dtype=np.float32)

        for i, class_name in enumerate(CHEXPERT_CLASSES):
            words = class_name.lower().split()
            found_vectors = [glove[w] for w in words if w in glove]

            if found_vectors:
                # Average word vectors for multi-word class names
                label_embeddings[i] = np.mean(found_vectors, axis=0)
                print(
                    f"  [{i:2d}] {class_name:35s} "
                    f"→ {len(found_vectors)}/{len(words)} words found"
                )
            else:
                # Use a deterministic random vector when no GloVe coverage exists
                rng = np.random.RandomState(i)
                label_embeddings[i] = rng.randn(300).astype(np.float32)
                print(f"  [{i:2d}] {class_name:35s} → NOT FOUND, using random")
    else:
        print(
            "[INFO] GloVe not provided → using random vectors (reproducible seed).\n"
            "       For better results, download GloVe and pass --glove <path>."
        )
        rng = np.random.RandomState(42)
        label_embeddings = rng.randn(NUM_CLASSES, 300).astype(np.float32)

    np.save(out_path, label_embeddings)
    print(f"\nWord vectors saved: {out_path}  shape={label_embeddings.shape}")
    return label_embeddings


# BioMedCLIP Text Embeddings

def gen_biomedclip_embeddings(
    save_path: str = "data/chexpert_biomedclip_vec.npy",
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode the 14 CheXpert class descriptions using BioMedCLIP's text encoder.

    The encoder is frozen (eval mode, no gradients). Output embeddings are
    L2-normalized to unit norm.

    Args:
        save_path (str): Output path for the .npy embedding file.
        device (str): Torch device string ("cpu" or "cuda").

    Returns:
        np.ndarray: L2-normalized label embeddings of shape (NUM_CLASSES, 512).

    Notes:
        - Requires open_clip and access to Hugging Face Hub.
        - The entire model is loaded only for feature extraction; no training occurs.
    """
    from open_clip import create_model_from_pretrained, get_tokenizer

    print("[BioMedCLIP] Loading model...")
    model, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # Freeze all parameters — extraction only, no gradient computation needed
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    # Encode all 14 class descriptions in a single batch
    descriptions = [CHEXPERT_DESCRIPTIONS[cls] for cls in CHEXPERT_CLASSES]
    tokens = tokenizer(descriptions, context_length=256).to(device)

    with torch.no_grad():
        label_embeddings = model.encode_text(tokens)         # (14, 512)
        # L2-normalize so cosine similarity equals dot product
        label_embeddings = label_embeddings / label_embeddings.norm(
            dim=-1, keepdim=True
        )                                                    # (14, 512)

    embeddings_np = label_embeddings.cpu().numpy().astype(np.float32)
    np.save(save_path, embeddings_np)

    print(f"[BioMedCLIP] Saved {save_path}")
    print(f"  shape : {embeddings_np.shape}")
    print(f"  dtype : {embeddings_np.dtype}")
    print(f"  norm  : {np.linalg.norm(embeddings_np, axis=1).mean():.4f}")  

    return embeddings_np


def verify_embeddings(path: str = "data/chexpert_biomedclip_vec.npy") -> None:
    """
    Print cosine similarity diagnostics to sanity-check label embedding quality.

    Clinically related classes (e.g. Edema ↔ Consolidation) should have higher
    similarity than unrelated classes (e.g. Fracture ↔ No Finding).

    Args:
        path (str): Path to the .npy embedding file to verify.
    """
    label_embeddings = np.load(path)                        # (14, 512)
    similarity_matrix = cosine_similarity(label_embeddings) # (14, 14)

    print(
        f"\n[Verify] shape={label_embeddings.shape}, "
        f"norm≈{np.linalg.norm(label_embeddings, axis=1).mean():.3f}"
    )

    sim_off_diag = similarity_matrix.copy()
    np.fill_diagonal(sim_off_diag, 0)
    top_idx = np.argsort(sim_off_diag.ravel())[::-1][:3]
    rows, cols = np.unravel_index(top_idx, sim_off_diag.shape)

    print("\nTop 3 similar class pairs (clinically related):")
    for r, c in zip(rows, cols):
        print(
            f"  {CHEXPERT_CLASSES[r]:30s} ↔ "
            f"{CHEXPERT_CLASSES[c]:30s}: {sim_off_diag[r, c]:.4f}"
        )

    np.fill_diagonal(sim_off_diag, 1)
    bottom_idx = np.argsort(sim_off_diag.ravel())[:3]
    rows2, cols2 = np.unravel_index(bottom_idx, sim_off_diag.shape)

    print("\nTop 3 dissimilar class pairs (clinically unrelated):")
    for r, c in zip(rows2, cols2):
        print(
            f"  {CHEXPERT_CLASSES[r]:30s} ↔ "
            f"{CHEXPERT_CLASSES[c]:30s}: {sim_off_diag[r, c]:.4f}"
        )


# Co-occurrence Adjacency Matrix

def build_adj_matrix(
    csv_path: str,
    out_path: str = "data/chexpert_adj.pkl",
    uncertain: str = "zeros",
) -> dict:
    """
    Build a label co-occurrence adjacency matrix from CheXpert training annotations.

    A[i][j] counts the number of images where both class i and class j are
    positive. nums[j] counts the number of positive samples for class j.

    Args:
        csv_path (str): Path to the CheXpert train.csv file.
        out_path (str): Output path for the pickled result dictionary.
        uncertain (str): How to handle uncertain labels (-1).
            "zeros" → treat as negative.
            "ones"  → treat as positive.

    Returns:
        dict: {"adj": np.ndarray of shape (14, 14), "nums": np.ndarray of shape (14,)}

    Notes:
        - Uses vectorized matrix multiplication for efficiency on large CSVs.
        - The saved dict is consumed by gen_A() in src/util.py at model init.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    label_df = df[CHEXPERT_CLASSES].copy().fillna(0)

    # Remap uncertain labels (-1) according to the chosen policy
    if uncertain == "ones":
        label_df = label_df.replace(-1, 1)
    else:
        label_df = label_df.replace(-1, 0)

    labels = label_df.values.astype(np.float32)   # (N, 14)
    N = len(labels)

    # Binary indicator matrix for positive labels
    binary = (labels == 1).astype(np.float32)     # (N, 14)

    # Co-occurrence matrix: A[i, j] = number of images positive for both i and j
    adjacency_matrix = binary.T @ binary           # (14, 14)
    positive_counts = binary.sum(axis=0)           # (14,)

    result = {"adj": adjacency_matrix, "nums": positive_counts}
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"\nAdjacency matrix saved: {out_path}  shape={adjacency_matrix.shape}")
    print("  Positive label counts per class:")
    for cls, n in zip(CHEXPERT_CLASSES, positive_counts):
        pct = 100 * n / N
        print(f"    {cls:35s}: {int(n):6d} ({pct:.1f}%)")

    return result


# Entry Point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CheXpert auxiliary data (word vectors + adjacency matrix)."
    )
    parser.add_argument(
        "--csv",
        default="CheXpert-v1.0-small/train.csv",
        help="Path to CheXpert train.csv",
    )
    parser.add_argument(
        "--glove",
        default=None,
        help="Path to GloVe 300d text file (optional; falls back to random vectors)",
    )
    parser.add_argument(
        "--uncertain",
        default="zeros",
        choices=["zeros", "ones"],
        help="How to handle uncertain (-1) labels when building the adjacency matrix",
    )
    parser.add_argument(
        "--out_dir",
        default="data",
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Word Vectors (GloVe)")
    print("=" * 60)
    vec_path = os.path.join(args.out_dir, "chexpert_glove_word2vec.npy")
    build_word_vectors(glove_path=args.glove, out_path=vec_path)

    print("\n" + "=" * 60)
    print("Adjacency Matrix (Co-occurrence)")
    print("=" * 60)
    adj_path = os.path.join(args.out_dir, "chexpert_adj.pkl")
    build_adj_matrix(csv_path=args.csv, out_path=adj_path, uncertain=args.uncertain)

    print("\nDone. Files created:")
    print(f"   {vec_path}")
    print(f"   {adj_path}")