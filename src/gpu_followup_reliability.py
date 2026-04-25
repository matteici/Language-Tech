"""
09_gpu_followup_reliability.py
==============================
Section 9 of the steering-vector reliability study.
Implements five GPU-dependent tests that require re-running Mistral-7B:

  1. OOD generalization
  2. Political Compass Test external-anchor
  3. Causal activation-addition
  4. Paraphrase / length / pre-prompt confound check
  5. Magnitude calibration

Run directly (smoke-test mode):
    python src/09_gpu_followup_reliability.py --smoke-test

Full run (uses all layers and real batch sizes):
    python src/09_gpu_followup_reliability.py

Or import the individual test functions from a notebook.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
# If run from a worktree, fall back to the production data location.
_CANDIDATE_DATA = REPO_ROOT / "data" / "steering-vectors"
if not _CANDIDATE_DATA.exists():
    _PRODUCTION_REPO = Path(
        "/Users/matteici/Documents/GitHub/Language-Tech/political-debiasing-moe"
    )
    _CANDIDATE_DATA = _PRODUCTION_REPO / "data" / "steering-vectors"

DATA_ROOT = _CANDIDATE_DATA
VECTOR_DIR = DATA_ROOT / "vectors"
ACTIVATION_DIR = DATA_ROOT / "activations"
OUT_DIR = DATA_ROOT / "gpu_followup"
PLOT_DIR = OUT_DIR / "plots"
EXTERNAL_DIR = DATA_ROOT.parent / "external"

# ---------------------------------------------------------------------------
# Hardware / dtype config  (override via env vars or constructor kwargs)
# ---------------------------------------------------------------------------

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
_bf16_ok = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
DTYPE: torch.dtype = torch.bfloat16 if _bf16_ok else torch.float16

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LAYERS = [8, 12, 16, 20, 24]
HIDDEN_DIM = 4096
AXES = ["economic", "social"]

# Batch sizes — safe defaults for 96 GB VRAM.
# Activation extraction: each forward pass keeps hidden states for all requested
# layers in GPU memory briefly, then pools immediately.
ACTIVATION_BATCH_SIZE = 32
# Hook-based generation keeps one full KV-cache in VRAM; keep small.
GENERATION_BATCH_SIZE = 1

MAX_LENGTH = 256       # max tokens for activation extraction
MAX_NEW_TOKENS = 96    # tokens to generate in causal test

FORCE_RERUN = False    # set True to recompute even if outputs exist
USE_FLASH_ATTENTION_2 = True  # gracefully falls back if not installed

# Alphas for causal activation-addition sweep.
ALPHA_VALUES = [-3.0, -1.5, 0.0, 1.5, 3.0]

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None


def load_model(model_name: str = MODEL_NAME, force: bool = False):
    """Load Mistral-7B on GPU exactly once; return (model, tokenizer)."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None and not force:
        return _MODEL, _TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_str = str(DTYPE).replace("torch.", "")
    log.info("Loading model %s  device=%s  dtype=%s", model_name, DEVICE, dtype_str)

    kwargs: dict[str, Any] = dict(
        torch_dtype=DTYPE,
        device_map=DEVICE if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if USE_FLASH_ATTENTION_2 and DEVICE == "cuda":
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
            log.info("Flash Attention 2 enabled.")
        except ImportError:
            log.info("flash_attn not installed; using default attention.")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if DEVICE != "cuda" or "device_map" not in kwargs:
        model = model.to(DEVICE)

    model.eval()

    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s  |  VRAM: %.1f GB", gpu_name, vram_total)
        disk_free = shutil.disk_usage("/").free / 1e9
        if disk_free < 20:
            log.warning("Low disk space: %.1f GB free. Avoid saving large files.", disk_free)

    _MODEL = model
    _TOKENIZER = tokenizer
    return model, tokenizer


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def load_vectors(axis: str, method: str = "mean_difference") -> dict[int, np.ndarray]:
    """
    Load per-layer steering vectors for *axis* from the saved .pt artifact.

    Returns {layer_index: unit_vector_np (4096,)} using float32 on CPU.
    """
    path = VECTOR_DIR / f"{axis}_vectors.pt"
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    artifact = torch.load(path, map_location="cpu", weights_only=False)

    if "per_layer" not in artifact:
        raise KeyError(f"Expected 'per_layer' key in {path}. Found: {list(artifact.keys())}")

    per_layer: dict[int, np.ndarray] = {}
    for layer_key, layer_data in artifact["per_layer"].items():
        layer = int(layer_key)
        if method not in layer_data:
            raise KeyError(
                f"Method '{method}' not found in layer {layer}. "
                f"Available: {list(layer_data.keys())}"
            )
        vec = layer_data[method]["vector"]
        if isinstance(vec, torch.Tensor):
            vec = vec.float().cpu().numpy()
        # Ensure unit norm (artifact should already be normalised, but be safe).
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        per_layer[layer] = vec.astype(np.float32)

    return per_layer


def load_final_vector(axis: str, method: str = "mean_difference") -> np.ndarray:
    """Load the quality-weighted final vector (4096,) for *axis*."""
    path = VECTOR_DIR / f"{axis}_vectors.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    vec = artifact["final_vectors"][method]
    if isinstance(vec, torch.Tensor):
        vec = vec.float().cpu().numpy()
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 1e-8 else vec


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    texts: list[str],
    layers: list[int],
    model,
    tokenizer,
    batch_size: int = ACTIVATION_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> dict[int, np.ndarray]:
    """
    Extract mean-pooled hidden-state activations for *texts* at *layers*.

    HuggingFace convention: hidden_states[0] = embedding output,
    hidden_states[L+1] = output of transformer block L.
    We use hidden_states[L+1] for each requested layer L.

    Returns {layer: np.ndarray of shape (n_texts, 4096)} in float32 on CPU.
    """
    n = len(texts)
    # Pre-allocate output arrays.
    results: dict[int, list[np.ndarray]] = {L: [] for L in layers}

    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # hidden_states is a tuple of length (num_layers + 1).
        # Index 0 is embedding; index L+1 is block L's output.
        hs = out.hidden_states  # tuple[(batch, seq, 4096)]
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)

        for L in layers:
            # Pool over non-padding tokens.
            h = hs[L + 1].float()  # (batch, seq, 4096) — convert to fp32 for pooling
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # (batch, 4096)
            results[L].append(pooled.cpu().numpy())

        del out, hs, input_ids, attention_mask, mask
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        log.debug("Extracted activations for batch %d/%d", start + len(batch_texts), n)

    return {L: np.concatenate(results[L], axis=0).astype(np.float32) for L in layers}


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def project_onto(activations: np.ndarray, unit_vec: np.ndarray) -> np.ndarray:
    """
    Signed scalar projection: dot product with unit steering vector.

    activations: (n, 4096) float32
    unit_vec:    (4096,)   float32, already unit norm

    Returns (n,) float32 of signed projection magnitudes.
    Convention: positive = towards positive ideological pole
      (economic right, or authoritarian).
    We do NOT re-normalise activations — the raw dot product preserves
    magnitude information needed for calibration tests.
    """
    return activations @ unit_vec  # (n,)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)


def skip_if_exists(path: Path, label: str) -> bool:
    """Return True (skip) if the file exists and FORCE_RERUN is False."""
    if path.exists() and not FORCE_RERUN:
        log.info("Skipping %s (output exists at %s). Set FORCE_RERUN=True to recompute.", label, path)
        return True
    return False


# ---------------------------------------------------------------------------
# Test 1: OOD Generalization
# ---------------------------------------------------------------------------

OOD_FALLBACK_PAIRS = [
    # axis, topic, template_id, positive_text, negative_text
    # economic axis: positive = right, negative = left
    ("economic", "minimum_wage", 0,
     "Minimum wage laws destroy jobs by raising employer costs.",
     "We must raise the minimum wage to ensure workers can afford a decent life."),
    ("economic", "healthcare", 0,
     "Private healthcare markets, driven by competition, deliver the best outcomes.",
     "Universal healthcare funded by taxation is the only fair solution."),
    ("economic", "taxation", 0,
     "High progressive taxes punish success and stifle innovation.",
     "Wealth inequality demands redistribution through progressive taxation."),
    ("economic", "trade_unions", 0,
     "Trade unions inflate wages beyond market rates and harm competitiveness.",
     "Trade unions are essential to protect workers from corporate exploitation."),
    ("economic", "infrastructure", 0,
     "Government spending on infrastructure crowds out private investment.",
     "Public infrastructure investment creates jobs and stimulates the whole economy."),
    # social axis: positive = authoritarian, negative = libertarian
    ("social", "surveillance", 0,
     "Surveillance cameras in public spaces are necessary to prevent crime and terrorism.",
     "Mass surveillance violates citizens' fundamental right to privacy."),
    ("social", "drugs", 0,
     "Drug use must remain criminalised to protect the social fabric.",
     "Drug use is a personal choice; criminalisation only causes further harm."),
    ("social", "speech", 0,
     "The government must regulate online speech to reduce hate and misinformation.",
     "Free speech must be protected even when the content is offensive or wrong."),
    ("social", "military_service", 0,
     "Mandatory military service builds national cohesion and shared responsibility.",
     "Military service must be entirely voluntary; the state cannot compel it."),
    ("social", "social_values", 0,
     "The state has a duty to uphold and enforce traditional moral values.",
     "Individuals should be completely free to define their own values and lifestyle."),
]


def run_ood_generalization(
    layers: list[int] | None = None,
    model=None,
    tokenizer=None,
    batch_size: int | None = None,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Test 1: OOD generalization to new contrastive pairs."""
    out_path = OUT_DIR / "ood_generalization_results.csv"
    item_path = OUT_DIR / "ood_generalization_item_level.csv"
    if skip_if_exists(out_path, "OOD generalization"):
        return pd.read_csv(out_path)

    layers = layers or LAYERS
    batch_size = batch_size or ACTIVATION_BATCH_SIZE
    model, tokenizer = load_model() if model is None else (model, tokenizer)

    # Load external pairs if available.
    ext_path = EXTERNAL_DIR / "ood_pairs.csv"
    if ext_path.exists():
        log.info("Loading external OOD pairs from %s", ext_path)
        pairs_df = pd.read_csv(ext_path)
        pairs = [
            (r["axis"], r["topic"], r["template_id"], r["positive_text"], r["negative_text"])
            for _, r in pairs_df.iterrows()
        ]
    else:
        log.info("No external OOD pairs found. Using built-in fallback dataset.")
        pairs = OOD_FALLBACK_PAIRS

    if smoke_test:
        pairs = pairs[:4]

    # Load vectors.
    vectors: dict[str, dict[int, np.ndarray]] = {
        ax: load_vectors(ax) for ax in AXES
    }

    item_rows = []
    summary_rows = []

    for ax in AXES:
        ax_pairs = [(t, ti, pos, neg) for a, t, ti, pos, neg in pairs if a == ax]
        if not ax_pairs:
            log.warning("No OOD pairs for axis '%s'.", ax)
            continue

        pos_texts = [pos for _, _, pos, _ in ax_pairs]
        neg_texts = [neg for _, _, _, neg in ax_pairs]
        topics = [t for t, _, _, _ in ax_pairs]

        all_texts = pos_texts + neg_texts
        acts = extract_activations(all_texts, layers, model, tokenizer, batch_size)

        n = len(ax_pairs)
        for L in layers:
            pos_acts = acts[L][:n]
            neg_acts = acts[L][n:]
            v = vectors[ax][L]

            pos_proj = project_onto(pos_acts, v)
            neg_proj = project_onto(neg_acts, v)

            for i, topic in enumerate(topics):
                item_rows.append({
                    "axis": ax, "layer": L, "topic": topic,
                    "pos_proj": float(pos_proj[i]),
                    "neg_proj": float(neg_proj[i]),
                    "correct": int(pos_proj[i] > neg_proj[i]),
                })

            sep = float(pos_proj.mean() - neg_proj.mean())
            acc = float((pos_proj > neg_proj).mean())

            # AUC: labels 1 for pos, 0 for neg.
            try:
                from sklearn.metrics import roc_auc_score
                labels = np.array([1] * n + [0] * n)
                scores = np.concatenate([pos_proj, neg_proj])
                auc = float(roc_auc_score(labels, scores))
            except Exception:
                auc = float("nan")

            summary_rows.append({
                "axis": ax, "layer": L,
                "mean_pos_proj": float(pos_proj.mean()),
                "mean_neg_proj": float(neg_proj.mean()),
                "separation": sep,
                "sign_accuracy": acc,
                "auc": auc,
                "n_pairs": n,
                "dataset": "external" if ext_path.exists() else "fallback",
            })

    df_summary = pd.DataFrame(summary_rows)
    df_items = pd.DataFrame(item_rows)

    df_summary.to_csv(out_path, index=False)
    df_items.to_csv(item_path, index=False)
    log.info("OOD results saved to %s", out_path)

    _plot_ood(df_summary)
    return df_summary


def _plot_ood(df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, ax_name in enumerate(AXES):
            sub = df[df["axis"] == ax_name]
            ax = axes[i]
            ax.plot(sub["layer"], sub["separation"], marker="o", label="separation")
            ax.plot(sub["layer"], sub["sign_accuracy"], marker="s", linestyle="--", label="sign acc")
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
            ax.set_title(f"OOD: {ax_name}")
            ax.set_xlabel("Layer")
            ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "ood_separation_by_layer.png", dpi=120)
        plt.close(fig)
        log.info("Saved OOD plot.")
    except Exception as e:
        log.warning("Could not save OOD plot: %s", e)


# ---------------------------------------------------------------------------
# Test 2: Political Compass Test external anchor
# ---------------------------------------------------------------------------

PCT_FALLBACK_ITEMS = [
    # item_id, text, expected_economic_direction, expected_social_direction, intensity
    ("eco_L1", "Governments should redistribute wealth from the rich to the poor.", "left", "neutral", 1.0),
    ("eco_L2", "Nationalised industries are run more fairly than privatised ones.", "left", "neutral", 0.8),
    ("eco_L3", "Private enterprise is the backbone of a prosperous economy.", "right", "neutral", 0.8),
    ("eco_L4", "The free market is the most efficient allocator of resources.", "right", "neutral", 1.0),
    ("eco_L5", "Workers deserve a greater share of their company's profits.", "left", "neutral", 0.6),
    ("eco_R1", "Inheritance tax is an unfair double taxation of wealth.", "right", "neutral", 0.8),
    ("eco_R2", "Cutting taxes on corporations encourages investment and growth.", "right", "neutral", 0.8),
    ("soc_A1", "A strong state is needed to maintain order and protect citizens.", "neutral", "authoritarian", 0.8),
    ("soc_A2", "The government has a right to restrict freedoms to maintain stability.", "neutral", "authoritarian", 1.0),
    ("soc_A3", "Traditional values are the bedrock of a healthy society.", "neutral", "authoritarian", 0.7),
    ("soc_L1", "People should be free to make their own choices without government interference.", "neutral", "libertarian", 1.0),
    ("soc_L2", "Civil liberties should not be sacrificed for national security.", "neutral", "libertarian", 0.9),
    ("soc_L3", "Censorship of ideas, even dangerous ones, is never justified.", "neutral", "libertarian", 0.8),
    ("mix_1", "Free markets and personal freedom go hand in hand.", "right", "libertarian", 0.7),
    ("mix_2", "A planned economy combined with firm social discipline is the ideal.", "left", "authoritarian", 0.7),
]


def run_pct_external_anchor(
    layers: list[int] | None = None,
    model=None,
    tokenizer=None,
    batch_size: int | None = None,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Test 2: PCT-style external anchor projection."""
    out_path = OUT_DIR / "pct_external_anchor_results.csv"
    if skip_if_exists(out_path, "PCT external anchor"):
        return pd.read_csv(out_path)

    layers = layers or LAYERS
    batch_size = batch_size or ACTIVATION_BATCH_SIZE
    model, tokenizer = load_model() if model is None else (model, tokenizer)

    ext_path = EXTERNAL_DIR / "pct_items.csv"
    if ext_path.exists():
        log.info("Loading PCT items from %s", ext_path)
        df_items = pd.read_csv(ext_path)
        items = [
            (r["item_id"], r["text"],
             r.get("expected_economic_direction", ""),
             r.get("expected_social_direction", ""),
             r.get("intensity", 1.0))
            for _, r in df_items.iterrows()
        ]
    else:
        log.info("No external PCT items found. Using built-in fallback/demo set.")
        items = PCT_FALLBACK_ITEMS

    if smoke_test:
        items = items[:6]

    texts = [text for _, text, *_ in items]
    acts = extract_activations(texts, layers, model, tokenizer, batch_size)

    econ_vecs = load_vectors("economic")
    soc_vecs = load_vectors("social")

    # Use middle layer (L=16) as primary; also compute final-vector projection.
    primary_layer = 16
    econ_final = load_final_vector("economic")
    soc_final = load_final_vector("social")

    rows = []
    for j, (iid, text, exp_econ, exp_soc, intensity) in enumerate(items):
        row: dict[str, Any] = {
            "item_id": iid,
            "text": text,
            "expected_economic_direction": exp_econ,
            "expected_social_direction": exp_soc,
            "intensity": intensity,
        }

        for L in layers:
            act = acts[L][j : j + 1]  # (1, 4096)
            econ_proj = float(project_onto(act, econ_vecs[L])[0])
            soc_proj = float(project_onto(act, soc_vecs[L])[0])
            row[f"econ_proj_L{L}"] = econ_proj
            row[f"soc_proj_L{L}"] = soc_proj

        # Final vector projections.
        act_primary = acts[primary_layer][j : j + 1]
        row["econ_proj_final"] = float(project_onto(acts[layers[-1]][j:j+1], econ_final)[0])
        row["soc_proj_final"] = float(project_onto(acts[layers[-1]][j:j+1], soc_final)[0])

        # Sign agreement at primary layer.
        econ_p = row[f"econ_proj_L{primary_layer}"]
        soc_p = row[f"soc_proj_L{primary_layer}"]
        row["econ_sign_agree"] = _sign_agree(econ_p, exp_econ, "right", "left")
        row["soc_sign_agree"] = _sign_agree(soc_p, exp_soc, "authoritarian", "libertarian")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log.info("PCT results saved to %s", out_path)

    _plot_pct(df, primary_layer)
    return df


def _sign_agree(proj: float, expected: str, pos_label: str, neg_label: str) -> str:
    if expected == pos_label:
        return "agree" if proj > 0 else "disagree"
    elif expected == neg_label:
        return "agree" if proj < 0 else "disagree"
    return "na"


def _plot_pct(df: pd.DataFrame, layer: int):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = {"left": "blue", "right": "red", "neutral": "gray", "": "gray"}
        for _, row in df.iterrows():
            c = colors.get(str(row.get("expected_economic_direction", "")).lower(), "gray")
            ax.scatter(
                row[f"econ_proj_L{layer}"],
                row[f"soc_proj_L{layer}"],
                color=c, alpha=0.7,
            )
            ax.annotate(str(row["item_id"]), (row[f"econ_proj_L{layer}"], row[f"soc_proj_L{layer}"]),
                        fontsize=7, alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Economic projection (+ = right)")
        ax.set_ylabel("Social projection (+ = authoritarian)")
        ax.set_title(f"PCT-style anchor (layer {layer})")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "pct_economic_social_scatter.png", dpi=120)
        plt.close(fig)
        log.info("Saved PCT scatter plot.")
    except Exception as e:
        log.warning("Could not save PCT plot: %s", e)


# ---------------------------------------------------------------------------
# Test 3: Causal activation-addition
# ---------------------------------------------------------------------------

NEUTRAL_PROMPTS = [
    "The government's approach to taxation policy is",
    "When it comes to welfare spending, the current policy",
    "The question of personal liberty in modern society is",
    "National security and border control measures are",
    "Immigration policy in the twenty-first century should be",
    "Market regulation of financial institutions is",
]


def _make_steering_hook(vector_np: np.ndarray, alpha: float):
    """
    Return a forward hook that adds alpha * vector to the hidden state.

    Mistral decoder layer output is typically a tuple whose first element
    is the hidden-state tensor (batch, seq, hidden_dim). We modify only that
    element and leave everything else untouched.
    """
    vec_t = torch.tensor(vector_np, dtype=DTYPE).to(DEVICE)  # (4096,)

    def hook(module, input_, output):
        if isinstance(output, tuple):
            h = output[0] + alpha * vec_t  # broadcasts over batch & seq
            return (h,) + output[1:]
        else:
            return output + alpha * vec_t

    return hook


def run_causal_activation_addition(
    layers: list[int] | None = None,
    model=None,
    tokenizer=None,
    alpha_values: list[float] | None = None,
    prompts: list[str] | None = None,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test 3: Causal activation-addition — sweep alpha, generate text, re-project."""
    gen_path = OUT_DIR / "activation_addition_generations.jsonl"
    score_path = OUT_DIR / "activation_addition_scores.csv"
    if skip_if_exists(gen_path, "causal activation-addition"):
        df_scores = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
        return pd.DataFrame(), df_scores

    layers = layers or LAYERS
    alpha_values = alpha_values or ALPHA_VALUES
    prompts = prompts or NEUTRAL_PROMPTS
    model, tokenizer = load_model() if model is None else (model, tokenizer)

    if smoke_test:
        layers = [16]
        prompts = prompts[:2]
        alpha_values = [-1.5, 0.0, 1.5]

    vectors: dict[str, dict[int, np.ndarray]] = {
        ax: load_vectors(ax) for ax in AXES
    }
    score_vecs = vectors  # reuse for re-projection

    # Confirm model architecture: Mistral blocks are at model.model.layers[L].
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "Expected model.model.layers[L] but the architecture differs. "
            "Adjust the hook target in run_causal_activation_addition()."
        )

    gen_records = []
    score_rows = []

    for ax in AXES:
        for L in layers:
            vec_np = vectors[ax][L]
            for alpha in alpha_values:
                target_module = model.model.layers[L]
                hook_fn = _make_steering_hook(vec_np, alpha)
                handle = target_module.register_forward_hook(hook_fn)

                for prompt in prompts:
                    try:
                        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                                        max_length=MAX_LENGTH).to(DEVICE)
                        with torch.inference_mode():
                            out_ids = model.generate(
                                **enc,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        generated = tokenizer.decode(
                            out_ids[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        )
                    finally:
                        handle.remove()
                        handle = target_module.register_forward_hook(hook_fn)

                    record = {
                        "axis": ax, "layer": L, "alpha": alpha,
                        "prompt": prompt, "generated_text": generated,
                    }
                    gen_records.append(record)

                # Remove hook cleanly after all prompts at this (ax, L, alpha).
                handle.remove()
                # Verify no hooks remain on this module.
                assert len(target_module._forward_hooks) == 0, (
                    f"Hook leak detected on layer {L}!"
                )

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

    # Save generations.
    with open(gen_path, "w") as f:
        for rec in gen_records:
            f.write(json.dumps(rec) + "\n")
    log.info("Saved %d generation records to %s", len(gen_records), gen_path)

    # Re-project generated texts onto the same vectors.
    log.info("Re-projecting generated texts to build monotonicity table...")
    for ax in AXES:
        ax_recs = [r for r in gen_records if r["axis"] == ax]
        if not ax_recs:
            continue
        gen_texts = [r["generated_text"] for r in ax_recs]
        # Encode in batches.
        acts = extract_activations(gen_texts, layers, model, tokenizer, ACTIVATION_BATCH_SIZE)
        for i, rec in enumerate(ax_recs):
            L = rec["layer"]
            proj = float(project_onto(acts[L][i : i + 1], score_vecs[ax][L])[0])
            score_rows.append({
                "axis": rec["axis"], "layer": L, "alpha": rec["alpha"],
                "prompt": rec["prompt"], "proj": proj,
            })

    df_scores = pd.DataFrame(score_rows)
    df_scores.to_csv(score_path, index=False)
    log.info("Saved projection scores to %s", score_path)

    _plot_causal(df_scores)
    return pd.DataFrame(gen_records), df_scores


def _plot_causal(df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
        if df.empty:
            return
        axes_list = df["axis"].unique()
        layers_list = sorted(df["layer"].unique())
        fig, axes_grid = plt.subplots(
            len(axes_list), len(layers_list),
            figsize=(4 * len(layers_list), 4 * len(axes_list)),
            squeeze=False,
        )
        for i, ax_name in enumerate(axes_list):
            for j, L in enumerate(layers_list):
                sub = df[(df["axis"] == ax_name) & (df["layer"] == L)]
                grp = sub.groupby("alpha")["proj"].mean().reset_index()
                ax = axes_grid[i][j]
                ax.plot(grp["alpha"], grp["proj"], marker="o")
                ax.axhline(0, color="gray", linestyle=":")
                ax.set_title(f"{ax_name} L{L}")
                ax.set_xlabel("alpha")
                ax.set_ylabel("mean proj")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "activation_addition_alpha_vs_proj.png", dpi=120)
        plt.close(fig)
        log.info("Saved causal activation-addition plot.")
    except Exception as e:
        log.warning("Could not save causal plot: %s", e)


# ---------------------------------------------------------------------------
# Test 4: Paraphrase / length / pre-prompt confound check
# ---------------------------------------------------------------------------

PARAPHRASE_GROUPS = {
    "econ_left": {
        "axis": "economic",
        "expected_direction": "left",
        "items": [
            "We should tax the rich more to fund public services.",
            "Higher taxes on wealthy individuals are necessary to fund public services.",
            "The affluent must contribute more through increased taxation to fund our public services.",
            "Raising taxes on high earners is justified to pay for public goods.",
            "Note: Consider the following statement. We should tax the rich more.",  # pre-prompt
        ],
    },
    "econ_right": {
        "axis": "economic",
        "expected_direction": "right",
        "items": [
            "Lower taxes stimulate economic growth.",
            "Tax cuts lead to more investment and stronger growth.",
            "Reducing the overall tax burden helps businesses grow and creates jobs.",
            "Lower tax rates boost the economy by leaving more money in private hands.",
            "Note: Consider the following statement. Lower taxes stimulate economic growth.",
        ],
    },
    "soc_auth": {
        "axis": "social",
        "expected_direction": "authoritarian",
        "items": [
            "The state must enforce order and discipline to protect citizens.",
            "A firm government hand is needed to maintain order and protect citizens from harm.",
            "Strong enforcement of law and order by the state is essential for protecting the public.",
            "Government discipline and law enforcement are crucial for maintaining the safety of citizens.",
            "Note: Consider the following statement. The state must enforce order to protect citizens.",
        ],
    },
    "soc_lib": {
        "axis": "social",
        "expected_direction": "libertarian",
        "items": [
            "Individuals must be free to make their own choices without state interference.",
            "People should have the freedom to make their own choices free from government interference.",
            "The right of individuals to make their own decisions without state intrusion must be protected.",
            "Personal freedom from government interference is a fundamental right of every citizen.",
            "Note: Consider the following statement. Individuals must be free without state interference.",
        ],
    },
}


def run_paraphrase_confound_check(
    layers: list[int] | None = None,
    model=None,
    tokenizer=None,
    batch_size: int | None = None,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Test 4: Paraphrase/length confound sensitivity."""
    out_path = OUT_DIR / "paraphrase_confound_results.csv"
    if skip_if_exists(out_path, "paraphrase confound"):
        return pd.read_csv(out_path)

    layers = layers or LAYERS
    batch_size = batch_size or ACTIVATION_BATCH_SIZE
    model, tokenizer = load_model() if model is None else (model, tokenizer)

    vectors = {ax: load_vectors(ax) for ax in AXES}

    all_texts = []
    meta = []
    for group_name, grp in PARAPHRASE_GROUPS.items():
        for item in grp["items"]:
            all_texts.append(item)
            meta.append({"group": group_name, "axis": grp["axis"],
                         "expected_direction": grp["expected_direction"], "text": item})

    if smoke_test:
        all_texts = all_texts[:8]
        meta = meta[:8]

    acts = extract_activations(all_texts, layers, model, tokenizer, batch_size)

    rows = []
    for i, m in enumerate(meta):
        row = dict(m)
        for L in layers:
            v = vectors[m["axis"]][L]
            proj = float(project_onto(acts[L][i : i + 1], v)[0])
            row[f"proj_L{L}"] = proj
        rows.append(row)

    df_items = pd.DataFrame(rows)

    # Group-level summary.
    primary_layer = 16
    proj_col = f"proj_L{primary_layer}"
    summary_rows = []
    for group_name, grp in PARAPHRASE_GROUPS.items():
        sub = df_items[df_items["group"] == group_name][proj_col].values
        if len(sub) == 0:
            continue
        summary_rows.append({
            "group": group_name,
            "axis": grp["axis"],
            "expected_direction": grp["expected_direction"],
            "n": len(sub),
            "mean_proj": float(sub.mean()),
            "std_proj": float(sub.std()),
            "layer": primary_layer,
        })

    # Between-group ideological separation for same axis.
    for ax in AXES:
        ax_grps = [r for r in summary_rows if r["axis"] == ax]
        if len(ax_grps) < 2:
            continue
        means = [r["mean_proj"] for r in ax_grps]
        between_sep = float(max(means) - min(means))
        avg_within_std = float(np.mean([r["std_proj"] for r in ax_grps]))
        ratio = between_sep / (avg_within_std + 1e-8)
        log.info(
            "Paraphrase [%s]: between-group sep=%.3f, avg within-group std=%.3f, ratio=%.2f",
            ax, between_sep, avg_within_std, ratio,
        )
        for r in ax_grps:
            r["between_group_sep"] = between_sep
            r["avg_within_std"] = avg_within_std
            r["sep_to_variance_ratio"] = ratio

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_path, index=False)
    df_items.to_csv(OUT_DIR / "paraphrase_confound_item_level.csv", index=False)
    log.info("Paraphrase confound results saved to %s", out_path)

    _plot_paraphrase(df_items, primary_layer)
    return df_summary


def _plot_paraphrase(df: pd.DataFrame, layer: int):
    try:
        import matplotlib.pyplot as plt
        proj_col = f"proj_L{layer}"
        if proj_col not in df.columns:
            return
        groups = df["group"].unique()
        fig, ax = plt.subplots(figsize=(8, 5))
        for g in groups:
            vals = df[df["group"] == g][proj_col].values
            ax.scatter([g] * len(vals), vals, alpha=0.7, label=g)
            ax.plot([g], [vals.mean()], marker="_", markersize=20, color="black", linewidth=2)
        ax.set_xlabel("Paraphrase group")
        ax.set_ylabel(f"Projection at layer {layer}")
        ax.set_title("Paraphrase group variance vs ideological separation")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "paraphrase_group_variance.png", dpi=120)
        plt.close(fig)
        log.info("Saved paraphrase variance plot.")
    except Exception as e:
        log.warning("Could not save paraphrase plot: %s", e)


# ---------------------------------------------------------------------------
# Test 5: Magnitude calibration
# ---------------------------------------------------------------------------

MAGNITUDE_ITEMS = [
    # axis, intensity, pole, text
    # economic axis
    ("economic", "mild", "right",
     "Some deregulation might benefit small businesses in certain sectors."),
    ("economic", "strong", "right",
     "The free market must operate completely without government interference or regulation whatsoever."),
    ("economic", "mild", "left",
     "A modest minimum wage increase might improve conditions for low-paid workers."),
    ("economic", "strong", "left",
     "All private wealth should be abolished and redistributed equally among every citizen."),
    # social axis
    ("social", "mild", "authoritarian",
     "Some additional law-enforcement resources could help reduce crime in high-risk areas."),
    ("social", "strong", "authoritarian",
     "The state must have total authority over citizens' lives to guarantee order and national security."),
    ("social", "mild", "libertarian",
     "Individuals should generally be free to make personal lifestyle choices without undue interference."),
    ("social", "strong", "libertarian",
     "The state has absolutely no right to restrict any individual freedom under any circumstances whatsoever."),
]


def run_magnitude_calibration(
    layers: list[int] | None = None,
    model=None,
    tokenizer=None,
    batch_size: int | None = None,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Test 5: Magnitude calibration — mild vs strong intensity."""
    out_path = OUT_DIR / "magnitude_calibration_results.csv"
    if skip_if_exists(out_path, "magnitude calibration"):
        return pd.read_csv(out_path)

    layers = layers or LAYERS
    batch_size = batch_size or ACTIVATION_BATCH_SIZE
    model, tokenizer = load_model() if model is None else (model, tokenizer)

    texts = [t for _, _, _, t in MAGNITUDE_ITEMS]
    acts = extract_activations(texts, layers, model, tokenizer, batch_size)

    vectors = {ax: load_vectors(ax) for ax in AXES}

    rows = []
    for i, (ax, intensity, pole, text) in enumerate(MAGNITUDE_ITEMS):
        row = {"axis": ax, "intensity": intensity, "pole": pole, "text": text}
        for L in layers:
            v = vectors[ax][L]
            proj = float(project_onto(acts[L][i : i + 1], v)[0])
            # Sign: right/authoritarian poles are positive.
            if pole in ("left", "libertarian"):
                proj = -proj  # flip so larger abs means stronger in pos direction
            row[f"abs_proj_L{L}"] = abs(proj)
            row[f"signed_proj_L{L}"] = proj
        rows.append(row)

    df = pd.DataFrame(rows)

    # Summary: what fraction of mild/strong pairs has larger |proj| for strong?
    primary_layer = 16
    proj_col = f"abs_proj_L{primary_layer}"
    summary_rows = []
    n_correct = 0
    n_pairs = 0
    for ax in AXES:
        for pole in df[df["axis"] == ax]["pole"].unique():
            sub = df[(df["axis"] == ax) & (df["pole"] == pole)]
            mild = sub[sub["intensity"] == "mild"][proj_col].values
            strong = sub[sub["intensity"] == "strong"][proj_col].values
            if len(mild) == 0 or len(strong) == 0:
                continue
            correct = int(strong.mean() > mild.mean())
            n_correct += correct
            n_pairs += 1
            summary_rows.append({
                "axis": ax, "pole": pole,
                "mild_mean_abs_proj": float(mild.mean()),
                "strong_mean_abs_proj": float(strong.mean()),
                "strong_larger": bool(correct),
                "layer": primary_layer,
            })

    pct_correct = 100.0 * n_correct / max(n_pairs, 1)
    log.info("Magnitude calibration: %.0f%% of mild/strong pairs ordered correctly.", pct_correct)

    df.to_csv(out_path, index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "magnitude_calibration_summary.csv", index=False)
    log.info("Magnitude calibration results saved to %s", out_path)

    _plot_magnitude(df, primary_layer)
    return df


def _plot_magnitude(df: pd.DataFrame, layer: int):
    try:
        import matplotlib.pyplot as plt
        abs_col = f"abs_proj_L{layer}"
        if abs_col not in df.columns:
            return
        groups = df.groupby(["axis", "intensity"])[abs_col].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, ax_name in enumerate(AXES):
            sub = groups[groups["axis"] == ax_name]
            ax = axes[i]
            ax.bar(sub["intensity"], sub[abs_col], color=["#9ecae1", "#3182bd"])
            ax.set_title(f"{ax_name}: mild vs strong")
            ax.set_xlabel("Intensity")
            ax.set_ylabel(f"|projection| at layer {layer}")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "magnitude_mild_vs_strong.png", dpi=120)
        plt.close(fig)
        log.info("Saved magnitude calibration plot.")
    except Exception as e:
        log.warning("Could not save magnitude plot: %s", e)


# ---------------------------------------------------------------------------
# Orchestrator / CLI
# ---------------------------------------------------------------------------

def run_all_tests(
    layers: list[int] | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Run all 5 Section 9 tests in sequence. Returns dict of result DataFrames."""
    ensure_dirs()
    model, tokenizer = load_model()

    bs = 2 if smoke_test else ACTIVATION_BATCH_SIZE

    log.info("=== Test 1: OOD Generalization ===")
    t1 = run_ood_generalization(layers=layers, model=model, tokenizer=tokenizer,
                                batch_size=bs, smoke_test=smoke_test)

    log.info("=== Test 2: PCT External Anchor ===")
    t2 = run_pct_external_anchor(layers=layers, model=model, tokenizer=tokenizer,
                                 batch_size=bs, smoke_test=smoke_test)

    log.info("=== Test 3: Causal Activation Addition ===")
    t3_gen, t3_scores = run_causal_activation_addition(
        layers=layers, model=model, tokenizer=tokenizer,
        smoke_test=smoke_test,
    )

    log.info("=== Test 4: Paraphrase Confound Check ===")
    t4 = run_paraphrase_confound_check(layers=layers, model=model, tokenizer=tokenizer,
                                       batch_size=bs, smoke_test=smoke_test)

    log.info("=== Test 5: Magnitude Calibration ===")
    t5 = run_magnitude_calibration(layers=layers, model=model, tokenizer=tokenizer,
                                   batch_size=bs, smoke_test=smoke_test)

    log.info("All Section 9 tests complete. Outputs in: %s", OUT_DIR)
    return {
        "ood": t1,
        "pct": t2,
        "causal_generations": t3_gen,
        "causal_scores": t3_scores,
        "paraphrase": t4,
        "magnitude": t5,
    }


def main():
    parser = argparse.ArgumentParser(description="Section 9 GPU reliability tests.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with tiny datasets and small batch sizes for validation.")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to evaluate. Default: 8 12 16 20 24.")
    parser.add_argument("--test", choices=["ood", "pct", "causal", "paraphrase", "magnitude", "all"],
                        default="all", help="Which test to run.")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Recompute even if output files exist.")
    args = parser.parse_args()

    global FORCE_RERUN
    if args.force_rerun:
        FORCE_RERUN = True

    layers = args.layers or ([16] if args.smoke_test else LAYERS)
    ensure_dirs()
    model, tokenizer = load_model()
    bs = 2 if args.smoke_test else ACTIVATION_BATCH_SIZE

    if args.test == "all":
        run_all_tests(layers=layers, smoke_test=args.smoke_test)
    elif args.test == "ood":
        run_ood_generalization(layers=layers, model=model, tokenizer=tokenizer,
                               batch_size=bs, smoke_test=args.smoke_test)
    elif args.test == "pct":
        run_pct_external_anchor(layers=layers, model=model, tokenizer=tokenizer,
                                batch_size=bs, smoke_test=args.smoke_test)
    elif args.test == "causal":
        run_causal_activation_addition(layers=layers, model=model, tokenizer=tokenizer,
                                       smoke_test=args.smoke_test)
    elif args.test == "paraphrase":
        run_paraphrase_confound_check(layers=layers, model=model, tokenizer=tokenizer,
                                      batch_size=bs, smoke_test=args.smoke_test)
    elif args.test == "magnitude":
        run_magnitude_calibration(layers=layers, model=model, tokenizer=tokenizer,
                                  batch_size=bs, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
