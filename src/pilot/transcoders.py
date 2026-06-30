"""Download and load the GPT-2 small transcoders from pchlenski/gpt2-transcoders."""

from __future__ import annotations

import os
import sys
from typing import List

import torch

_VENDOR = os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'transcoder_circuits')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from sae_training.sparse_autoencoder import SparseAutoencoder

_TC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'transcoders')
_TEMPLATE = "final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
_N_LAYERS = 12


def _tc_path(layer: int) -> str:
    return os.path.join(_TC_DIR, _TEMPLATE.format(layer) + ".pt")


def download_transcoders(force: bool = False) -> None:
    """Download all 12 GPT-2 small transcoders from HuggingFace (once)."""
    os.makedirs(_TC_DIR, exist_ok=True)
    missing = [l for l in range(_N_LAYERS) if not os.path.exists(_tc_path(l))]
    if not missing and not force:
        return

    print(f"Downloading {len(missing)} transcoder(s) from pchlenski/gpt2-transcoders …")
    from huggingface_hub import hf_hub_download
    for layer in missing:
        filename = _TEMPLATE.format(layer) + ".pt"
        hf_hub_download(
            repo_id="pchlenski/gpt2-transcoders",
            filename=filename,
            local_dir=_TC_DIR,
        )
        print(f"  layer {layer} OK")


def _load_transcoder_cpu(path: str) -> SparseAutoencoder:
    """Load a single transcoder checkpoint, forcing CPU regardless of saved device."""
    import pickle, gzip
    state = torch.load(path, map_location="cpu")
    cfg = state["cfg"]
    # Override device in cfg so all internal buffers are created on CPU
    cfg.device = "cpu"
    instance = SparseAutoencoder(cfg=cfg)
    instance.load_state_dict(state["state_dict"])
    return instance.eval()


def load_transcoders() -> List[SparseAutoencoder]:
    """Load all 12 transcoders into CPU eval mode."""
    download_transcoders()
    return [_load_transcoder_cpu(_tc_path(layer)) for layer in range(_N_LAYERS)]
