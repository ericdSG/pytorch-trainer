from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from MLtools.AudioLoader.src.core import AudioLoader, RTSBlock
from MLtools.AudioLoader.src.transforms import KaldiFbankTransform

logger = logging.getLogger(__name__)


def get_classes(rts_paths: list[str]) -> set[str]:
    classes = set()
    for rts in rts_paths:
        with open(rts) as f:
            f.readline()  # skip frame rate
            classes = classes | set(f.readline().strip().split(","))
    return classes


def create_dl(
    x_dir: Path,
    y_dir: Path,
    batch_size: int = 1,
    valid: bool = False,
) -> DataLoader | Tuple[DataLoader, DataLoader]:

    logging.debug("Creating DataLoader(s)")

    wav_paths = sorted([str(path) for path in Path(x_dir).glob("*.wav")])
    rts_paths = sorted([str(path) for path in Path(y_dir).glob("*.rts")])

    classes = get_classes(rts_paths)

    al = AudioLoader()
    al.load_audio(wav_paths, cache=False)
    al.load_labels(rts_paths, cls=RTSBlock, classes=classes, cache=False)

    if valid:
        # seed required for deterministic shuffling across GPU processes
        al.split_by_valid_pct(0.25, seed=0)

    al.add(transforms=[KaldiFbankTransform(num_mel_bins=40)])
    al.bs = batch_size

    if valid:
        t_dl = al.batches("train", max_pad=0.5)
        v_dl = al.batches("valid")
        return t_dl, v_dl
    else:
        return al.batches("test")
