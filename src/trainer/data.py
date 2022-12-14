from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

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


def get_dl(
    x_dir: Path,
    y_dir: Path,
    batch_size: int = 1,
    valid: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:

    logging.debug("Creating DataLoader(s)")

    wav_paths = sorted([str(path) for path in Path(x_dir).glob("*.wav")])
    rts_paths = sorted([str(path) for path in Path(y_dir).glob("*.rts")])

    classes = get_classes(rts_paths)

    al = AudioLoader()
    al.load_audio(wav_paths, cache=False)
    al.load_labels(rts_paths, cls=RTSBlock, classes=classes, cache=False)

    if valid:
        al.split_by_valid_pct(0.1)

    al.add(transforms=[KaldiFbankTransform(num_mel_bins=40)])
    al.bs = batch_size

    if valid:
        return al.batches("train"), al.batches("valid")
    else:
        return al.batches("test")
