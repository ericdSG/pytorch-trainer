"""
A PyTorch evaluation template.

Created: Dec 2022 by Eric DeMattos
"""

import logging
import shutil
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from MLtools.AnimationMetric.AnimationMetric import AnimationMetric
from MLtools.CarnivalTools.carnival_tools import CarnivalRTS

from .data import get_classes
from .train import Trainer

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        cfg: DictConfig,
        trainer: Trainer,
        dl: DataLoader,
        rank: int = 0,
    ) -> None:
        self.cfg = cfg
        self.trainer = trainer
        self.dl = dl
        self.rank = rank

    def _write_temp_rts(
        self,
        predictions: list[torch.Tensor],
        utt_ids: list[str],
        rts_ref_dir: Path,
        out_dir: Path,
    ) -> None:
        """
        Convert PyTorch tensors to Carnival RTS files.
        """
        rts_ref_paths = rts_ref_dir.glob("*.rts")
        classes = get_classes(rts_ref_paths)

        for pred, utt_id in zip(predictions, utt_ids):
            pred = pred.cpu().numpy().squeeze().T
            rts = CarnivalRTS.load_from_dict(dict(zip(classes, pred)))
            rts.save(out_dir / f"{utt_id}.rts")

    def evaluate(self, model: str) -> None:

        # load model and run inference on test data
        self.trainer.load_checkpoint(self.trainer.cfg.experiment_dir / model)
        predictions, utt_ids = zip(*self.trainer.predict(self.dl, test=True))

        # parallelization no longer needed; only run eval from main process
        if self.rank != 0:
            return

        rts_ref_dir = self.cfg.train.data.y_dir
        temp_out_dir = rts_ref_dir.parent / "eval"
        temp_out_dir.mkdir(exist_ok=True, parents=True)

        # write tensors to file in RTS format
        self._write_temp_rts(predictions, utt_ids, rts_ref_dir, temp_out_dir)

        am = AnimationMetric(str(rts_ref_dir), str(temp_out_dir))
        am.compute_metric()
        am.log(level="set")

        # clean up
        shutil.rmtree(temp_out_dir)
