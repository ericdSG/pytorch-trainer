"""
A PyTorch evaluation template.

Created: Dec 2022 by Eric DeMattos
"""

import logging
import shutil
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from MLtools.AnimationMetric.AnimationMetric import AnimationMetric
from MLtools.CarnivalTools.carnival_tools import CarnivalRTS

from .data import get_classes

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        dl: torch.utils.data.DataLoader,
        checkpoint: str,
    ) -> None:

        logging.debug("Setting up Evaluator")

        self.cfg = cfg
        self.model = model
        self.dl = dl
        self.checkpoint_path = self.cfg.experiment_dir / checkpoint
        self.device = cfg.cuda.visible_devices[0]

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

    def predict(self) -> list[Tuple[torch.Tensor, str]]:

        self.model.eval()  # switch off grad engine

        preds = [self.model(x.to(self.device)).detach() for x, _ in self.dl]
        utt_ids = [Path(path).stem for path in self.dl.dataset.labels]

        return [(pred, utt_id) for pred, utt_id in zip(preds, utt_ids)]

    def evaluate(self) -> None:

        # load model and run inference on test data
        self.load_checkpoint(self.checkpoint_path)
        predictions, utt_ids = zip(*self.predict())

        # write tensors to file in RTS format
        rts_ref_dir = self.cfg.train.data.y_dir
        temp_out_dir = rts_ref_dir.parent / "eval"
        temp_out_dir.mkdir(exist_ok=True, parents=True)
        self._write_temp_rts(predictions, utt_ids, rts_ref_dir, temp_out_dir)

        # log scores (similarity, power, score) to console and file
        am = AnimationMetric(str(rts_ref_dir), str(temp_out_dir))
        am.compute_metric()
        am.log(level="set")

        # clean up
        shutil.rmtree(temp_out_dir)

    def load_checkpoint(self, checkpoint_path: Path) -> None:

        logger.debug(f"Loading {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        if self.cfg.cuda.num_gpus == 1:
            consume_prefix_in_state_dict_if_present(
                checkpoint["model_state"], "module."
            )
        self.model.load_state_dict(checkpoint["model_state"])
