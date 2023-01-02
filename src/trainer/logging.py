from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

ISO_8601 = "%Y-%m-%d %H:%M:%S"
PROCESS_FMT_MARKUP = "[dim]%(processName)-11s[/dim] %(message)s"
PROCESS_FMT = re.sub(r"\[.*\]", "", re.sub(r"\[/.*\]", "", PROCESS_FMT_MARKUP))
FILE_HANDLER_FMT = f"%(asctime)s %(levelname)-8s {PROCESS_FMT}"


def create_rich_handler(level: int = logging.INFO) -> logging.Handler():

    rich_handler = RichHandler(
        level=level,
        highlighter=NullHighlighter(),
        markup=True,
        omit_repeated_times=False,
        log_time_format=ISO_8601,
        rich_tracebacks=True,
        tracebacks_width=80,
        tracebacks_extra_lines=0,
        tracebacks_word_wrap=False,
        tracebacks_suppress=os.environ["CONDA_PREFIX"],
    )

    formatter = logging.Formatter(fmt=PROCESS_FMT_MARKUP)
    rich_handler.setFormatter(formatter)

    return rich_handler


def create_file_handler(
    filename: Path,
    level: int = logging.INFO,
) -> logging.Handler():
    """
    Create logging FileHandler based on log filename specified in config
    """
    file_handler = logging.FileHandler(filename, mode="a", delay=True)
    formatter = logging.Formatter(fmt=FILE_HANDLER_FMT, datefmt=ISO_8601)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    return file_handler
