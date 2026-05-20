#!/usr/bin/env python

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "distill/outputs/qwen35_9b_trackg_sft_full/zero_to_fp32.py"

text = SOURCE.read_text(encoding="utf-8")
text = text.replace(
    "from deepspeed.utils import logger\n"
    "from deepspeed.checkpoint.constants import (DS_VERSION, OPTIMIZER_STATE_DICT, SINGLE_PARTITION_OF_FP32_GROUPS,\n"
    "                                            FP32_FLAT_GROUPS, ZERO_STAGE, PARTITION_COUNT, PARAM_SHAPES, BUFFER_NAMES,\n"
    "                                            FROZEN_PARAM_SHAPES, FROZEN_PARAM_FRAGMENTS)\n",
    "class _Logger:\n"
    "    def info(self, msg):\n"
    "        print(msg)\n"
    "logger = _Logger()\n"
    "DS_VERSION = 'ds_version'\n"
    "OPTIMIZER_STATE_DICT = 'optimizer_state_dict'\n"
    "SINGLE_PARTITION_OF_FP32_GROUPS = 'single_partition_of_fp32_groups'\n"
    "FP32_FLAT_GROUPS = 'fp32_flat_groups'\n"
    "ZERO_STAGE = 'zero_stage'\n"
    "PARTITION_COUNT = 'partition_count'\n"
    "PARAM_SHAPES = 'param_shapes'\n"
    "BUFFER_NAMES = 'buffer_names'\n"
    "FROZEN_PARAM_SHAPES = 'frozen_param_shapes'\n"
    "FROZEN_PARAM_FRAGMENTS = 'frozen_param_fragments'\n",
)

namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
exec(compile(text, str(SOURCE), "exec"), namespace)
