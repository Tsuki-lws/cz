"""
数据集加载器
支持加载SimpleQA和2WikiMultihopQA数据集
"""

import json
import random
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from loguru import logger

from config.settings import settings


@dataclass
class QAItem:
    """统一的QA数据项"""
    id: str
    question: str
    answer: str
    question_type: str = "unknown"  # factual/multi-hop/comparison
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetLoader:
    """
    数据集加载器

    支持的数据集：
    1. SimpleQA (SimpleVQA) - 简单问答/视觉问答
    2. 2WikiMultihopQA - 多跳推理问答
    """

    def __init__(self):
        self._cache: dict[str, list[QAItem]] = {}

    def load_simpleqa(
        self,
        data_path: Optional[str] = None,
        split: str = "test",
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> list[QAItem]:
        """
        加载SimpleQA数据集

        Args:
            data_path: 数据路径（默认从settings读取）
            split: 数据集划分
            sample_size: 采样大小（None=全部）
            seed: 随机种子

        Returns:
            list[QAItem]: QA数据列表
        """
        cache_key = f"simpleqa_{split}_{sample_size}_{seed}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data_path = Path(data_path or settings.eval.simpleqa_path)
        items = []

        # 尝试从本地parquet/json加载
        parquet_file = data_path / f"{split}.parquet"
        json_file = data_path / f"{split}.json"
        jsonl_file = data_path / f"{split}.jsonl"

        if parquet_file.exists():
            items = self._load_simpleqa_parquet(parquet_file)
        elif json_file.exists():
            items = self._load_simpleqa_json(json_file)
        elif jsonl_file.exists():
            items = self._load_simpleqa_jsonl(jsonl_file)
        else:
            # 从HuggingFace加载
            items = self._load_simpleqa_from_hf(split)

        if not items:
            logger.warning("SimpleQA dataset is empty. Please download the data first.")
            logger.info(
                "Download: huggingface-cli download m-a-p/SimpleVQA "
                f"--local-dir {data_path}"
            )
            return []

        # 采样
        if sample_size and sample_size < len(items):
            random.seed(seed)
            items = random.sample(items, sample_size)

        self._cache[cache_key] = items
        logger.info(f"Loaded SimpleQA: {len(items)} items (split={split})")
        return items

    def load_2wiki(
        self,
        data_path: Optional[str] = None,
        split: str = "validation",
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> list[QAItem]:
        """
        加载2WikiMultihopQA数据集

        Args:
            data_path: 数据路径
            split: 数据集划分 (train/validation/test)
            sample_size: 采样大小
            seed: 随机种子

        Returns:
            list[QAItem]: QA数据列表
        """
        cache_key = f"2wiki_{split}_{sample_size}_{seed}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data_path = Path(data_path or settings.eval.wiki2_path)
        items = []

        # 尝试多种本地格式
        parquet_file = data_path / f"{split}.parquet"
        json_file = data_path / f"{split}.json"
        jsonl_file = data_path / f"{split}.jsonl"

        if parquet_file.exists():
            items = self._load_2wiki_parquet(parquet_file)
        elif json_file.exists():
            items = self._load_2wiki_json(json_file)
        elif jsonl_file.exists():
            items = self._load_2wiki_jsonl(jsonl_file)
        else:
            items = self._load_2wiki_from_hf(split)

        if not items:
            logger.warning("2Wiki dataset is empty. Please download the data first.")
            logger.info(
                "Download: huggingface-cli download framolfese/2WikiMultihopQA "
                f"--local-dir {data_path}"
            )
            return []

        # 采样
        if sample_size and sample_size < len(items):
            random.seed(seed)
            items = random.sample(items, sample_size)

        self._cache[cache_key] = items
        logger.info(f"Loaded 2Wiki: {len(items)} items (split={split})")
        return items

    # ==================== SimpleQA 加载器 ====================

    def _load_simpleqa_parquet(self, path: Path) -> list[QAItem]:
        """从parquet加载SimpleQA"""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            items = []
            for _, row in df.iterrows():
                items.append(QAItem(
                    id=str(row.get("data_id", len(items))),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    question_type="factual",
                    metadata={
                        "category": str(row.get("vqa_category", "")),
                        "source": str(row.get("source", "")),
                    },
                ))
            return items
        except Exception as e:
            logger.error(f"Failed to load SimpleQA parquet: {e}")
            return []

    def _load_simpleqa_json(self, path: Path) -> list[QAItem]:
        """从JSON加载SimpleQA"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = []
            for entry in data:
                items.append(QAItem(
                    id=str(entry.get("data_id", len(items))),
                    question=str(entry.get("question", "")),
                    answer=str(entry.get("answer", "")),
                    question_type="factual",
                ))
            return items
        except Exception as e:
            logger.error(f"Failed to load SimpleQA JSON: {e}")
            return []

    def _load_simpleqa_jsonl(self, path: Path) -> list[QAItem]:
        """从JSONL加载SimpleQA"""
        items = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        items.append(QAItem(
                            id=str(entry.get("data_id", len(items))),
                            question=str(entry.get("question", "")),
                            answer=str(entry.get("answer", "")),
                            question_type="factual",
                        ))
        except Exception as e:
            logger.error(f"Failed to load SimpleQA JSONL: {e}")
        return items

    def _load_simpleqa_from_hf(self, split: str) -> list[QAItem]:
        """从HuggingFace加载SimpleQA"""
        try:
            from datasets import load_dataset
            ds = load_dataset("m-a-p/SimpleVQA", split=split)
            items = []
            for row in ds:
                items.append(QAItem(
                    id=str(row.get("data_id", len(items))),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    question_type="factual",
                    metadata={
                        "category": str(row.get("vqa_category", "")),
                    },
                ))
            return items
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")
            return []

    # ==================== 2Wiki 加载器 ====================

    def _load_2wiki_parquet(self, path: Path) -> list[QAItem]:
        """从parquet加载2Wiki"""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            items = []
            for _, row in df.iterrows():
                q_type = str(row.get("type", "unknown"))
                items.append(QAItem(
                    id=str(row.get("id", len(items))),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    question_type=self._map_2wiki_type(q_type),
                    metadata={
                        "type": q_type,
                        "supporting_facts": row.get("supporting_facts"),
                    },
                ))
            return items
        except Exception as e:
            logger.error(f"Failed to load 2Wiki parquet: {e}")
            return []

    def _load_2wiki_json(self, path: Path) -> list[QAItem]:
        """从JSON加载2Wiki"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = []
            for entry in data:
                q_type = str(entry.get("type", "unknown"))
                items.append(QAItem(
                    id=str(entry.get("id", entry.get("_id", len(items)))),
                    question=str(entry.get("question", "")),
                    answer=str(entry.get("answer", "")),
                    question_type=self._map_2wiki_type(q_type),
                    metadata={
                        "type": q_type,
                        "supporting_facts": entry.get("supporting_facts"),
                    },
                ))
            return items
        except Exception as e:
            logger.error(f"Failed to load 2Wiki JSON: {e}")
            return []

    def _load_2wiki_jsonl(self, path: Path) -> list[QAItem]:
        """从JSONL加载2Wiki"""
        items = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        q_type = str(entry.get("type", "unknown"))
                        items.append(QAItem(
                            id=str(entry.get("id", entry.get("_id", len(items)))),
                            question=str(entry.get("question", "")),
                            answer=str(entry.get("answer", "")),
                            question_type=self._map_2wiki_type(q_type),
                            metadata={
                                "type": q_type,
                            },
                        ))
        except Exception as e:
            logger.error(f"Failed to load 2Wiki JSONL: {e}")
        return items

    def _load_2wiki_from_hf(self, split: str) -> list[QAItem]:
        """从HuggingFace加载2Wiki"""
        try:
            from datasets import load_dataset
            ds = load_dataset("framolfese/2WikiMultihopQA", split=split)
            items = []
            for row in ds:
                q_type = str(row.get("type", "unknown"))
                items.append(QAItem(
                    id=str(row.get("id", len(items))),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    question_type=self._map_2wiki_type(q_type),
                    metadata={"type": q_type},
                ))
            return items
        except Exception as e:
            logger.warning(f"Failed to load 2Wiki from HuggingFace: {e}")
            return []

    def _map_2wiki_type(self, q_type: str) -> str:
        """映射2Wiki问题类型"""
        type_map = {
            "bridge_comparison": "multi-hop",
            "compositional": "multi-hop",
            "comparison": "comparison",
            "inference": "multi-hop",
        }
        return type_map.get(q_type, "multi-hop")
