"""
记忆模块
三层记忆架构：工作记忆 + 情景记忆 + 语义记忆
支持：结构化存储、语义检索、置信度更新、记忆整合
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings
from core.llm import LLMClient
from config.prompts import Prompts


# ==================== 数据结构 ====================


@dataclass
class EpisodicEntry:
    """情景记忆条目 - 具体的任务经验"""
    entry_id: str
    question: str
    question_type: str  # "factual", "multi-hop", "comparison"
    strategy_used: str  # 使用的策略描述
    outcome: str  # "success" / "failure"
    key_insight: str  # 关键洞察
    trajectory_summary: str  # 轨迹摘要
    confidence: float = 0.8
    usage_count: int = 0  # 被检索使用的次数
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[list] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)  # 不序列化embedding到JSON
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicEntry":
        d.pop("embedding", None)
        return cls(**d)


@dataclass
class SemanticRule:
    """语义记忆条目 - 抽象的通用规则"""
    rule_id: str
    content: str  # 规则内容
    applies_to: str  # 适用的任务类型
    confidence: float = 0.8
    verified_count: int = 0  # 被验证正确的次数
    violated_count: int = 0  # 被违反/失败的次数
    source_tasks: list = field(default_factory=list)  # 来源任务ID
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[list] = field(default=None, repr=False)

    @property
    def effective_confidence(self) -> float:
        """有效置信度（考虑验证和违反）"""
        total = self.verified_count + self.violated_count
        if total == 0:
            return self.confidence
        return self.confidence * (self.verified_count / total)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticRule":
        d.pop("embedding", None)
        return cls(**d)


# ==================== 嵌入模型 ====================


class EmbeddingModel:
    """轻量级嵌入模型封装"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.memory.embedding_model
        self._model = None

    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to keyword-based similarity."
                )
                self._model = "fallback"

    def encode(self, text: str) -> Optional[list]:
        """编码文本为向量"""
        self._load_model()
        if self._model == "fallback":
            return None
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[Optional[list]]:
        """批量编码"""
        self._load_model()
        if self._model == "fallback":
            return [None] * len(texts)
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @staticmethod
    def cosine_similarity(vec1: list, vec2: list) -> float:
        """计算余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ==================== 记忆系统 ====================


class MemorySystem:
    """
    三层记忆系统

    架构：
    - WorkingMemory: 当前任务的上下文缓冲（短期，不持久化）
    - EpisodicMemory: 具体的成功/失败经验（中期，持久化）
    - SemanticMemory: 抽象的通用规则（长期，持久化）

    核心操作：
    - store: 存储新经验
    - retrieve: 检索相关记忆
    - update: 更新置信度
    - consolidate: 整合记忆（提取新规则/合并相似规则）
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.memory_dir = Path(memory_dir or settings.memory.memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm

        # 三层记忆
        self.working_memory: list[str] = []  # 当前任务上下文
        self.episodic_memory: list[EpisodicEntry] = []
        self.semantic_memory: list[SemanticRule] = []

        # 嵌入模型
        self.embedder = EmbeddingModel()

        # 加载持久化的记忆
        self._load()

    # ==================== 存储 ====================

    def store_experience(
        self,
        question: str,
        outcome: str,
        strategy: str,
        insight: str,
        trajectory_summary: str = "",
        question_type: str = "unknown",
    ) -> EpisodicEntry:
        """
        存储一条新经验到情景记忆

        Args:
            question: 问题
            outcome: "success" or "failure"
            strategy: 使用的策略
            insight: 关键洞察
            trajectory_summary: 轨迹摘要
            question_type: 问题类型

        Returns:
            EpisodicEntry: 存储的条目
        """
        entry_id = hashlib.md5(
            f"{question}{time.time()}".encode()
        ).hexdigest()[:12]

        entry = EpisodicEntry(
            entry_id=entry_id,
            question=question,
            question_type=question_type,
            strategy_used=strategy,
            outcome=outcome,
            key_insight=insight,
            trajectory_summary=trajectory_summary,
        )

        # 计算嵌入
        entry.embedding = self.embedder.encode(question)

        # 容量控制
        if len(self.episodic_memory) >= settings.memory.max_episodic_entries:
            self._evict_episodic()

        self.episodic_memory.append(entry)
        self._save()

        logger.info(
            f"Stored episodic memory: {entry_id} | "
            f"outcome={outcome} | insight: {insight[:60]}"
        )
        return entry

    def store_rule(
        self,
        content: str,
        applies_to: str = "general",
        source_task_id: str = "",
        confidence: float = 0.8,
    ) -> SemanticRule:
        """
        存储一条新规则到语义记忆

        Args:
            content: 规则内容
            applies_to: 适用范围
            source_task_id: 来源任务ID
            confidence: 初始置信度

        Returns:
            SemanticRule: 存储的规则
        """
        # 检查是否已有相似规则
        existing = self._find_similar_rule(content)
        if existing:
            # 强化已有规则
            existing.verified_count += 1
            existing.confidence = min(1.0, existing.confidence + settings.memory.confidence_boost)
            if source_task_id:
                existing.source_tasks.append(source_task_id)
            self._save()
            logger.info(f"Reinforced existing rule: {existing.rule_id}")
            return existing

        # 创建新规则
        rule_id = hashlib.md5(content.encode()).hexdigest()[:12]
        rule = SemanticRule(
            rule_id=rule_id,
            content=content,
            applies_to=applies_to,
            confidence=confidence,
            source_tasks=[source_task_id] if source_task_id else [],
        )
        rule.embedding = self.embedder.encode(content)

        # 容量控制
        if len(self.semantic_memory) >= settings.memory.max_semantic_rules:
            self._evict_semantic()

        self.semantic_memory.append(rule)
        self._save()

        logger.info(f"Stored new rule: {rule_id} | {content[:60]}")
        return rule

    # ==================== 检索 ====================

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        检索与查询相关的记忆，返回格式化的文本

        Args:
            query: 查询文本（通常是新任务的问题）
            top_k: 返回结果数

        Returns:
            str: 格式化的记忆上下文文本（直接注入prompt）
        """
        top_k = top_k or settings.memory.retrieval_top_k

        # 检索情景记忆
        episodic_results = self._retrieve_episodic(query, top_k)

        # 检索语义规则
        semantic_results = self._retrieve_semantic(query, top_k)

        # 格式化输出
        context_parts = []

        if semantic_results:
            context_parts.append("### General Rules:")
            for rule in semantic_results:
                context_parts.append(
                    f"- [{rule.applies_to}] {rule.content} "
                    f"(confidence: {rule.effective_confidence:.1%})"
                )

        if episodic_results:
            context_parts.append("\n### Relevant Past Experiences:")
            for entry in episodic_results:
                status = "Success" if entry.outcome == "success" else "Failed"
                context_parts.append(
                    f"- [{status}] {entry.key_insight}"
                )

        return "\n".join(context_parts) if context_parts else ""

    def _retrieve_episodic(self, query: str, top_k: int) -> list[EpisodicEntry]:
        """检索情景记忆"""
        if not self.episodic_memory:
            return []

        query_embedding = self.embedder.encode(query)

        if query_embedding is not None:
            # 向量检索
            scores = []
            for entry in self.episodic_memory:
                if entry.embedding is not None:
                    sim = EmbeddingModel.cosine_similarity(query_embedding, entry.embedding)
                else:
                    sim = self._keyword_similarity(query, entry.question)
                scores.append(sim)
        else:
            # 关键词匹配备选
            scores = [
                self._keyword_similarity(query, entry.question)
                for entry in self.episodic_memory
            ]

        # 按相似度排序
        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for idx, score in indexed[:top_k]:
            if score >= settings.memory.similarity_threshold:
                entry = self.episodic_memory[idx]
                entry.usage_count += 1
                results.append(entry)

        return results

    def _retrieve_semantic(self, query: str, top_k: int) -> list[SemanticRule]:
        """检索语义规则"""
        if not self.semantic_memory:
            return []

        # 只返回有效置信度高的规则
        valid_rules = [
            r for r in self.semantic_memory
            if r.effective_confidence >= settings.memory.min_confidence
        ]

        if not valid_rules:
            return []

        query_embedding = self.embedder.encode(query)

        if query_embedding is not None:
            scores = []
            for rule in valid_rules:
                if rule.embedding is not None:
                    sim = EmbeddingModel.cosine_similarity(query_embedding, rule.embedding)
                else:
                    sim = self._keyword_similarity(query, rule.content)
                # 加权：考虑置信度
                scores.append(sim * rule.effective_confidence)
        else:
            scores = [
                self._keyword_similarity(query, rule.content) * rule.effective_confidence
                for rule in valid_rules
            ]

        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for idx, score in indexed[:top_k]:
            if score > 0:
                results.append(valid_rules[idx])

        return results

    # ==================== 更新 ====================

    def update_confidence(self, rule_id: str, success: bool):
        """
        更新规则置信度

        Args:
            rule_id: 规则ID
            success: 规则是否有效
        """
        for rule in self.semantic_memory:
            if rule.rule_id == rule_id:
                if success:
                    rule.verified_count += 1
                    rule.confidence = min(
                        1.0, rule.confidence + settings.memory.confidence_boost
                    )
                else:
                    rule.violated_count += 1
                    rule.confidence = max(
                        settings.memory.min_confidence,
                        rule.confidence - settings.memory.confidence_decay,
                    )
                self._save()
                logger.debug(
                    f"Updated rule {rule_id}: confidence={rule.confidence:.2f}"
                )
                break

    # ==================== 整合 ====================

    def consolidate(self):
        """
        记忆整合

        从情景记忆中提取新的语义规则，合并相似规则
        """
        if not self.llm or len(self.episodic_memory) < 5:
            return

        logger.info("Starting memory consolidation...")

        # 1. 从最近的经验中提取规则
        recent = self.episodic_memory[-20:]
        experiences_text = "\n".join([
            f"- [{e.outcome}] Q: {e.question[:80]} | Insight: {e.key_insight}"
            for e in recent
        ])

        prompt = Prompts.MEMORY_EXTRACT_INSIGHTS.format(experiences=experiences_text)

        try:
            response = self.llm.simple_generate(prompt)
            new_rules = self._parse_extracted_rules(response)

            for rule_text, applies_to in new_rules:
                self.store_rule(
                    content=rule_text,
                    applies_to=applies_to,
                    confidence=0.6,  # 新提取的规则初始置信度较低
                )

            logger.info(f"Consolidation complete: extracted {len(new_rules)} new rules")

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

        # 2. 清理低置信度规则
        self.semantic_memory = [
            r for r in self.semantic_memory
            if r.effective_confidence >= settings.memory.min_confidence
        ]
        self._save()

    # ==================== 工作记忆 ====================

    def set_working_context(self, context: str):
        """设置当前工作记忆上下文"""
        self.working_memory = [context]

    def append_working_context(self, item: str):
        """追加工作记忆"""
        self.working_memory.append(item)

    def clear_working_memory(self):
        """清除工作记忆"""
        self.working_memory = []

    # ==================== 辅助方法 ====================

    def _find_similar_rule(self, content: str) -> Optional[SemanticRule]:
        """查找相似的已有规则"""
        content_embedding = self.embedder.encode(content)

        for rule in self.semantic_memory:
            if content_embedding is not None and rule.embedding is not None:
                sim = EmbeddingModel.cosine_similarity(content_embedding, rule.embedding)
                if sim > 0.85:
                    return rule
            else:
                if self._keyword_similarity(content, rule.content) > 0.7:
                    return rule

        return None

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """基于关键词的相似度（嵌入模型不可用时的备选）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)  # Jaccard similarity

    def _evict_episodic(self):
        """淘汰情景记忆（LRU + 低使用率）"""
        # 按使用次数和时间排序，淘汰最不常用的
        self.episodic_memory.sort(
            key=lambda e: (e.usage_count, e.timestamp)
        )
        # 淘汰前10%
        evict_count = max(1, len(self.episodic_memory) // 10)
        self.episodic_memory = self.episodic_memory[evict_count:]

    def _evict_semantic(self):
        """淘汰语义规则（低置信度优先）"""
        self.semantic_memory.sort(key=lambda r: r.effective_confidence)
        self.semantic_memory = self.semantic_memory[1:]  # 淘汰最低的

    def _parse_extracted_rules(self, response: str) -> list[tuple[str, str]]:
        """解析LLM提取的规则"""
        rules = []
        current_rule = ""
        current_applies = "general"

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("RULE:"):
                if current_rule:
                    rules.append((current_rule, current_applies))
                current_rule = line[5:].strip()
                current_applies = "general"
            elif line.startswith("APPLIES_TO:"):
                current_applies = line[11:].strip()

        if current_rule:
            rules.append((current_rule, current_applies))

        return rules

    # ==================== 持久化 ====================

    def _save(self):
        """保存记忆到磁盘"""
        # 保存情景记忆
        episodic_path = self.memory_dir / settings.memory.episodic_file
        with open(episodic_path, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self.episodic_memory],
                f, ensure_ascii=False, indent=2,
            )

        # 保存语义记忆
        semantic_path = self.memory_dir / settings.memory.semantic_file
        with open(semantic_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in self.semantic_memory],
                f, ensure_ascii=False, indent=2,
            )

    def _load(self):
        """从磁盘加载记忆"""
        # 加载情景记忆
        episodic_path = self.memory_dir / settings.memory.episodic_file
        if episodic_path.exists():
            try:
                with open(episodic_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.episodic_memory = [EpisodicEntry.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.episodic_memory)} episodic memories")
            except Exception as e:
                logger.warning(f"Failed to load episodic memory: {e}")

        # 加载语义记忆
        semantic_path = self.memory_dir / settings.memory.semantic_file
        if semantic_path.exists():
            try:
                with open(semantic_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.semantic_memory = [SemanticRule.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.semantic_memory)} semantic rules")
            except Exception as e:
                logger.warning(f"Failed to load semantic memory: {e}")

    # ==================== 统计 ====================

    @property
    def stats(self) -> dict:
        """记忆系统统计"""
        return {
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "working_memory_items": len(self.working_memory),
            "avg_rule_confidence": (
                sum(r.effective_confidence for r in self.semantic_memory)
                / len(self.semantic_memory)
                if self.semantic_memory
                else 0.0
            ),
        }
