"""
全局配置文件
所有可配置参数集中管理，支持环境变量覆盖
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM推理服务配置"""
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("LLM_API_KEY", "dummy")
    model_name: str = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-9B")
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    # Qwen3.5特有：是否启用思考模式
    enable_thinking: bool = False
    # 请求超时(秒)
    timeout: int = 120
    # 重试次数
    max_retries: int = 3


@dataclass
class SandboxConfig:
    """沙盒浏览器/搜索服务配置"""
    base_url: str = os.getenv("SANDBOX_URL", "http://localhost:8080")
    # 浏览器相关
    browser_timeout: int = 30  # 单次浏览器操作超时(秒)
    max_page_text_length: int = 8000  # 页面文本最大截取长度
    max_concurrent_pages: int = 3  # 最大并发页面数
    # 搜索相关
    search_timeout: int = 15  # 搜索超时(秒)
    max_search_results: int = 5  # 每次搜索返回最大结果数


@dataclass
class AgentConfig:
    """智能体运行配置"""
    # ReAct循环控制
    max_iterations: int = 10  # 最大推理轮次
    total_timeout: int = 300  # 总超时(秒)
    single_turn_timeout: int = 60  # 单轮超时(秒)

    # 死循环检测
    max_repeated_actions: int = 3  # 连续相同action阈值
    loop_detection_window: int = 6  # 循环检测窗口大小

    # 渐进式提醒
    force_answer_remaining: int = 2  # 剩余N轮时强制给答案

    # 反思重试
    max_reflection_retries: int = 2  # 反思后最大重试次数


@dataclass
class MemoryConfig:
    """记忆模块配置"""
    # 存储
    memory_dir: str = "data/memory"
    episodic_file: str = "episodic_memory.json"
    semantic_file: str = "semantic_memory.json"

    # 容量
    max_episodic_entries: int = 500  # 最大情景记忆条目
    max_semantic_rules: int = 100  # 最大语义规则数

    # 检索
    retrieval_top_k: int = 3  # 检索返回top-k
    similarity_threshold: float = 0.5  # 相似度阈值

    # 更新
    confidence_boost: float = 0.1  # 验证成功时置信度提升
    confidence_decay: float = 0.2  # 验证失败时置信度降低
    min_confidence: float = 0.1  # 最低置信度（低于则删除）

    # 嵌入模型
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class EvalConfig:
    """评估配置"""
    # 数据集路径
    simpleqa_path: str = "data/simpleqa"
    wiki2_path: str = "data/2wiki"
    results_dir: str = "data/results"

    # 采样
    eval_sample_size: int = 100  # 评测样本数
    random_seed: int = 42

    # 并发
    max_concurrent_eval: int = 5  # 最大并发评测任务数


@dataclass
class Settings:
    """全局设置聚合"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # 日志
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = "data/logs"

    @classmethod
    def load(cls) -> "Settings":
        """加载配置（支持从环境变量覆盖）"""
        return cls()


# 全局单例
settings = Settings.load()
