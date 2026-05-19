"""快速验证所有模块导入"""
import sys
sys.path.insert(0, '.')

from config.settings import settings
print(f'[OK] Config loaded: LLM={settings.llm.base_url}')

from config.prompts import Prompts
print(f'[OK] Prompts loaded')

from tools.base import BaseTool, ToolResult
from tools.registry import ToolRegistry
print(f'[OK] Tools layer')

from core.llm import LLMClient
from core.harness import Harness, HarnessStatus
print(f'[OK] Core layer')

from modules.evaluator import Evaluator
from modules.reflection import ReflectionModule, FailureCategory
print(f'[OK] Modules layer')

from evaluation.metrics import MetricsCalculator
from evaluation.datasets import DatasetLoader
print(f'[OK] Evaluation layer')

# Test metrics
m = MetricsCalculator.compute_single('United States', 'united states')
print(f'[OK] Metrics: EM={m.exact_match}, F1={m.f1_score}')

# Test tool registry
reg = ToolRegistry()
from tools.search import SearchTool
from tools.browser import BrowserNavigateTool
reg.register(SearchTool())
reg.register(BrowserNavigateTool())
print(f'[OK] Tools registered: {reg.tool_names}')

# Test harness
h = Harness()
h.start()
assert h.next_iteration() == True
assert h.current_iteration == 1
h.record_action("web_search", {"query": "test"})
print(f'[OK] Harness works: iter={h.current_iteration}')

# Test ToolResult
r = ToolResult.success("hello world", raw_data={"key": "val"})
assert r.is_success
print(f'[OK] ToolResult works')

print()
print('=' * 50)
print('  ALL MODULES VERIFIED SUCCESSFULLY!')
print('=' * 50)
