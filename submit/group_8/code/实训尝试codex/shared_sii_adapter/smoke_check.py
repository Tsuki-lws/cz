from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

from .submit_writer import write_submission
from .tools import TOOL_FN_MAP, TOOLS_SCHEMA
from .types import AgentRunResult


def main() -> None:
    print(f"tools={len(TOOLS_SCHEMA)} names={sorted(TOOL_FN_MAP)}")
    for name in [
        "track_a_distill.agent",
        "track_b_mia.agent",
        "track_c_ahe.agent",
        "track_d_ttl.agent",
        "track_e_engineering.agent",
        "track_f_wke.agent",
    ]:
        module = importlib.import_module(name)
        assert hasattr(module, "run_one"), name
        print(f"track={name} ok")
    with tempfile.TemporaryDirectory() as tmp:
        result = AgentRunResult(index="0", instruction="demo", pred="demo answer")
        paths = write_submission([result], Path(tmp), "demo")
        for path in paths.values():
            assert Path(path).exists(), path
        print(f"submission_writer ok: {paths}")


if __name__ == "__main__":
    main()
