from __future__ import annotations

from shared_sii_adapter.compliance import assert_teacher_model_size_le_32b


def validate_teacher_model(model_name: str) -> None:
    assert_teacher_model_size_le_32b(model_name)

