import json

import pytest
from pydantic import ValidationError

from arkive_rag.contracts import LivenessResponse


def test_liveness_serializes_to_ok_status() -> None:
    response = LivenessResponse(status="ok")

    assert json.loads(response.model_dump_json()) == {"status": "ok"}


def test_liveness_requires_status() -> None:
    with pytest.raises(ValidationError):
        LivenessResponse.model_validate({})


@pytest.mark.parametrize("status", ["degraded", True, 1, b"ok"])
def test_liveness_rejects_status_other_than_ok_string(status: object) -> None:
    with pytest.raises(ValidationError):
        LivenessResponse.model_validate({"status": status})


def test_liveness_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LivenessResponse.model_validate({"status": "ok", "detail": "healthy"})


def test_liveness_is_immutable() -> None:
    response = LivenessResponse(status="ok")

    with pytest.raises(ValidationError):
        response.status = "ok"
