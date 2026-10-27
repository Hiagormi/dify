from concurrent.futures import ThreadPoolExecutor

import pytest

from dify_graph.context.transient_instruction import (
    capture_instruction_context,
    instruction_scope,
    transient_instruction_for,
)


def test_instruction_is_bound_to_app_run_and_target_node():
    with instruction_scope("app-a", "reply", "private care instruction"):
        context = capture_instruction_context("app-a", "run-a")
        other_app = capture_instruction_context("app-b", "run-a")
    assert context.run(transient_instruction_for, "reply", "run-a") == "private care instruction"
    assert context.run(transient_instruction_for, "summary", "run-a") is None
    assert context.run(transient_instruction_for, "reply", "run-b") is None
    assert other_app.run(transient_instruction_for, "reply", "run-a") is None
    assert transient_instruction_for("reply", "run-a") is None


def test_normal_request_clears_context_in_reused_worker():
    with instruction_scope("app", "reply", "private"):
        care = capture_instruction_context("app", "care-run")
    ordinary = capture_instruction_context("app", "chat-run")
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(care.run, transient_instruction_for, "reply", "care-run").result() == "private"
        assert pool.submit(ordinary.run, transient_instruction_for, "reply", "chat-run").result() is None


def test_concurrent_requests_do_not_share_instructions():
    with instruction_scope("app", "reply", "first"):
        first = capture_instruction_context("app", "run-1")
    with instruction_scope("app", "reply", "second"):
        second = capture_instruction_context("app", "run-2")
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.run, transient_instruction_for, "reply", "run-1")
        b = pool.submit(second.run, transient_instruction_for, "reply", "run-2")
        assert (a.result(), b.result()) == ("first", "second")


def test_failed_request_resets_scope_and_nested_runs_do_not_inherit_pending_text():
    with instruction_scope("app", "reply", "private"):
        captured = capture_instruction_context("app", "run")

    def fail_request():
        with instruction_scope("app", "reply", "private"):
            raise RuntimeError("request failed")

    with pytest.raises(RuntimeError):
        fail_request()
    ordinary = capture_instruction_context("app", "next")
    assert ordinary.run(transient_instruction_for, "reply", "next") is None
    nested = captured.run(capture_instruction_context, "app", "nested")
    assert nested.run(transient_instruction_for, "reply", "nested") is None


def test_target_validation_rejects_missing_non_llm_and_pause_before_generation():
    from dify_graph.context.transient_instruction import validate_instruction_target

    with instruction_scope("app", "reply", "private"):
        validate_instruction_target("app", [{"id": "reply", "data": {"type": "llm"}}])
        for nodes in [
            [],
            [{"id": "reply", "data": {"type": "code"}}],
            [{"id": "reply", "data": {"type": "llm"}}, {"id": "pause", "data": {"type": "human-input"}}],
        ]:
            with pytest.raises(ValueError):
                validate_instruction_target("app", nodes)
        with pytest.raises(ValueError):
            validate_instruction_target("other-app", [{"id": "reply", "data": {"type": "llm"}}])


def test_payload_excludes_hidden_text_and_rejects_streaming():
    from pydantic import ValidationError

    from controllers.service_api.app.completion import TransientChatRequestPayload

    request = {
        "inputs": {},
        "query": "authorized record facts",
        "transient_context": {"node_id": "reply", "text": "private"},
    }
    payload = TransientChatRequestPayload.model_validate(request)
    persisted_args = payload.model_dump(exclude_none=True, exclude={"transient_context"})
    assert "transient_context" not in persisted_args
    assert persisted_args["query"] == "authorized record facts"
    with pytest.raises(ValidationError):
        TransientChatRequestPayload.model_validate(dict(request, response_mode="streaming"))
    with pytest.raises(ValidationError):
        TransientChatRequestPayload.model_validate(dict(request, auto_generate_name=True))
