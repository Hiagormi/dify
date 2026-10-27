"""Request-only instructions for a single LLM node.

Never put these values in args, variable pools, generate entities, or prompt
execution records. The API scope is copied into the worker context and bound to
one workflow run. Normal/nested runs explicitly clear it, including reused threads.
"""

import contextvars
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Instruction:
    app_id: str
    node_id: str
    text: str = field(repr=False)


@dataclass(frozen=True)
class _BoundInstruction:
    instruction: _Instruction
    run_id: str


_pending: contextvars.ContextVar[_Instruction | None] = contextvars.ContextVar("pending_instruction", default=None)
_bound: contextvars.ContextVar[_BoundInstruction | None] = contextvars.ContextVar("bound_instruction", default=None)


@contextmanager
def instruction_scope(app_id: str, node_id: str, text: str) -> Iterator[None]:
    """Make an instruction available only while the authenticated request runs."""
    token = _pending.set(_Instruction(app_id, node_id, text))
    try:
        yield
    finally:
        _pending.reset(token)


def capture_instruction_context(app_id: str, run_id: str | None) -> contextvars.Context:
    """Capture worker context without allowing nested runs to inherit the task."""
    context = contextvars.copy_context()
    instruction = _pending.get()
    bound = _BoundInstruction(instruction, run_id) if instruction and instruction.app_id == app_id and run_id else None
    context.run(_pending.set, None)
    context.run(_bound.set, bound)
    return context


def transient_instruction_for(node_id: str, run_id: str) -> str | None:
    """Return text only for the designated node in this exact workflow run."""
    bound = _bound.get()
    if bound and bound.run_id == run_id and bound.instruction.node_id == node_id:
        return bound.instruction.text
    return None


def has_pending_instruction() -> bool:
    return _pending.get() is not None


def validate_instruction_target(app_id: str, nodes: Sequence[Mapping[str, object]]) -> None:
    """Reject missing targets and resumable graphs before any message is created."""
    instruction = _pending.get()
    if instruction is None:
        return
    if instruction.app_id != app_id:
        raise ValueError("Transient instruction app mismatch")
    found = False
    for node in nodes:
        data = node.get("data")
        if not isinstance(data, Mapping):
            continue
        if data.get("type") == "human-input":
            raise ValueError("Transient instructions do not support paused workflows")
        if node.get("id") == instruction.node_id and data.get("type") == "llm":
            found = True
    if not found:
        raise ValueError("Transient instruction target must be an LLM node")
