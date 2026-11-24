"""Small name-based registry used by encoders, heads, and models.

A component is registered with a decorator::

    @register("encoder", "transformer")
    class TransformerEncoder(nn.Module):
        ...

and built from config::

    build("encoder", "transformer", d_model=384)

The registry keeps the config surface stable: adding a new encoder only
requires a decorator and an import side effect, not a branch in a factory.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[tuple[str, str], Callable[..., Any]] = {}


def register(kind: str, name: str, *, replace: bool = False) -> Callable[[type[T]], type[T]]:
    """Register a class under ``(kind, name)``.

    Args:
        kind: Component family, for example ``"encoder"`` or ``"head"``.
        name: Name used in config files.
        replace: Allow overwriting an existing registration.  Useful for
            downstream projects that intentionally provide a custom variant.
    """

    def decorator(cls: type[T]) -> type[T]:
        key = (kind, name)
        if key in _REGISTRY and not replace:
            raise KeyError(f"duplicate {kind} registration: {name}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a registered component.

    Raises:
        KeyError: If ``name`` is unknown for ``kind``; the message lists the
            available names to make configuration mistakes obvious.
    """
    try:
        cls = _REGISTRY[(kind, name)]
    except KeyError as exc:
        options = ", ".join(sorted(n for k, n in _REGISTRY if k == kind)) or "<none>"
        raise KeyError(f"unknown {kind} '{name}', available: {options}") from exc
    return cls(**kwargs)


def available(kind: str) -> list[str]:
    """Return sorted component names registered for ``kind``."""
    return sorted(name for k, name in _REGISTRY if k == kind)
