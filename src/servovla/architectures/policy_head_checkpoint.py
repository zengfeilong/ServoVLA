from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def _resolve_policy_head(module: nn.Module) -> nn.Module | None:
    policy_head = getattr(module, "policy_head", None)
    if isinstance(policy_head, nn.Module):
        return policy_head

    nested_model = getattr(module, "model", None)
    if isinstance(nested_model, nn.Module):
        nested_policy_head = getattr(nested_model, "policy_head", None)
        if isinstance(nested_policy_head, nn.Module):
            return nested_policy_head
    return None


def _strip_leading_wrappers(key: str) -> str:
    clean = str(key)
    clean = clean.replace("._orig_mod.", ".")
    clean = clean.replace("._orig_mod", "")

    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "policy_head.", "_orig_mod."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                changed = True
    return clean


def _policy_head_key_candidate(key: str) -> tuple[str, bool]:
    clean = str(key)
    clean = clean.replace("._orig_mod.", ".")
    clean = clean.replace("._orig_mod", "")

    while clean.startswith("module."):
        clean = clean[len("module.") :]
    if clean.startswith("model."):
        clean = clean[len("model.") :]
    while clean.startswith("module."):
        clean = clean[len("module.") :]

    has_policy_prefix = clean.startswith("policy_head.")
    return _strip_leading_wrappers(clean), has_policy_prefix


def _target_key_aliases(target: nn.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key in target.state_dict().keys():
        aliases[str(key)] = str(key)
        aliases[_strip_leading_wrappers(str(key))] = str(key)
    return aliases


def normalize_policy_head_state_dict(
    target: nn.Module,
    state_dict: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    aliases = _target_key_aliases(target)
    normalized: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        candidate, has_policy_prefix = _policy_head_key_candidate(str(key))
        if has_policy_prefix:
            normalized[aliases.get(candidate, candidate)] = value
        elif candidate in aliases:
            normalized[aliases[candidate]] = value
    return normalized


def prefixed_policy_head_state_dict(
    module: nn.Module,
    *,
    prefix: str = "policy_head.",
    destination: OrderedDict[str, torch.Tensor] | None = None,
    keep_vars: bool = False,
    clone_to_cpu: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    target = _resolve_policy_head(module) or module
    if destination is None:
        destination = OrderedDict()

    for key, value in target.state_dict(keep_vars=keep_vars).items():
        tensor = value if keep_vars else value.detach()
        if clone_to_cpu:
            tensor = tensor.cpu().clone()
        destination[f"{prefix}{key}"] = tensor
    return destination


def load_policy_head_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
    assign: bool = False,
):
    target = _resolve_policy_head(module) or module
    normalized = normalize_policy_head_state_dict(target, state_dict)
    return target.load_state_dict(normalized, strict=strict, assign=assign)
