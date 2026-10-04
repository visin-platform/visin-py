"""The ids of the stores Visin can point at, as the server stores and expects them.

Nothing else spells one: a module that writes or reads a store's pointer imports its id from here, so a
second store is one more constant and the registries that name it (checkpoint names, data kinds), not a
search for string literals.
"""

from __future__ import annotations

HUB = "hf"
"""Hugging Face: a model or dataset pinned to a full commit of a Hub repo."""
