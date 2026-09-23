"""Per-kernel JIT compile caches.

Replaces the module-local ``_compile_cache = {}`` dict that each kernel host
wrapper carried.  Reuses ``flash_attn.cute.cache_utils`` so we inherit the same
dict-like protocol (``__contains__`` / ``__getitem__`` / ``__setitem__`` /
``clear``) that ``_flash_attn_fwd`` uses upstream, and attach one named cache per
kernel exactly as it does.

Persistent (on-disk) caching stays OFF.  ``get_jit_cache`` only consults disk
when ``FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1``, and its invalidation
fingerprint hashes the ``.py`` files under ``flash_attn/cute/`` ONLY -- editing
an AdaSplash kernel would not invalidate a persisted entry, so enabling it today
would serve stale cubins.  Fixing that means extending the fingerprint to cover
this package; until then, leave the env var unset.
"""

from flash_attn.cute.cache_utils import get_jit_cache


__all__ = ["get_jit_cache"]
