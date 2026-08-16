"""Study 2 — controlled within-family position-bias follow-up (GPT-OSS 20B vs 120B).

Kept in its own package, with its own config, data directory and result
directory, so it can never overwrite or be confused with Study 1 (D-26).
"""

from . import design, metrics

__all__ = ["design", "metrics"]
