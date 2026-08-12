"""Observability for the agent loop: per-step traces and a run summary.

See trace.py. Wired through the loop's existing on_step callback, so nothing in
the loop changes to switch it on.
"""

from .trace import Tracer

__all__ = ["Tracer"]
