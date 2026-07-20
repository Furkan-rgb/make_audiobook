"""The edits-only adaptation contract: address, anchor, judge, apply.

A model is never asked to rewrite a passage — only to list the small changes it
would make to one. This package owns every part of that: how a passage is cut
into addressable sentences and shown to the model (:mod:`spans`), how a quoted
edit is located in the source (:mod:`anchoring`), and how the located edits are
judged and spliced in (:mod:`application`).

Keeping it together is the point. Edit application used to live partly in the
prompt layer and partly in each provider adapter, which meant a second adapter
could apply edits slightly differently and silently produce different text from
identical model output. Providers now parse; this package decides.
"""

from .anchoring import resolve_edit
from .application import apply_edits
from .spans import numbered_view, sentence_spans, strip_label

__all__ = [
    "apply_edits",
    "numbered_view",
    "resolve_edit",
    "sentence_spans",
    "strip_label",
]
