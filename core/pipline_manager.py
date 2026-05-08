"""Branch-aware pipeline manager.

Each image-transforming operation produces a HistoryEntry (image + label + params).
Entries are grouped into Branches.  When the user applies a new operation while
NOT at the tip of the current branch (i.e., after an undo), a new Branch is
forked automatically instead of truncating the existing history — so Redo always
remains available on the original branch.

Public signals
--------------
image_updated(object)   ndarray of the image that is now current
error_occurred(str)     error message from apply_operation
pipeline_changed()      structure of branches/history changed; panel should refresh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .image_state import ImageState


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class HistoryEntry:
    image: np.ndarray
    label: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Branch:
    branch_id: int
    name: str
    entries: List[HistoryEntry]
    current_idx: int = 0
    parent_branch_id: Optional[int] = None
    fork_step: Optional[int] = None     # step index in parent at which this branch was created


# ── Manager ───────────────────────────────────────────────────────────────────

class PipelineManager(QObject):
    image_updated   = pyqtSignal(object)
    error_occurred  = pyqtSignal(str)
    pipeline_changed = pyqtSignal()

    def __init__(self, state: ImageState):
        super().__init__()
        self.state = state
        self.branches: List[Branch] = []
        self.current_branch_idx: int = 0
        self._next_branch_id: int = 0

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def current_branch(self) -> Optional[Branch]:
        if not self.branches:
            return None
        return self.branches[self.current_branch_idx]

    # ── public API ────────────────────────────────────────────────────────────

    def load_initial(self, img: np.ndarray, metadata: dict, path: str):
        self.state.original = img.copy()
        self.state.current  = img.copy()
        self.state.metadata = metadata
        self.state.file_path = path

        self._next_branch_id = 0
        entry = HistoryEntry(image=img.copy(), label="Load",
                             params={"file": path})
        main = Branch(branch_id=0, name="Main",
                      entries=[entry], current_idx=0)
        self._next_branch_id = 1
        self.branches = [main]
        self.current_branch_idx = 0

        self.image_updated.emit(img)
        self.pipeline_changed.emit()

    def apply_result(self, result: np.ndarray,
                     label: str = "Operation",
                     params: Dict[str, Any] = None):
        if result is None:
            return
        if params is None:
            params = {}

        branch = self.current_branch
        if branch is None:
            return

        tip = len(branch.entries) - 1
        if branch.current_idx < tip:
            # Not at tip: fork instead of truncating so redo stays valid
            self._fork_and_apply(result, label, params)
            return

        entry = HistoryEntry(image=result.copy(), label=label, params=params)
        branch.entries.append(entry)
        branch.current_idx += 1
        self.state.current = result.copy()
        self.image_updated.emit(result)
        self.pipeline_changed.emit()

    def undo(self):
        branch = self.current_branch
        if branch is None or branch.current_idx <= 0:
            return
        branch.current_idx -= 1
        img = branch.entries[branch.current_idx].image
        self.state.current = img.copy()
        self.image_updated.emit(self.state.current)
        self.pipeline_changed.emit()

    def redo(self):
        branch = self.current_branch
        if branch is None:
            return
        if branch.current_idx < len(branch.entries) - 1:
            branch.current_idx += 1
            img = branch.entries[branch.current_idx].image
            self.state.current = img.copy()
            self.image_updated.emit(self.state.current)
            self.pipeline_changed.emit()

    def reset(self):
        """Reset current branch to its first entry (fork-point or original load)."""
        branch = self.current_branch
        if branch is None:
            return
        branch.current_idx = 0
        img = branch.entries[0].image
        self.state.current  = img.copy()
        self.state.original = img.copy()
        self.image_updated.emit(self.state.current)
        self.pipeline_changed.emit()

    def jump_to(self, branch_idx: int, step_idx: int):
        """Navigate to an arbitrary step on any branch."""
        if branch_idx < 0 or branch_idx >= len(self.branches):
            return
        branch = self.branches[branch_idx]
        if step_idx < 0 or step_idx >= len(branch.entries):
            return
        self.current_branch_idx = branch_idx
        branch.current_idx = step_idx
        img = branch.entries[step_idx].image
        self.state.current  = img.copy()
        self.state.original = branch.entries[0].image.copy()
        self.image_updated.emit(self.state.current)
        self.pipeline_changed.emit()

    def switch_branch(self, branch_idx: int):
        """Switch to a different branch at its last-known current position."""
        if branch_idx < 0 or branch_idx >= len(self.branches):
            return
        self.current_branch_idx = branch_idx
        branch = self.branches[branch_idx]
        img = branch.entries[branch.current_idx].image
        self.state.current  = img.copy()
        self.state.original = branch.entries[0].image.copy()
        self.image_updated.emit(self.state.current)
        self.pipeline_changed.emit()

    # ── legacy (not called from UI) ───────────────────────────────────────────

    def apply_operation(self, func, *args, **kwargs):
        try:
            result = func(self.state.current, *args, **kwargs)
            if result is None:
                return
            self.apply_result(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    # ── private helpers ───────────────────────────────────────────────────────

    def _fork_and_apply(self, result: np.ndarray,
                        label: str, params: Dict[str, Any]):
        parent = self.current_branch
        fork_at = parent.current_idx

        # Copy history up to and including the fork point into the new branch
        base = [
            HistoryEntry(image=e.image.copy(), label=e.label, params=dict(e.params))
            for e in parent.entries[: fork_at + 1]
        ]
        base.append(HistoryEntry(image=result.copy(), label=label, params=params))

        new_branch = Branch(
            branch_id=self._next_branch_id,
            name=f"Branch {self._next_branch_id}",
            entries=base,
            current_idx=len(base) - 1,
            parent_branch_id=parent.branch_id,
            fork_step=fork_at,
        )
        self._next_branch_id += 1
        self.branches.append(new_branch)
        self.current_branch_idx = len(self.branches) - 1

        self.state.current  = result.copy()
        self.state.original = parent.entries[fork_at].image.copy()

        self.image_updated.emit(result)
        self.pipeline_changed.emit()
