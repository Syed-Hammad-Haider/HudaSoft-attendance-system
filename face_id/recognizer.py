"""
Matches a live-detected face embedding against known employee
embeddings loaded from the database.

MATCHING ALGORITHM -- VECTORIZED FOR SCALE: all known embeddings are
stacked into a single (N, 512) numpy matrix at load time. Since
embeddings are L2-normalized (embedder.py), matching a live embedding
against all N stored embeddings is one matrix-vector multiply
(`embedding_matrix @ live_embedding`), executed as a single BLAS call
-- not a Python loop calling np.dot() once per stored embedding.

This distinction matters far more at "thousands of employees" scale
than it does for a handful: a Python-level loop with N iterations
(each with real per-call overhead on top of the actual float math)
scales linearly with real wall-clock cost that becomes measurable
running per-person, periodically, during live monitoring -- with
thousands of employees x 3 angles each, that's thousands of stored
vectors. A single vectorized matrix multiply scales to that size with
no meaningfully different latency (still comfortably sub-millisecond
on CPU), because the underlying BLAS routine is doing the same total
arithmetic in optimized, batched, cache-friendly compiled code instead
of one Python-interpreted call at a time.

WHY COMPARE AGAINST EVERY STORED ANGLE, NOT AN AVERAGED EMBEDDING:
Averaging an employee's front/left/right embeddings into one vector is
tempting (fewer comparisons), but it blurs the very information that
makes multi-angle capture useful -- the averaged vector doesn't
closely resemble ANY real angle well. Comparing against each stored
angle separately and taking the best match per employee is more
compute, but the vectorization above makes that cost negligible even
at thousands of employees, so there's no real reason to give up the
accuracy benefit.

THRESHOLD: a match is only accepted above `min_similarity`. Below that,
the person is reported as UNRECOGNIZED rather than guessing -- for an
attendance system, a false "yes that's Ahmed" is a much worse failure
than correctly saying "I don't know who this is," so the threshold
errs toward caution (see config for the exact value and reasoning).

AMBIGUITY / MARGIN CHECK: on top of the plain threshold, the best
match must also beat the best-scoring match belonging to a DIFFERENT
employee by at least `min_margin_over_runner_up`. This specifically
guards against the failure mode that gets more likely as headcount
grows: two different employees both scoring above the acceptance
threshold, with the "winner" decided by a razor-thin margin that's as
likely to be camera noise as genuine similarity. See
config/settings.py's comment on FaceRecognitionConfig.min_margin_over_runner_up
for the full reasoning.
"""

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from database.employee_repository import StoredEmbedding, load_all_embeddings

logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    employee_id: Optional[int]
    full_name: Optional[str]
    similarity: float  # best match score found, even if below threshold (for logging/debugging)
    matched_angle: Optional[str]
    runner_up_similarity: Optional[float] = None  # best score from a DIFFERENT employee, for debugging ambiguous cases
    rejected_reason: Optional[str] = None          # "below_threshold" | "ambiguous_margin" | None if recognized
    candidate_employee_id: Optional[int] = None    # best match even when rejected (used to confirm weak matches)
    candidate_name: Optional[str] = None

    @property
    def is_recognized(self) -> bool:
        return self.employee_id is not None


class FaceRecognizer:
    def __init__(self, min_similarity: float = 0.45, min_margin_over_runner_up: float = 0.03):
        self._min_similarity = min_similarity
        self._min_margin_over_runner_up = min_margin_over_runner_up

        # NEW (this session): a lock protecting the four parallel
        # arrays below. Previously safe by construction, because
        # nothing ever called reload_known_faces() from a different
        # thread than the one calling recognize() -- main.py and
        # web_dashboard.py's detection loop both only ever reloaded
        # once, at startup, before the loop began. That assumption
        # broke the moment browser-based employee registration (see
        # web_dashboard.py's /api/registration/finalize) needed to
        # reload embeddings into the SAME, already-running recognizer
        # instance the detection loop thread is actively calling
        # recognize() on -- without a lock, reload_known_faces()
        # reassigns these four attributes one at a time (not
        # atomically), so a recognize() call on the detection thread
        # could read a genuinely inconsistent mix of old and new state
        # mid-reload (e.g. a new, longer _embedding_matrix paired with
        # the OLD, shorter _employee_ids -- an index that's valid for
        # one but out of range for the other). The lock makes both the
        # write (reload) and the read (recognize) atomic with respect
        # to each other; the matching algorithm/logic itself is
        # completely unchanged.
        self._lock = threading.Lock()

        # Parallel arrays/lists, one entry per stored embedding, built
        # fresh on every reload_known_faces() call. Kept as numpy
        # arrays (not a list of StoredEmbedding objects) specifically
        # so recognize() can do the vectorized matrix-vector multiply
        # described in this module's docstring, and vectorized boolean
        # masking for the runner-up/margin computation.
        self._embedding_matrix: Optional[np.ndarray] = None  # shape (N, 512), or None if nothing loaded
        self._employee_ids: Optional[np.ndarray] = None      # shape (N,), dtype int
        self._full_names: List[str] = []                     # length N, indexed the same as the arrays above
        self._angle_labels: List[str] = []                   # length N

    def reload_known_faces(self, db_path=None) -> int:
        """
        Re-read all employee embeddings from the database into memory
        and rebuild the vectorized matching structures. Call this at
        startup, and again any time a new employee is registered while
        the system is running. Returns the count loaded, for a clear
        startup log line.
        """
        stored: List[StoredEmbedding] = load_all_embeddings(db_path) if db_path else load_all_embeddings()

        if not stored:
            with self._lock:
                self._embedding_matrix = None
                self._employee_ids = None
                self._full_names = []
                self._angle_labels = []
            logger.info("Loaded 0 face embeddings for recognition (no employees registered yet).")
            return 0

        # Built into LOCAL variables first, then swapped in together
        # under the lock -- see the lock's docstring above for why the
        # swap itself (not just each individual assignment) needs to
        # be atomic with respect to recognize().
        embedding_matrix = np.stack([s.embedding for s in stored]).astype(np.float32)
        employee_ids = np.array([s.employee_id for s in stored], dtype=np.int64)
        full_names = [s.full_name for s in stored]
        angle_labels = [s.angle_label for s in stored]

        with self._lock:
            self._embedding_matrix = embedding_matrix
            self._employee_ids = employee_ids
            self._full_names = full_names
            self._angle_labels = angle_labels

        logger.info(
            "Loaded %d face embeddings for recognition (%d unique employees).",
            len(stored), len(set(employee_ids.tolist())),
        )
        return len(stored)

    def recognize(self, live_embedding: np.ndarray) -> RecognitionResult:
        """Find the best-matching known employee for a live face embedding."""
        # Snapshot under the lock (see __init__'s comment) so this read
        # can never observe a partially-updated set of the four
        # parallel arrays -- everything after this point works from
        # local copies, not self._... directly.
        with self._lock:
            embedding_matrix = self._embedding_matrix
            employee_ids = self._employee_ids
            full_names = self._full_names
            angle_labels = self._angle_labels

        if embedding_matrix is None:
            return RecognitionResult(employee_id=None, full_name=None, similarity=0.0, matched_angle=None)

        # Single vectorized matrix-vector multiply against every stored
        # embedding -- see module docstring for why this replaces a
        # Python-level loop.
        similarities = embedding_matrix @ live_embedding.astype(np.float32)

        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_employee_id = int(employee_ids[best_idx])

        # Best score among entries belonging to a DIFFERENT employee --
        # this is the ambiguity/margin check. Vectorized boolean mask,
        # not a second loop.
        different_employee_mask = employee_ids != best_employee_id
        if np.any(different_employee_mask):
            runner_up_similarity = float(similarities[different_employee_mask].max())
        else:
            runner_up_similarity = None  # only one employee enrolled -- no ambiguity is possible

        if best_similarity < self._min_similarity:
            return RecognitionResult(
                employee_id=None, full_name=None, similarity=best_similarity,
                matched_angle=angle_labels[best_idx],
                runner_up_similarity=runner_up_similarity, rejected_reason="below_threshold",
                candidate_employee_id=best_employee_id, candidate_name=full_names[best_idx],
            )

        if runner_up_similarity is not None and (best_similarity - runner_up_similarity) < self._min_margin_over_runner_up:
            logger.info(
                "Ambiguous match rejected: best=%.3f (employee_id=%d) vs runner-up=%.3f (different employee), "
                "margin=%.3f < required %.3f.",
                best_similarity, best_employee_id, runner_up_similarity,
                best_similarity - runner_up_similarity, self._min_margin_over_runner_up,
            )
            return RecognitionResult(
                employee_id=None, full_name=None, similarity=best_similarity,
                matched_angle=angle_labels[best_idx],
                runner_up_similarity=runner_up_similarity, rejected_reason="ambiguous_margin",
                candidate_employee_id=best_employee_id, candidate_name=full_names[best_idx],
            )

        return RecognitionResult(
            employee_id=best_employee_id,
            full_name=full_names[best_idx],
            similarity=best_similarity,
            matched_angle=angle_labels[best_idx],
            runner_up_similarity=runner_up_similarity,
            rejected_reason=None,
        )
