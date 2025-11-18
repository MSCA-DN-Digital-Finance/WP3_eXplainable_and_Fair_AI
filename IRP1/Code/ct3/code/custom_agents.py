from __future__ import annotations
import numpy as np
from sklearn.linear_model import SGDClassifier
from tsdm.agents import Agent


class SGDClassifierAgent(Agent):
    """
    SGDClassifierAgent (CT3-ready)

    - Trains online (partial_fit) with a sliding window of past values to predict next movement.
    - Provides a true freeze/eval mode: during evaluation, the model is not updated.
    - Exposes calibrated-like probabilities via log-loss SGD (predict_proba).
    - Supports soft_reset() to clear buffers without losing trained weights.
    - Emits action distributions for CT3 divergence (JS/KL) calculations.

    Conventions:
      * Actions: 0 = "down", 1 = "up"
      * Label at time t: y_t = 1{x_t > x_{t-1}}
      * During warmup (< window_size observations), defaults to action 0 (down)
    """

    def __init__(self, window_size: int = 50, random_state: int = 42, temperature: float = 1.0):
        super().__init__()
        self.window_size = int(window_size)
        self.random_state = int(random_state)
        self.temperature = float(temperature)  # >0; use >1 to soften, <1 to sharpen probs

        # Use log-loss so predict_proba is available
        self.model = SGDClassifier(loss="log_loss", random_state=self.random_state)
        self.has_been_fitted = False

        # Runtime state
        self.frozen = False               # when True: no learning in observe()
        self._obs = []                    # observed_values buffer (list[float])

        # Numerical safety for probabilities
        self._eps = 1e-9

    # ------------- Core TSDM interface -------------

    def observe(self, value: float) -> None:
        """
        Append new observation. If not frozen and enough history is available,
        update the classifier on the (window -> next-move) pair using partial_fit.
        """
        self._obs.append(float(value))
        n = len(self._obs)
        if n < self.window_size + 1:
            return

        # Build single-sample window X_t and label y_t (based on last two values)
        X_train = np.asarray(self._obs[-self.window_size-1:-1], dtype=float).reshape(1, -1)
        y_train = np.array([1 if self._obs[-1] > self._obs[-2] else 0], dtype=int)

        if not self.frozen:
            if not self.has_been_fitted:
                # First call must pass classes
                self.model.partial_fit(X_train, y_train, classes=np.array([0, 1], dtype=int))
                self.has_been_fitted = True
            else:
                self.model.partial_fit(X_train, y_train)

    def place_bet(self) -> int:
        """
        Predicts action for next step based on the most recent window.
        Returns 0 during warmup or if model not fitted yet.
        """
        if len(self._obs) < self.window_size or not self.has_been_fitted:
            return 0
        p1 = self._proba_next_up()
        return int(p1 >= 0.5)

    def reset(self) -> None:
        """
        Full reset: clears buffers and reinitializes the model (training starts from scratch).
        """
        super().reset()
        self.model = SGDClassifier(loss="log_loss", random_state=self.random_state)
        self.has_been_fitted = False
        self.frozen = False
        self._obs = []

    # ------------- CT3 helpers -------------

    def soft_reset(self) -> None:
        """
        Soft reset: clears observation buffer and unfreezes evaluation state,
        but keeps the trained model weights intact.
        Use this before CT3 evaluation runs to avoid re-training.
        """
        self._obs = []

    def freeze(self) -> None:
        """Disable learning (no partial_fit in observe)."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Enable learning (partial_fit resumes in observe)."""
        self.frozen = False

    def action_distribution(self) -> np.ndarray:
        """
        Returns the action distribution [P(0), P(1)] for the next step,
        suitable for divergence metrics in CT3.
        During warmup or before first fit, returns [1.0, 0.0].
        """
        if len(self._obs) < self.window_size or not self.has_been_fitted:
            return np.array([1.0, 0.0], dtype=float)
        p1 = self._proba_next_up()
        p1 = float(np.clip(p1, self._eps, 1.0 - self._eps))
        return np.array([1.0 - p1, p1], dtype=float)

    # ------------- Internals -------------

    def _proba_next_up(self) -> float:
        """
        Predict P(up | last window). Applies temperature scaling if != 1.0.
        """
        X = np.asarray(self._obs[-self.window_size:], dtype=float).reshape(1, -1)
        # predict_proba returns [[P(0), P(1)]]
        proba = self.model.predict_proba(X)[0, 1]  # P(1)
        if self.temperature != 1.0:
            # Temperature scaling on logits: logit = log(p/(1-p))
            p = float(np.clip(proba, self._eps, 1.0 - self._eps))
            logit = np.log(p) - np.log(1.0 - p)
            logit /= self.temperature
            p = 1.0 / (1.0 + np.exp(-logit))
            return p
        return float(proba)
    



class SGDAllocAgent:
    """
    Online-learning allocation agent using SGDClassifier (logistic loss).
    - At time t, on observe(values_t): it *trains* on (features_{t-1} -> argmax rel_change_{t-1->t}),
      if features_{t-1} are available.
    - On place_bet(): it returns predicted class probabilities over assets as allocations.
    - freeze(): disables learning for evaluation.
    - soft_reset(): clears buffers but keeps weights.
    """
    def __init__(self, n_assets: int, random_state: int = 0, alpha: float = 1e-4):
        self.n_assets = int(n_assets)
        self.model = SGDClassifier(loss="log_loss", alpha=alpha, random_state=random_state)
        self.classes_ = np.arange(self.n_assets, dtype=int)
        self._is_fitted = False
        self._frozen = False

        # rolling buffers
        self._last_values: Optional[np.ndarray] = None
        self._last_features: Optional[np.ndarray] = None

    # ---------- utils ----------
    @staticmethod
    def _features_from_values(values: np.ndarray) -> np.ndarray:
        """
        Very simple features: normalized current values and a bias term.
        You can make this richer (moving averages, last returns, etc.).
        """
        v = np.asarray(values, float).reshape(-1)
        s = v.sum()
        norm = v / (s if s != 0 else 1.0)
        return np.concatenate([norm, [1.0]], axis=0)  # add bias

    @staticmethod
    def _argmax_relative_change(curr: np.ndarray, prev: np.ndarray) -> int:
        prev_safe = np.where(prev == 0.0, np.nan, prev)
        rc = curr / prev_safe - 1.0
        rc = np.where(np.isnan(rc), 0.0, rc)
        return int(np.argmax(rc))

    # ---------- API expected by AllocationTask ----------
    def observe(self, values: np.ndarray) -> None:
        """
        Called by AllocationTask *before* each step.
        We use this to train on (last_features -> label computed from last_values -> current values).
        """
        values = np.asarray(values, float).reshape(-1)

        # If we have a previous observation, compute the label and perform partial_fit
        if (not self._frozen) and (self._last_values is not None) and (self._last_features is not None):
            y = self._argmax_relative_change(values, self._last_values)
            X = self._last_features.reshape(1, -1)
            y_arr = np.array([y], dtype=int)
            if not self._is_fitted:
                self.model.partial_fit(X, y_arr, classes=self.classes_)
                self._is_fitted = True
            else:
                self.model.partial_fit(X, y_arr)

        # Prepare features for the upcoming action
        self._last_features = self._features_from_values(values)
        self._last_values = values.copy()

    def place_bet(self) -> np.ndarray:
        """
        Return allocation vector (length n_assets) that sums to 1.
        If not fitted yet, use uniform allocation.
        """
        if self._last_features is None or not self._is_fitted:
            return np.full(self.n_assets, 1.0 / self.n_assets, dtype=float)
        proba = self.model.predict_proba(self._last_features.reshape(1, -1))[0]
        # Ensure simplex (just in case of numerical oddities)
        proba = np.maximum(proba, 0.0)
        s = proba.sum()
        return (proba / s) if s > 0 else np.full(self.n_assets, 1.0 / self.n_assets, dtype=float)

    # ---------- lifecycle helpers ----------
    def freeze(self):
        """Disable learning (used for evaluation)."""
        self._frozen = True

    def soft_reset(self):
        """Clear rolling buffers (keep model weights)."""
        self._last_values = None
        self._last_features = None

