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





class UCBBanditAgent(Agent):
    """
    UCBBanditAgent (CT3-ready, non-contextual)

    - Two actions: 0 = "down", 1 = "up"
    - Non-contextual UCB-style bandit: keeps empirical mean reward per arm and
      uses an upper-confidence bound to choose actions.
    - Reward is defined from directional accuracy:
        * +1 if last action matches sign(y_t - y_{t-1})
        * -1 otherwise (0 can optionally be treated as neutral)
    - Learning happens in observe(): when a new value arrives, we compute the
      reward for the *previous* action and update the bandit's statistics.
    - Provides freeze()/soft_reset() to integrate cleanly with CT3:
        * freeze()   -> stop updating statistics (no learning in observe)
        * soft_reset() -> clear observation buffer and last_action, but keep
                          learned statistics (means + counts) intact.
    - action_distribution() returns a probability vector [P(0), P(1)] based on
      a softmax over empirical mean rewards (with temperature scaling).

    Conventions:
      * Actions: 0 = "down", 1 = "up"
      * At time t, label is based on Δy_t = y_t - y_{t-1}
      * During very early stages (no past reward), defaults to action 0 (down)
    """

    def __init__(
        self,
        exploration_c: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.exploration_c = float(exploration_c)
        self.temperature = float(temperature)  # >0; >1 soften, <1 sharpen

        # Bandit statistics
        self._counts = np.zeros(2, dtype=float)   # n_a
        self._values = np.zeros(2, dtype=float)   # empirical mean rewards μ̂_a

        # Runtime state
        self._obs: list[float] = []               # observed values y_t
        self._last_action: int | None = None
        self.frozen: bool = False                 # if True, no updates in observe

        # Numerical safety for probabilities
        self._eps = 1e-9

    # ------------- Core TSDM interface -------------

    def observe(self, value: float) -> None:
        """
        Append new observation. If there is a previous observation and a last
        action, compute the reward for that action based on the direction of
        movement and update the bandit's statistics (unless frozen).
        """
        self._obs.append(float(value))

        # Need at least two observations to define Δy_t
        if len(self._obs) < 2:
            return

        if self._last_action is None:
            return

        # Compute label / reward for the *previous* action
        dy = self._obs[-1] - self._obs[-2]

        if dy > 0:
            true_dir = 1  # up
        elif dy < 0:
            true_dir = 0  # down
        else:
            # Flat move: you can choose to treat this as neutral
            reward = 0.0
            if not self.frozen:
                self._update_bandit(self._last_action, reward)
            return

        reward = 1.0 if self._last_action == true_dir else -1.0

        if not self.frozen:
            self._update_bandit(self._last_action, reward)

    def place_bet(self) -> int:
        """
        Choose an action according to UCB1 on the empirical rewards.

        Early stage behavior:
          - If no actions have been tried yet, default to 0 ("down").
          - If exactly one action has been tried, try the other one at least once.
        """
        # If no history at all, pick 0 by convention
        if self._counts.sum() == 0:
            action = 0
        # Ensure each arm is tried at least once
        elif self._counts[0] == 0:
            action = 0
        elif self._counts[1] == 0:
            action = 1
        else:
            # Standard UCB1
            t = self._counts.sum()
            c = self.exploration_c
            # Avoid div-by-zero (we know counts > 0 here)
            ucb = self._values + c * np.sqrt(2.0 * np.log(t) / self._counts)
            action = int(np.argmax(ucb))

        self._last_action = action
        return action

    def reset(self) -> None:
        """
        Full reset: clears buffers and reinitializes bandit statistics.
        Use this if you want to re-train from scratch.
        """
        super().reset()
        self._counts[:] = 0.0
        self._values[:] = 0.0
        self._obs = []
        self._last_action = None
        self.frozen = False

    # ------------- CT3 helpers -------------

    def soft_reset(self) -> None:
        """
        Soft reset: clears observation buffer and last_action,
        but keeps the learned bandit statistics (counts + values) intact.
        This is the right choice before CT3 evaluation runs.
        """
        self._obs = []
        self._last_action = None

    def freeze(self) -> None:
        """
        Disable learning: no updates to bandit statistics in observe().
        """
        self.frozen = True

    def unfreeze(self) -> None:
        """
        Enable learning: updates in observe() are applied again.
        """
        self.frozen = False

    def action_distribution(self) -> np.ndarray:
        """
        Returns a probability vector [P(0), P(1)] representing the policy's
        tendency over actions, based on a softmax over the empirical mean
        rewards (with temperature scaling).

        During the very early phase (no rewards yet), returns [1.0, 0.0].
        """
        # No information yet: default deterministic "down"
        if self._counts.sum() == 0:
            return np.array([1.0, 0.0], dtype=float)

        # Softmax over empirical means
        vals = self._values.astype(float)
        # Temperature scaling
        if self.temperature != 1.0:
            vals = vals / max(self.temperature, self._eps)

        # Stable softmax
        max_v = np.max(vals)
        exp_v = np.exp(vals - max_v)
        probs = exp_v / (np.sum(exp_v) + self._eps)

        # Numerical safety
        probs = np.clip(probs, self._eps, 1.0 - self._eps)
        probs = probs / probs.sum()

        return probs

    # ------------- Internals -------------

    def _update_bandit(self, action: int, reward: float) -> None:
        """
        Incremental update of empirical mean reward for the given action.
        """
        a = int(action)
        n = self._counts[a]
        # New count
        self._counts[a] = n + 1.0
        # Incremental mean update: μ_new = μ_old + (r - μ_old) / (n + 1)
        self._values[a] = self._values[a] + (reward - self._values[a]) / (n + 1.0)
