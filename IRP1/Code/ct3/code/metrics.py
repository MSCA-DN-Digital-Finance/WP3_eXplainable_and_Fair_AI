from typing import List, Dict, Any, Callable, Optional
import numpy as np

MetricFn = Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, float]]

def metric_prediction(warmup: int = 0) -> MetricFn:
    def _metric(ref_log, mu_log):
        # actions
        a_ref = np.array([int(r["bet"]) for r in ref_log], int)
        a_mu  = np.array([int(r["bet"]) for r in mu_log],  int)
        n = min(len(a_ref), len(a_mu))
        if n <= warmup:
            return {"D_ham": np.nan, "delta_p": np.nan, "p_up": np.nan}
        a_ref, a_mu = a_ref[warmup:n], a_mu[warmup:n]
        D_ham  = float(np.mean(a_mu != a_ref))
        p_up_mu, p_up_ref = float(np.mean(a_mu == 1)), float(np.mean(a_ref == 1))
        return {"D_ham": D_ham, "delta_p": p_up_mu - p_up_ref, "p_up": p_up_mu}
    return _metric



import numpy as np
from typing import List, Dict, Any

def metric_allocation(trend_idx: int = 1, warmup: int = 0):
    """
    CT3 metrics for AllocationTask: one divergence, one response metric.

    Returns dict:
        {
            "alloc_L1": <float>,   # divergence magnitude
            "alloc_response": <float>,  # mechanism-aligned directional response
        }
    """
    def _metric(ref_log: List[Dict[str, Any]], mu_log: List[Dict[str, Any]]):
        n = min(len(ref_log), len(mu_log))
        if n <= warmup:
            return {"alloc_L1": np.nan, "alloc_response": np.nan}

        ref_log, mu_log = ref_log[warmup:n], mu_log[warmup:n]
        alloc_ref = np.array([r["allocations"] for r in ref_log])
        alloc_mu  = np.array([r["allocations"] for r in mu_log])

        # divergence: mean L1 distance
        D_alloc = float(np.mean(np.sum(np.abs(alloc_mu - alloc_ref), axis=1)))

        # response: signed mean difference of trend allocation
        delta_alloc = float(np.mean(alloc_mu[:, trend_idx] - alloc_ref[:, trend_idx]))

        return {"alloc_L1": D_alloc, "alloc_response": delta_alloc}
    return _metric
