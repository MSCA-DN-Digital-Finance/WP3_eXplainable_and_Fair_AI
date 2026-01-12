import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loaders import load_parquet, load_pickle

sector_colors = {
    'Communication Services': "#FF0015",      # Red
    'Consumer Discretionary': '#F77F00',      # Orange
    'Consumer Staples': "#FBD500",            # Yellow
    'Energy': "#947CFF",                      # Mint Green
    'Financials': "#006AFF",                  # Blue
    'Health Care': "#00FF6A",                # Dark Blue
    'Industrials': "#9F5D34",                 # Brown
    'Information Technology': "#8C00FF",      # Purple
    'Materials': "#FF59AC",                   # Pink
    'Real Estate': "#31833D",                 # Burgundy
    'Utilities':"#3F3FF7",               
}

def title_with_info_centered(title, tooltip_text):
    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:8px;
            font-size:18px;
            font-weight:600;
            margin-bottom:6px;
        ">
            <span>{title}</span>
            <span class="tooltip-icon">i
                <span class="tooltip-text">{tooltip_text}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_aggregate_frontier_percent(selected_period_raw):
    title_with_info_centered(
    "Tracking Error vs Carbon Reduction (%)",
    (
        "This plot shows the efficient frontier between tracking error and achieved "
        "carbon reduction. Each point corresponds to an optimal portfolio constructed "
        "under a different tracking-error constraint. Steeper curves indicate that "
        "meaningful decarbonization can be achieved at lower tracking error."
    )
)
    sector_frontiers = load_pickle(f"data/optimal_portfolios/optimal_portfolios_all_te_{selected_period_raw}.pkl")

    fig = go.Figure()

    for sector, d in sector_frontiers.items():
        te = np.array(d["tracking_errors"])
        cr = np.array(d["carbon_reductions"])   # already in %

        color = sector_colors.get(sector, "#999")

        fig.add_trace(
        go.Scatter(
            x=te,
            y=cr,
            mode="lines",
            name=sector,
            line=dict(color=color, width=2),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Tracking Error: %{x:.2f} bps<br>"
                "Carbon Reduction: %{y:.2f}%<br>"
                "<extra></extra>"
            )
        )
    )



    fig.update_layout(
    xaxis_title="Tracking Error (bps)",
    yaxis_title="Carbon Reduction (%)",
    template="plotly_dark",
        height=650,
        margin=dict(
            l=70,
            r=40,
            t=10,      # 🔑 reduce top margin since title is now outside
            b=120
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )

    
    st.plotly_chart(fig, use_container_width=True)

def plot_aggregate_frontier_absolute(selected_period_raw):
    title_with_info_centered(
    "Tracking Error vs Absolute Portfolio Carbon Intensity",
    (
        "This plot shows how the portfolio’s absolute (weighted) carbon intensity "
        "evolves as the tracking-error constraint is relaxed. Lower values indicate "
        "cleaner portfolios in absolute terms, while the slope reflects how efficiently "
        "carbon intensity can be reduced as tracking error increases."
    )
)

    sector_frontiers = load_pickle(f"data/optimal_portfolios/optimal_portfolios_all_te_{selected_period_raw}.pkl")

    fig = go.Figure()

    for sector, d in sector_frontiers.items():
        te = np.array(d["tracking_errors"])
        weights_by_te = d["weights_by_te"]
        carbon_intensity = np.array(d["carbon_intensity"])

        abs_carbon = [np.dot(w, carbon_intensity) for w in weights_by_te]

        color = sector_colors.get(sector, "#999")

        fig.add_trace(
    go.Scatter(
        x=te,
        y=abs_carbon,
        mode="lines",
        name=sector,
        line=dict(color=color, width=2),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Tracking Error: %{x:.2f} bps<br>"
            "Weighted Carbon Intensity: %{y:.2f}<br>"
            "<extra></extra>"
        )
    )
)


    fig.update_layout(
        xaxis_title="Tracking Error (bps)",
        yaxis_title="Weighted Carbon Intensity",
        template="plotly_dark",
        height=650,
        margin=dict(
            l=70,
            r=40,
            t=10,      # 🔑 reduce top margin since title is now outside
            b=120
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )


    st.plotly_chart(fig, use_container_width=True)

def make_sector_radar(sector_name, values, color, dri_score):
    labels = ["Room for Maneuver", "Flexibility", "Sensitivity", "Robustness"]

    r = [
        values["Room for Maneuver"],
        values["Flexibility"],
        values["Sensitivity"],
        values["Robustness"]
    ]
    r = r + [r[0]]  # close polygon

    fig = go.Figure()
    fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=labels + [labels[0]],
                fill="toself",
                name=sector_name,
                line=dict(color=color, width=3),
                opacity=0.85,
                showlegend=False,
                hovertemplate=(
                    "<b>%{theta}</b><br>"
                    "Score: %{r:.2f}<br>"
                    "<extra></extra>"
                )
            )
        )

    # Sector label + DRI box
    fig.add_annotation(
    x=1.02,            # 🔹 move further right (try 1.03–1.08)
    y=0.15,            # 🔹 vertical placement
    xref="paper",
    yref="paper",
    text=(
        f"<b>{sector_name}</b><br>"
        f"<span style='font-size:11px'>DRI = {dri_score:.2f}</span>"

    ),
    showarrow=False,
    align="left",
    font=dict(
        size=12,
        color="white"
    ),
    bgcolor="rgba(0,0,0,0.6)",
    bordercolor="white",   # ✅ white border
    borderwidth=1.5,
    borderpad=6
)

    fig.update_layout(
    polar=dict(
        domain=dict(x=[0.08, 0.92], y=[0.08, 0.92]),
        radialaxis=dict(
            range=[0, 1],
            showticklabels=False,
            ticks=""
        ),
        angularaxis=dict(
            tickfont=dict(size=10)
        )
    ),
    width=300,
    height=280,
    margin=dict(l=10, r=10, t=14, b=10),
    template="plotly_dark",
    showlegend=False
)



    return fig

def title_with_info(title, tooltip_text):
    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            align-items:center;
            gap:6px;
            font-size:32px;
            font-weight:600;
            margin-bottom:4px;
        ">
            <span>{title}</span>
            <span class="tooltip-icon">i
                <span class="tooltip-text">{tooltip_text}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_aggregate_radars_sorted():
    title_with_info(
    "Decarbonization Readiness Index (DRI)",
    (
        "The Decarbonization Readiness Index (DRI) summarizes a sector’s ability "
        "to decarbonize effectively and robustly under portfolio constraints. "
        "It combines four equally weighted dimensions—Room for Maneuver, Flexibility, "
        "Sensitivity, and Robustness—each scaled to [0,1]. Higher values indicate "
        "greater strategic capacity to achieve and sustain decarbonization."
    )
)
    # Load normalized dimension data

  
    dri_data = load_pickle("data/DRI/dri_radar_data.pkl") # sector → dict of 4 normalized dimensions

    # Compute DRI score (mean of dimensions)
    dri_scores = {
        sector: np.mean(list(values.values()))
        for sector, values in dri_data.items()
    }

    # Sort sectors by DRI
    ordered_sectors = sorted(
        dri_scores.keys(),
        key=lambda s: dri_scores[s],
        reverse=True
    )

    n_cols = 3
    cols = st.columns(n_cols)
    col_idx = 0

    for sector in ordered_sectors:
        fig = make_sector_radar(
            sector_name=sector,
            values=dri_data[sector],
            color=sector_colors.get(sector, "#6A5AE0"),
            dri_score=dri_scores[sector]
        )

        cols[col_idx].plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        col_idx = (col_idx + 1) % n_cols
        if col_idx == 0:
            cols = st.columns(n_cols)

def load_risk_return_panel():
    df = load_parquet("data/robustness/risk_return_summary_all_periods.parquet")
    df["Period"] = df["Period"].astype(str)

    df["Active_Return_3m"] = df["Decarb_Return_3m"] - df["Bench_Return_3m"]
    return df

def plot_distribution(
    df,
    *,
    y,
    title,
    tooltip_text,
    y_label,
    color=None,
    kind="box",
    percent=False,
    bps=False,
    sector_colors=None
):
    import plotly.express as px

    assert not (percent and bps), "Use either percent or bps, not both."

    # 🔹 External title with info icon
    title_with_info_centered(title, tooltip_text)

    color_kwargs = {}
    if color is None and sector_colors is not None:
        color_kwargs["color"] = "Sector"
        color_kwargs["color_discrete_map"] = sector_colors
    elif color is not None:
        color_kwargs["color"] = color

    if kind == "violin":
        fig = px.violin(
            df,
            x="Sector",
            y=y,
            box=True,
            points="outliers",
            hover_data=["Period"],
            **color_kwargs
        )
    else:
        fig = px.box(
            df,
            x="Sector",
            y=y,
            points="outliers",
            hover_data=["Period"],
            **color_kwargs
        )

    fig.update_layout(
        xaxis_title="Sector",
        yaxis_title=y_label,
        template="simple_white",
        margin=dict(t=10, b=80, l=50, r=20)  # 🔑 reduced top margin
    )

    fig.update_xaxes(tickangle=45)

    if percent:
        fig.update_yaxes(tickformat=".1%")

    if bps:
        fig.update_yaxes(ticksuffix=" bps", tickformat=",.0f")

    st.plotly_chart(fig, use_container_width=True)

def render_aggregate_risk_return_distributions():
    title_with_info(
    "Out-of-sample Risk–Return Distributions (across periods)",
   (
    "Each plot shows the distribution across quarters of realized 3-month out-of-sample outcomes. "
    "For every sector-period, we compute annualized TE, annualized volatility (Benchmark vs Decarbonized), "
    "and 3-month return (Benchmark vs Decarbonized). Wider spreads/outliers mean outcomes are less stable over time."
))


    df = load_risk_return_panel()

    plot_kind = st.radio(
        "Plot type",
        ["Box", "Violin"],
        horizontal=True
    )
    kind = "box" if plot_kind == "Box" else "violin"

    # 1) TE
    df["TE_bps"] = df["TE_ann"] * 10000

    plot_distribution(
    df,
    y="TE_bps",
    title="Tracking Error (annualized)",
  tooltip_text=(
    "Distribution of realized out-of-sample tracking error, measured as the "
    "annualized volatility of daily active returns over the following 3 months. "
    "Each sector contributes one observation per period."
),

    y_label="Tracking Error",
    kind=kind,
    bps=True,
    sector_colors=sector_colors
)



    # 2) Volatility (bench vs decarb)
    vol_long = df.melt(
        id_vars=["Sector", "Period"],
        value_vars=["Bench_Vol_ann", "Decarb_Vol_ann"],
        var_name="Portfolio",
        value_name="Vol_ann"
    )
    vol_long["Portfolio"] = vol_long["Portfolio"].replace({
        "Bench_Vol_ann": "Benchmark",
        "Decarb_Vol_ann": "Decarbonized"
    })

    plot_distribution(
    vol_long,
    y="Vol_ann",
    title="Volatility (annualized)",
    tooltip_text=(
        "Distribution of annualized volatility for benchmark and decarbonized portfolios "
        "across all periods. Differences indicate changes in total risk after decarbonization."
    ),
    y_label="Volatility (annualized)",
    color="Portfolio",
    kind=kind,
    percent=True,
    sector_colors=sector_colors
)


    # 3) 3m returns (bench vs decarb)
    ret_long = df.melt(
        id_vars=["Sector", "Period"],
        value_vars=["Bench_Return_3m", "Decarb_Return_3m"],
        var_name="Portfolio",
        value_name="Return_3m"
    )
    ret_long["Portfolio"] = ret_long["Portfolio"].replace({
        "Bench_Return_3m": "Benchmark",
        "Decarb_Return_3m": "Decarbonized"
    })

    plot_distribution(
    ret_long,
    y="Return_3m",
    title="3-month Return",
    tooltip_text=(
        "Distribution of realized 3-month returns across all evaluation periods, "
        "comparing benchmark and decarbonized portfolios."
    ),
    y_label="3m Return",
    color="Portfolio",
    kind=kind,
    percent=True,
    sector_colors=sector_colors
)


def _avg_by_sector(df: pd.DataFrame, cols: list[str], sector_col="Sector"):
    out = df.groupby(sector_col, as_index=False)[cols].mean(numeric_only=True)
    return out

def plot_bar_subplots_horizontal(
    df_sector_avg,
    metrics,                      # [(colname, subtitle), ...]
    *,
    dri_order,                    # list of sectors ordered by DRI
    percent_cols=set(),
    bps_cols=set(),
    bar_color="#6A9CFD",          # ← benchmark light blue
):
    dfp = df_sector_avg.copy()

    # Enforce DRI ordering
    order = [s for s in dri_order if s in set(dfp["Sector"])]

    dfp["Sector"] = pd.Categorical(dfp["Sector"], categories=order, ordered=True)
    dfp = dfp.sort_values("Sector")

    sectors = dfp["Sector"].tolist()[::-1]
    dfp = dfp.iloc[::-1]

    
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        subplot_titles=[m[1] for m in metrics],
        shared_yaxes=True,
        horizontal_spacing=0.08
    )

    for j, (col, subtitle) in enumerate(metrics, start=1):
        y = pd.to_numeric(dfp[col], errors="coerce").astype(float).values

        # ---- Unit handling ----
        if col in bps_cols:
            x = y * 10000
            hover_fmt = ":.1f"
            hover_suf = " bps"
            xaxis_title = "bps"
        elif col in percent_cols:
            x = y * 100
            hover_fmt = ":.2f"
            hover_suf = "%"
            xaxis_title = "%"
        else:
            x = y
            hover_fmt = ":.3f"
            hover_suf = ""
            xaxis_title = ""

        fig.add_trace(
            go.Bar(
                x=x,
                y=sectors,
                orientation="h",
                marker=dict(color=bar_color),
                showlegend=False,   # ← no legend
                hovertemplate=(
                    "%{y}<br>"
                    f"{subtitle}: %{{x{hover_fmt}}}{hover_suf}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=j
        )

        if xaxis_title:
            fig.update_xaxes(title_text=xaxis_title, row=1, col=j)

    # Hide default y-axis labels
    fig.update_yaxes(showticklabels=False)

    # ---- White sector labels (annotations) ----
    for s in sectors:
        fig.add_annotation(
            x=0,
            xref="paper",
            y=s,
            yref="y",
            text=f"<b>{s}</b>",
            showarrow=False,
            xanchor="right",
            align="right",
            font=dict(color="white", size=12),  # ← white labels
        )

    fig.update_layout(

        template="plotly_dark",      # white labels need dark background
        height=520,
        margin=dict(t=60, b=20, l=190, r=20),
    )

    return fig

def get_dri_order():
    dri = load_parquet("data/DRI/decarbonization_readiness_index.parquet")
    dri = dri.sort_values("DRI", ascending=False)
    return dri["Sector"].tolist()

def explanation_box(title: str, bullets_or_text: str):
    with st.expander("Explanation", expanded=False):
        st.markdown(bullets_or_text)

def render_room_for_maneuver_bars():
    dri_order = get_dri_order()
    title_with_info_centered(
        "Room for Maneuver — Aggregate Metrics",
        (
             "Room for Maneuver measures early decarbonization potential at low tracking error"
        )
    )
    df = load_parquet("data/room_for_maneuver/room_for_maneuver_scores_by_period.parquet")
    df["Period"] = df["Period"].astype(str)

    avg = _avg_by_sector(df, ["C_at_1pct", "Alignment", "TE_for_50pctCut"])

    fig = plot_bar_subplots_horizontal(
        avg,
        metrics=[
            ("C_at_1pct", "Carbon reduction @ 1% TE"),
            ("Alignment", "Carbon–Weight alignment"),
            ("TE_for_50pctCut", "TE for 50% max decarb"),
        ],
        dri_order=dri_order,
        percent_cols={"C_at_1pct"},
        bps_cols={"TE_for_50pctCut"}
    )
    st.plotly_chart(fig, use_container_width=True)
    explanation_box(
        "Room for Maneuver (Aggregate)",
        """
These bars show Room for Maneuver metrics averaged over all periods for each sector, with sectors sorted by DRI.
Room for Maneuver captures **how quickly a sector can decarbonize under tight tracking-error constraints**.

- **Carbon reduction @ 1% TE:** How much of the sector’s *maximum attainable* decarbonization can already be reached at a very tight 1% tracking-error cap. Higher = more “early” decarbonization potential.
- **Carbon–Weight alignment:** Spearman correlation between benchmark weights and benchmark carbon contributions (weight × carbon intensity). Lower (more negative) typically means carbon is concentrated in smaller benchmark weights, making early decarbonization easier.
- **TE for 50% max decarb:** The tracking error needed to reach half of the sector’s maximum attainable decarbonization. Lower = the sector reaches meaningful decarbonization sooner.
"""
    )

def render_flexibility_bars():
    dri_order = get_dri_order()

    title_with_info_centered(
        "Flexibility — Aggregate Metrics",
        (
        "Flexibility measures how many different portfolio allocations can achieve essentially the same decarbonization outcome."

        )
    )
    df = load_parquet("data/flexibility/sector_flexibility_raw.parquet")
    df["Period"] = df["Period"].astype(str)

    avg = _avg_by_sector(df, ["Median_bandwidth", "L2_lower_bound_same_obj"])

    fig = plot_bar_subplots_horizontal(
        avg,
        metrics=[
            ("Median_bandwidth", "Median ε-bandwidth"),
            ("L2_lower_bound_same_obj", "L2 lower bound"),
        ],
        dri_order=dri_order,
        percent_cols={"Median_bandwidth"}
    )
    st.plotly_chart(fig, use_container_width=True)
    explanation_box(
        "Flexibility (Aggregate)",
        """
These bars show Flexibility metrics averaged over all periods for each sector, with sectors sorted by DRI.
Flexibility captures whether a sector’s decarbonized solution is **unique and fragile** or whether **many different allocations** can achieve essentially the same outcome.

- **Median ε-bandwidth (ε = 2%):** Stock-level “wiggle room.” For each stock, we find its maximum and minimum feasible weight while remaining long-only, fully invested, within the 2% tracking-error cap, and achieving at least **98% of the optimal carbon reduction**. The bandwidth is (max − min). The chart shows the **median bandwidth across stocks** in the sector. Higher = many near-equivalent implementations.
- **L2 lower bound:** Portfolio-level flexibility. It measures how far the sector can move away from the baseline optimized allocation while still meeting the same tracking-error and carbon objectives. Higher = the sector admits **structurally different portfolios** with similar decarbonization performance.
"""
    )

def render_sensitivity_bars():
    dri_order = get_dri_order()

    title_with_info_centered(
        "Sensitivity — Aggregate Metrics",
        (
            "Sensivitiy measures stability to estimation noise and data uncertainty"
        )
    )
    df = load_parquet("data/sensitivity/sensitivity_scores_by_period.parquet")
    df["Period"] = df["Period"].astype(str)

    df["Inv_Median_Cosine"] = 1 - df["Median_Cosine"]

    avg = _avg_by_sector(df, ["Median_Turnover_pct", "Inv_Median_Cosine", "P95_CarbonLoss_pp"])

    fig = plot_bar_subplots_horizontal(
        avg,
        metrics=[
            ("Median_Turnover_pct", "Median turnover (%)"),
            ("Inv_Median_Cosine", "Inverse Median Cosine"),
            ("P95_CarbonLoss_pp", "P95 carbon loss (pp)"),
        ],
        dri_order=dri_order,
        # turnover already in percent units -> don't scale
        percent_cols=set()
    )
    st.plotly_chart(fig, use_container_width=True)
    explanation_box(
        "Sensitivity (Aggregate)",
        """
These bars show Sensitivity metrics averaged over all periods for each sector, with sectors sorted by DRI.
Sensitivity captures **how much optimized portfolios change under estimation noise** in the risk inputs (covariance estimated from returns).

We run **200 perturbation trials per sector and period**, adding small random noise to the monthly returns used to estimate the covariance matrix, and re-optimizing.

- **Median turnover (%):** Typical fraction of portfolio weight that must be reallocated under perturbations (classic one-way turnover). Higher = less stable weights.
- **Inverse median cosine:** A structure-change indicator. Cosine similarity close to 1 means perturbed portfolios point in the same “direction” as the baseline weights; taking (1 − cosine) makes higher = bigger structural change.
- **P95 carbon loss (pp):** “Worst plausible” loss in achieved carbon reduction. P95 means **95% of trials have carbon loss ≤ this value** (in percentage points) relative to the baseline optimized portfolio.
"""
    )

def render_robustness_bars(out_of_sample_freq="annualized"):
    dri_order = get_dri_order()
    title_with_info_centered(
        "Robustness — Aggregate Metrics",
        (
            "Robustness measures how well decarbonized portfolios control risk out of sample, relative to sector volatility." 
    ))
    rob = load_parquet("data/robustness/robustness_scores_by_period.parquet")
    rob["period"] = rob["period"].astype(str)
    rob = rob.rename(columns={"sector": "Sector", "period": "Period"})

    te_col  = f"{out_of_sample_freq}_TE"
    vol_col = f"{out_of_sample_freq}_volatility"

    avg = rob.groupby("Sector", as_index=False)[[te_col, vol_col]].mean(numeric_only=True)

    fig = plot_bar_subplots_horizontal(
        avg,
        metrics=[
            (te_col,  "Tracking error (annualized)"),
            (vol_col, "Benchmark volatility (annualized)"),
        ],
        dri_order=dri_order,
        bps_cols={te_col},
        percent_cols={vol_col}
    )
    st.plotly_chart(fig, use_container_width=True)
    explanation_box(
        "Robustness (Aggregate)",
        """
These bars show Robustness metrics averaged over all periods for each sector, with sectors sorted by DRI.
Robustness asks whether decarbonized portfolios **keep tracking the benchmark in practice**, not just in-sample.

- **Out-of-sample tracking error (annualized):** Realized volatility of active returns (decarbonized minus benchmark) over the next 3 months, annualized. Lower = the portfolio tracks its benchmark more reliably out of sample.
- **Benchmark volatility (annualized):** Sector’s own return volatility over the same out-of-sample window. Higher volatility means that small weight differences can translate into larger active risk, making tight tracking structurally harder.
"""
    )

# ---------------------------------------------------------
# Aggregate composition tables: concentration + active weights
# ---------------------------------------------------------


# --- config ---
PERIODS = ["0321","0621","0921","1221","0322","0622","0922","1222","0323","0623","0923","1223"]
TARGET_TE_BPS = 200  # 2% TE





# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _hhi(w: np.ndarray) -> float:
    w = np.asarray(w, float).ravel()
    return float(np.sum(w**2))

def _neff(w: np.ndarray) -> float:
    h = _hhi(w)
    return float(1.0 / h) if h > 0 else np.nan

def get_dri_order() -> list[str]:
    dri_path = "data/DRI/decarbonization_readiness_index.parquet"

    dri = load_parquet(dri_path)
    dri = dri.sort_values("DRI", ascending=False)
    return dri["Sector"].tolist()

def _load_optimal_portfolios_all_te(period: str) -> dict:
    pkl = f"data/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"
    return load_pickle(pkl)

def _extract_at_target_te(opt_all_te_for_one_sector: dict, target_te_bps: float):
    """
    Robustly pick the closest TE point (in bps) to target_te_bps.
    Works even if your stored structure is:
      - "tracking_errors" + "weights_by_te"
      - OR "frontier" list of dicts
      - OR already stored 'w_opt'/'w_bench' (single point)
    Returns (w_bench, w_opt) or (None, None).
    """
    d = opt_all_te_for_one_sector

    # Case A: already single-point
    if "w_bench" in d and "w_opt" in d:
        return np.asarray(d["w_bench"], float).ravel(), np.asarray(d["w_opt"], float).ravel()

    # Case B: arrays by TE
    if "tracking_errors" in d and "weights_by_te" in d:
        te = np.asarray(d["tracking_errors"], float).ravel()
        weights_by_te = d["weights_by_te"]  # list/array of weight vectors (opt)

        if len(te) == 0 or len(weights_by_te) == 0:
            return None, None

        idx = int(np.nanargmin(np.abs(te - target_te_bps)))
        w_opt = np.asarray(weights_by_te[idx], float).ravel()

        # benchmark weights: try common keys
        for k in ["w_bench", "benchmark_weights", "bench_weights"]:
            if k in d:
                w_bench = np.asarray(d[k], float).ravel()
                return w_bench, w_opt

        return None, None

    # Case C: frontier list
    if "frontier" in d and isinstance(d["frontier"], list) and len(d["frontier"]) > 0:
        te = np.array([pt.get("tracking_error", np.nan) for pt in d["frontier"]], float)
        idx = int(np.nanargmin(np.abs(te - target_te_bps)))
        pt = d["frontier"][idx]
        w_opt = np.asarray(pt.get("w_opt", None), float).ravel() if pt.get("w_opt", None) is not None else None
        w_bench = np.asarray(pt.get("w_bench", None), float).ravel() if pt.get("w_bench", None) is not None else None
        return w_bench, w_opt

    return None, None


# ---------------------------------------------------------
# Build tables
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_aggregate_concentration_tables(
    periods: list[str] = PERIODS,
    target_te_bps: float = TARGET_TE_BPS
):
    """
    Returns:
      df_neff_avg: Sector-level averages across periods (Neff + Delta)
      df_hhi_avg:  Sector-level averages across periods (HHI + Delta)
      df_panel:    Sector-period panel (raw, in case you want it)
    """
    rows = []

    for period in periods:
        all_te = _load_optimal_portfolios_all_te(period)
        if not all_te:
            continue

        for sector, d in all_te.items():
            w_b, w_o = _extract_at_target_te(d, target_te_bps)
            if w_b is None or w_o is None:
                continue

            rows.append({
                "Sector": sector,
                "Period": period,
                "HHI_Benchmark": _hhi(w_b),
                "HHI_Decarbonized": _hhi(w_o),
                "Neff_Benchmark": _neff(w_b),
                "Neff_Decarbonized": _neff(w_o),
            })

    df_panel = pd.DataFrame(rows)
    if df_panel.empty:
        return pd.DataFrame(), pd.DataFrame(), df_panel

    df_panel["Delta_Neff"] = df_panel["Neff_Decarbonized"] - df_panel["Neff_Benchmark"]
    df_panel["Delta_HHI"] = df_panel["HHI_Decarbonized"] - df_panel["HHI_Benchmark"]

    # Averages across periods
    df_avg = (
        df_panel
        .groupby("Sector", as_index=False)
        .mean(numeric_only=True)
    )

    df_neff_avg = df_avg[[
        "Sector", "Neff_Benchmark", "Neff_Decarbonized", "Delta_Neff"
    ]].copy()

    df_hhi_avg = df_avg[[
        "Sector", "HHI_Benchmark", "HHI_Decarbonized", "Delta_HHI"
    ]].copy()

    return df_neff_avg, df_hhi_avg, df_panel


@st.cache_data(show_spinner=False)
def build_aggregate_active_weight_table(
    periods: list[str] = PERIODS,
    target_te_bps: float = TARGET_TE_BPS,
    big_move_threshold: float = 0.01,  # 1%
):
    """
    Per sector-period, compute summary stats of Δw = w_opt - w_bench.
    Then average those stats across periods for each sector.
    """
    rows = []

    for period in periods:
        all_te = _load_optimal_portfolios_all_te(period)
        if not all_te:
            continue

        for sector, d in all_te.items():
            w_b, w_o = _extract_at_target_te(d, target_te_bps)
            if w_b is None or w_o is None:
                continue

            w_b = np.asarray(w_b, float).ravel()
            w_o = np.asarray(w_o, float).ravel()
            if w_b.shape != w_o.shape or w_b.size == 0:
                continue

            dw = w_o - w_b
            abs_dw = np.abs(dw)

            rows.append({
    "Sector": sector,
    "Period": period,

    # Directional (optional, diagnostic)
    "Mean_|Δw|": float(np.mean(abs_dw)),
    "Median_|Δw|": float(np.median(abs_dw)),

    # Dispersion (what you want to show)
    "Min_|Δw|": float(np.min(abs_dw)),
    "Max_|Δw|": float(np.max(abs_dw)),
    "Std_|Δw|": float(np.std(abs_dw, ddof=0)),
    f"Pct_|Δw|>{big_move_threshold:.0%}": float(np.mean(abs_dw > big_move_threshold) * 100.0),
})


    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame(), panel

    avg = (
        panel
        .groupby("Sector", as_index=False)
        .mean(numeric_only=True)
    )

    return avg, panel


# ---------------------------------------------------------
# Render in Streamlit (Aggregate view)
# ---------------------------------------------------------
def render_aggregate_composition_tables():
    dri_order = get_dri_order()

    st.markdown("## Composition & Concentration (average across periods)")

    df_neff, df_hhi, _panel = build_aggregate_concentration_tables()

    if df_neff.empty:
        st.info("No concentration data found (check optimal portfolio PKLs / structure).")
        return

    # Order by DRI if available
    if dri_order:
        cat = pd.Categorical(df_neff["Sector"], categories=dri_order, ordered=True)
        df_neff = df_neff.assign(_ord=cat).sort_values("_ord").drop(columns="_ord")
        cat = pd.Categorical(df_hhi["Sector"], categories=dri_order, ordered=True)
        df_hhi = df_hhi.assign(_ord=cat).sort_values("_ord").drop(columns="_ord")

    # Nicely formatted copies
    df_neff_show = df_neff.copy()
    for c in ["Neff_Benchmark", "Neff_Decarbonized", "Delta_Neff"]:
        df_neff_show[c] = df_neff_show[c].round(2)

    df_hhi_show = df_hhi.copy()
    for c in ["HHI_Benchmark", "HHI_Decarbonized", "Delta_HHI"]:
        df_hhi_show[c] = df_hhi_show[c].round(4)

    title_with_info(
    "Concentration (HHI)",
    (
    "Herfindahl–Hirschman Index (HHI) measures how concentrated portfolio weights are across holdings."
    "It is computed as the sum of squared portfolio weights."
    "Higher values = greater concentration in a few names."
    )
)
    st.dataframe(df_hhi_show, use_container_width=True, hide_index=True)


    title_with_info(
    "Effective holdings (Neff = 1 /(HHI)",
    (
    "Effective holdings (Neff) indicate how many equally weighted stocks"
    " would generate the same concentration."
    )
)
    st.dataframe(df_neff_show, use_container_width=True, hide_index=True)

    title_with_info(
    "Active weight dispersion (average across periods)",
    (
    "Active weight dispersion captures the spread of absolute differences"
    " between decarbonized and benchmark weights across stocks."
    ))
    df_aw_avg, _aw_panel = build_aggregate_active_weight_table(big_move_threshold=0.01)

    if df_aw_avg.empty:
        st.info("No active weight data found (check optimal portfolio PKLs / structure).")
        return

    if dri_order:
        cat = pd.Categorical(df_aw_avg["Sector"], categories=dri_order, ordered=True)
        df_aw_avg = df_aw_avg.assign(_ord=cat).sort_values("_ord").drop(columns="_ord")

    # Format
    df_aw_show = df_aw_avg.copy()
    # Δw stats are in weight units (0–1). Keep as percentages for readability:
    for c in ["Mean_|Δw|", "Median_|Δw|", "Min_|Δw|", "Max_|Δw|", "Std_|Δw|"]:
        df_aw_show[c] = (df_aw_show[c] * 100).round(3)  # percentage points
        df_aw_show = df_aw_show.rename(columns={c: c.replace("Δw", "Δw (pp)")})
    # pct column already in %
    pct_col = [c for c in df_aw_show.columns if c.startswith("Pct_|")]
    if pct_col:
        df_aw_show[pct_col[0]] = df_aw_show[pct_col[0]].round(1)

    st.dataframe(df_aw_show, use_container_width=True, hide_index=True)
