from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from data_loaders import load_json, load_parquet, load_pickle

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


OUT_OF_SAMPLE_FREQ = "annualized"

# ---------------------------------------------------------
# Load metric descriptions
# ---------------------------------------------------------
METRIC_DESCRIPTIONS = load_json("metric_descriptions.json")

# ---------------------------------------------------------
# Metric card helper (keep exactly as is)
# ---------------------------------------------------------
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



def info_tooltip(text: str):
    help_html = (text or "").replace("\n", "<br>")
    st.markdown(
        f"""
        <span class="tooltip-icon">i
            <span class="tooltip-text">{help_html}</span>
        </span>
        """,
        unsafe_allow_html=True
    )

def label_with_info(label, tooltip_text):
    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:6px;
            font-size:14px;
            font-weight:500;
            margin-bottom:4px;
        ">
            <span>{label}</span>
            <span class="tooltip-icon">i
                <span class="tooltip-text">{tooltip_text}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

def info_metric(label, value, help_text="", delta=None):
    help_html = (help_text or "").replace("\n", "<br>")

    st.markdown("""
    <style>
    .metric-card {
        border: 1px solid #555;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 4px;
        background-color: #1e1e1e;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #ddd;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
        color: white;
    }
    .delta-pos { color: #00d26a; font-size: 14px; }
    .delta-neg { color: #ff4b4b; font-size: 14px; }
    .tooltip-icon {
        background-color: #d0d0d0;
        color: #333;
        border-radius: 50%;
        width: 15px;
        height: 15px;
        font-size: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: help;
        position: relative;
    }
    .tooltip-icon:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    .tooltip-text {
        visibility: hidden;
        opacity: 0;
        width: 220px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 100;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        font-size: 12px;
        line-height: 1.3;
        transition: opacity 0.25s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

    # Delta formatted
    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float)):
            cls = "delta-pos" if delta >= 0 else "delta-neg"
            sign = "+" if delta >= 0 else ""
            delta_html = f"<span class='{cls}'>{sign}{delta}</span>"

    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">
            {label}
            <span class="tooltip-icon">i
                <span class="tooltip-text">{help_html}</span>
            </span>
        </div>
        <div class="metric-value">{value} {delta_html}</div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)
def section_title(title, sector_name, tooltip_text=None):
    color = sector_colors.get(sector_name, "#6A5AE0")

    tooltip_html = ""
    if tooltip_text:
        tooltip_html = f"""
        <span style="margin-left:8px;">
            <span class="tooltip-icon">i
                <span class="tooltip-text">{tooltip_text}</span>
            </span>
        </span>
        """

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:6px;
            font-size:22px;
            font-weight:700;
            padding:10px 18px;
            background:#111;
            border-left:4px solid {color};
            border-radius:6px;
            margin-top:30px;
            margin-bottom:10px;
        ">
            <span>{title}</span>
            {tooltip_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def explanation_box(title: str, bullets_or_text: str):
    with st.expander("Explanation", expanded=False):
        st.markdown(bullets_or_text)
# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def render_dimension_metrics(sector_name: str, period: str):
    """
    Render Room for Maneuver, Flexibility, Sensitivity and Robustness
    metrics for a given sector and period.
    """

    # =====================================================
    # ROOM FOR MANEUVER
    # =====================================================
    section_title("Room for Maneuver — Key Metrics", sector_name, "Early decarbonization potential at low tracking error")

    rfm_panel = load_parquet("data/room_for_maneuver/room_for_maneuver_scores_by_period.parquet")
    rfm_panel["Period"] = rfm_panel["Period"].astype(str)

    row = rfm_panel[
        (rfm_panel["Sector"] == sector_name) &
        (rfm_panel["Period"] == period)
    ].iloc[0]

    c_at_1pct = row["C_at_1pct"]
    alignment = row["Alignment"]
    te50_raw  = row["TE_for_50pctCut"]

    te50_label = f"{te50_raw * 10000:.0f} bps" if pd.notna(te50_raw) else "Not reached"

    col1, col2, col3 = st.columns(3)

    with col1:
        info_metric(
            "Early carbon reduction (1% TE)",
            f"{c_at_1pct * 100:.0f}%",
            METRIC_DESCRIPTIONS["c_at_1pct"]
        )

    with col2:
        info_metric(
            "Carbon–Weight Alignment",
            f"{alignment:.2f}",
            METRIC_DESCRIPTIONS["alignment"]
        )

    with col3:
        info_metric(
            "TE for 50% reduction",
            te50_label,
            METRIC_DESCRIPTIONS["te_50pct"]
        )

    # --- Explanation (Room for Maneuver) ---
    explanation_box(
        "Room for Maneuver",
        f"""
    - **Early carbon reduction ({c_at_1pct*100:.0f}% at 1% TE):** This means that, under a very tight tracking-error cap of 1%, the sector can already reach **{c_at_1pct*100:.0f}%** of its *own* maximum attainable decarbonization (as defined by the frontier normalization). Higher values indicate more “early” decarbonization potential without needing large benchmark deviations.
    - **Carbon–Weight Alignment ({alignment:.2f}):** This is the Spearman rank correlation (ranges between -1 and 1) between benchmark weights and benchmark carbon contributions (weight × carbon intensity). A **lower** (more negative) alignment typically indicates that carbon intensity is concentrated in smaller-weight names, which would make carbon reduction easier at low tracking error.
    - **TE for 50% reduction ({te50_label}):** This is the smallest tracking error at which the sector reaches **50%** of its own maximum decarbonization. Lower required TE implies the sector reaches meaningful decarbonization earlier along the frontier.
    """
    )


    # =====================================================
    # FLEXIBILITY
    # =====================================================
    section_title(
    "Flexibility",
    sector_name,
    "How many different portfolio allocations can achieve essentially the same decarbonization outcome."
)


    flex = load_parquet("data/flexibility/sector_flexibility_raw.parquet")
    flex["Period"] = flex["Period"].astype(str)

    row = flex[
        (flex["Sector"] == sector_name) &
        (flex["Period"] == period)
    ].iloc[0]

    cols = st.columns(2)

    metrics = [
        ("Median ε-bandwidth", f"{row['Median_bandwidth']:.2%}", "median_bandwidth"),
        ("L2 lower bound", f"{row['L2_lower_bound_same_obj']:.3f}", "l2_lower_bound"),
    ]

    for col, (label, value, key) in zip(cols, metrics):
        with col:
            info_metric(label, value, METRIC_DESCRIPTIONS.get(key, ""))

    explanation_box(
    "Flexibility",
    f"""
    - **ε-bands with ε = 2% ({row['Median_bandwidth']:.2%}):** For each stock in the sector, we solve two feasibility problems that **maximize** and **minimize** its weight while keeping the portfolio long-only, fully invested, within the 2% tracking-error cap, and achieving **at least 98% of the optimal carbon reduction** at that same tracking-error level.
    The difference between each stock's max weight and min weight defines that stock’s **ε-bandwidth**. The **median ε-bandwidth ({row['Median_bandwidth']:.2%})**, computed **across all stocks in the sector**, summarizes typical stock-level flexibility: higher values are an evidence of many near-equivalent decarbonized allocations and therefore less fragile portfolio weights.
    - **L2 lower bound ({row['L2_lower_bound_same_obj']:.3f})**: This metric captures **how different two feasible decarbonized portfolios can be** while still meeting the same tracking-error and carbon-reduction objectives. Starting from the optimized portfolio, we search for alternative portfolios that remain feasible but differ as much as possible in their overall composition. A larger value indicates greater **global flexibility**, meaning the sector allows structurally distinct portfolio allocations that achieve essentially the same decarbonization outcome.
"""
    )


    # =====================================================
    # SENSITIVITY
    # =====================================================
    section_title("Sensitivity — Key Metrics", sector_name, "Stability to estimation noise and data uncertainty")

    sens = load_parquet("data/sensitivity/sensitivity_scores_by_period.parquet")
    sens["Period"] = sens["Period"].astype(str)

    row = sens[
        (sens["Sector"] == sector_name) &
        (sens["Period"] == period)
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        info_metric("Median Turnover (%)", f"{row['Median_Turnover_pct']:.2f}%", METRIC_DESCRIPTIONS["median_turnover"])

    with col2:
        info_metric("Median Cosine Similarity", f"{row['Median_Cosine']:.3f}", METRIC_DESCRIPTIONS["median_cosine"])

    with col3:
        info_metric("P95 Carbon Loss (pp)", f"{row['P95_CarbonLoss_pp']:.2f}", METRIC_DESCRIPTIONS["p95_carbon_loss"])

    explanation_box(
    "Sensitivity",
    f"""
Sensitivity measures how **stable the optimized decarbonized portfolio remains when the input data are slightly noisy**.

To assess this, we run **200 perturbation trials** per sector and period. In each trial, small random noise is added to the historical monthly return data used to estimate risk. This mimics realistic estimation uncertainty arising from limited return histories and tests whether the optimized portfolio changes materially under plausible data fluctuations.

- **Median Turnover ({row['Median_Turnover_pct']:.2f}%) = Weight reallocation under noise:**  
  The typical fraction of the portfolio that would need to be reallocated when inputs are perturbed. Higher values indicate less stable portfolio weights.

- **Median Cosine Similarity ({row['Median_Cosine']:.3f}) = Stability of portfolio structure:**  
  Measures how similar the perturbed portfolios are to the baseline allocation. Values closer to 1 indicate that the overall structure of the portfolio is preserved; lower values indicate stronger structural shifts.

- **P95 carbon loss ({row['P95_CarbonLoss_pp']:.2f}) = Worst-case loss in carbon reduction:**  
 In 95% of perturbation trials, the achieved carbon reduction deteriorates by no more than {row['P95_CarbonLoss_pp']:.2f} percentage points relative to the baseline optimized portfolio. Larger values indicate greater fragility of decarbonization outcomes to estimation noise.
"""
)

    # =====================================================
    # ROBUSTNESS
    # =====================================================
    section_title("Robustness — Key Metrics", sector_name, "How well decarbonized portfolios control risk out of sample, relative to sector volatility.")

    rob = load_parquet("data/robustness/robustness_scores_by_period.parquet")
    rob["period"] = rob["period"].astype(str)
    rob = rob.rename(columns={"sector": "Sector", "period": "Period"})

    row = rob[
        (rob["Sector"] == sector_name) &
        (rob["Period"] == period)
    ].iloc[0]

    te_bps  = row[f"{OUT_OF_SAMPLE_FREQ}_TE"] * 10000
    vol_pct = row[f"{OUT_OF_SAMPLE_FREQ}_volatility"] * 100

    col1, col2 = st.columns(2)

    with col1:
        info_metric(
            "Out-of-sample Tracking Error (bps)",
            f"{te_bps:.1f}",
            METRIC_DESCRIPTIONS[f"{OUT_OF_SAMPLE_FREQ}_te"]
        )

    with col2:
        info_metric(
            "Sector Benchmark Volatility",
            f"{vol_pct:.2f}%",
            METRIC_DESCRIPTIONS[f"{OUT_OF_SAMPLE_FREQ}_volatility"]
        )
    explanation_box(
    "Robustness",
    f"""
    - **Out-of-sample TE:** Realized tracking error over the next 3 months indicates how closely the decarbonized portfolio tracks its sector benchmark out of sample.
    - **Benchmark volatility:** Captures the inherent return variability of the sector. In more volatile sectors, small differences in portfolio weights translate into larger active risk, so maintaining a tight tracking-error constraint is structurally more challenging."""
    )

def load_sector_frontier(sector_name, period):
    """Load TE–carbon frontier for a given sector and period."""
    pickle_path = f"data/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"


    all_periods_data = load_pickle(pickle_path)

    if sector_name not in all_periods_data:
        st.error(f"No data for sector {sector_name} in period {period}.")
        st.stop()

    data = all_periods_data[sector_name]

    te = np.asarray(data["tracking_errors"], float)     # bps
    cr = np.asarray(data["carbon_reductions"], float)   # %

    return te, cr

def plot_frontier_percent(sector_name, period):
   
    title_with_info_centered(
        f"{sector_name} — TE–Carbon Frontier",
        (
            "The TE–Carbon frontier shows the trade-off between tracking error and achievable "
            "carbon reduction. Each point corresponds to an optimal portfolio under a different "
            "tracking-error constraint. Steeper frontiers indicate stronger decarbonization "
            "potential at low tracking error."
        )
    )

    te, cr = load_sector_frontier(sector_name, period)

    # ---- Elbow point (L-curve) ----
    x0, y0 = te[0], cr[0]
    x1, y1 = te[-1], cr[-1]

    distances = np.abs(
        (te - x0) * (y1 - y0) - (cr - y0) * (x1 - x0)
    ) / np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    elbow_idx = np.argmax(distances)
    elbow_te = te[elbow_idx]
    elbow_cr = cr[elbow_idx]

    # ---- Plot ----
    fig = go.Figure()

    color = sector_colors.get(sector_name, "#6A5AE0")
# Frontier line
    fig.add_trace(
        go.Scatter(
            x=te,
            y=cr,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            showlegend=False,
            hovertemplate=(
                "TE: %{x:.1f} bps<br>"
                "Carbon reduction: %{y:.1f}%"
                "<extra></extra>"   # 🔴 removes “trace 0”
            )
        )
    )

    # Elbow marker (STAR)
    fig.add_trace(
        go.Scatter(
            x=[elbow_te],
            y=[elbow_cr],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=18,
                color=color,
                line=dict(color="black", width=1)
            ),
            showlegend=False,
            hovertemplate=(
                "<b>Elbow</b><br>"
                "TE: %{x:.1f} bps<br>"
                "Carbon reduction: %{y:.1f}%"
                "<extra></extra>"   # 🔴 removes “trace 0”
            )
        )
    )



    fig.update_layout(
        xaxis_title="Tracking Error (bps)",
        yaxis_title="Carbon Reduction (%)",
        template="simple_white",
        hoverlabel=dict(
            font_size=14,
            bgcolor="white",
            font_color="black"
        ),
         margin=dict(t=10, b=40), 
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_marginal_gains(sector_name, period):
    title_with_info_centered(
    f"{sector_name} — Marginal Carbon Gain",
    (
        "The marginal carbon gain shows how much additional carbon reduction is obtained "
        "for a small increase in tracking error along the TE–Carbon frontier. "
        "Higher values indicate that relaxing the tracking-error constraint yields "
        "large incremental decarbonization benefits, while lower values suggest "
        "diminishing returns to additional tracking error."
    )
)

    te, cr = load_sector_frontier(sector_name, period)
    color = sector_colors.get(sector_name, "#6A5AE0")
    # Marginal gains
    marginal = np.gradient(cr, te)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=te,
        y=marginal,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=6),
        hovertemplate=(
            "TE (bps): %{x:.2f}<br>"
            "Marginal Gain: %{y:.2f}<extra></extra>"
        ),
        showlegend=False
    ))

    fig.update_layout(
        xaxis_title="Tracking Error (bps)",
        yaxis_title="Marginal Gain (ΔCR / ΔTE)",
        template="simple_white",
        hoverlabel=dict(
            font_size=14,
            bgcolor="white",
            font_color="black"
        ),
         margin=dict(t=20, b=40),  
    )

    st.plotly_chart(fig, use_container_width=True)

# app_individual_view_function.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_individual_radar(sector_name, sectors):
    """
    Render the Decarbonization Readiness Index (DRI) section:
    - Radar chart
    - DRI score table
    - Dimension score table
    - Optional sector comparison
    """

    # ---------------------------------------------------------
    # Vertical spacing
    # ---------------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Load DRI table
    # ---------------------------------------------------------
    dri_df = load_parquet("data/DRI/decarbonization_readiness_index.parquet")

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
    # Selected sector row
    row = dri_df.loc[dri_df["Sector"] == sector_name].iloc[0]

    room_norm   = float(row["Room_norm"])
    flex_norm   = float(row["Flex_norm"])
    sens_norm   = float(row["Sens_norm"])
    robust_norm = float(row["Robust_norm"])
    dri_score   = float(row["DRI"])

    # ---------------------------------------------------------
    # Layout: Radar | Tables | Compare
    # ---------------------------------------------------------
    col1, col2, spacer, col3 = st.columns([1, 0.7, 0.005, 0.45])

    # ---------------------------------------------------------
    # Compare selector
    # ---------------------------------------------------------

    compare_options = ["None"] + [s for s in sectors if s != sector_name]


    with col3:
        st.markdown("<div style='height:120px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="display:flex; justify-content:center;">
            """,
            unsafe_allow_html=True
        )

        compare_sector = st.selectbox(
            "Compare with:",
            options=compare_options,
            index=0,
            key="compare_selector"
        )

        st.markdown("</div>", unsafe_allow_html=True)



    # ---------------------------------------------------------
    # Load comparison sector (if any)
    # ---------------------------------------------------------
    if compare_sector != "None":
        comp_row = dri_df.loc[dri_df["Sector"] == compare_sector].iloc[0]

        comp_room_norm   = float(comp_row["Room_norm"])
        comp_flex_norm   = float(comp_row["Flex_norm"])
        comp_sens_norm   = float(comp_row["Sens_norm"])
        comp_robust_norm = float(comp_row["Robust_norm"])
        comp_dri_score   = float(comp_row["DRI"])

    # ---------------------------------------------------------
    # Radar chart
    # ---------------------------------------------------------
    with col1:
        labels = ["Room for Maneuver", "Flexibility", "Sensitivity", "Robustness"]

        r_main = [room_norm, flex_norm, sens_norm, robust_norm, room_norm]

        fig = go.Figure()
    
        main_color = sector_colors.get(sector_name, "#6A5AE0")

  
        fig.add_trace(
        go.Scatterpolar(
            r=r_main,
            theta=labels + [labels[0]],
            fill="toself",
            name=sector_name,
            line=dict(color=main_color, width=3),
            opacity=0.85,
            showlegend=False,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Score: %{r:.2f}<br>"
                "<extra></extra>"
            )
        )
    )

        if compare_sector != "None":
            r_comp = [
                comp_room_norm,
                comp_flex_norm,
                comp_sens_norm,
                comp_robust_norm,
                comp_room_norm
            ]
            comp_color = sector_colors.get(compare_sector, "#FF9F1C")

            fig.add_trace(go.Scatterpolar(
                r=r_comp,
                theta=labels + [labels[0]],
                fill="toself",
                name=compare_sector,
                line=dict(color=comp_color, width=3),
                opacity=0.55
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False, ticks=""),
                angularaxis=dict(tickfont=dict(size=12)),
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            ),
            width=500,
            height=400,
            margin=dict(l=30, r=30, t=40, b=40),
            template="plotly_dark"
        )

        st.plotly_chart(fig, config={"displayModeBar": False})

  
    # ---------------------------------------------------------
    # Tables (DRI + dimensions)
    # ---------------------------------------------------------
    with col2:

        st.markdown("#### Overall DRI Score")

        if compare_sector == "None":
            st.markdown(
                f"""
                <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
                    <tr>
                        <td style="font-size:16px; font-weight:600;">DRI</td>
                        <td style="
                            font-size:20px;
                            font-weight:700;
                            color:{main_color};
                            text-align:right;
                        ">
                            {dri_score:.3f}
                        </td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
                    <tr>
                        <th></th>
                        <th style="text-align:right;">{sector_name}</th>
                        <th style="text-align:right;">{compare_sector}</th>
                    </tr>
                    <tr>
                        <td>DRI</td>
                        <td style="text-align:right; color:{main_color}; font-weight:600;">
                            {dri_score:.3f}
                        </td>
                        <td style="text-align:right; color:{comp_color}; font-weight:600;">
                            {comp_dri_score:.3f}
                        </td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

        # --------------------------------------------------
        # Dimension scores
        # --------------------------------------------------
        st.markdown("#### Dimension Scores")

        def dim_row(name, v1, v2=None):
            if v2 is None:
                return (
                    f"<tr>"
                    f"<td>{name}</td>"
                    f"<td style='text-align:right; color:{main_color}; font-weight:600;'>"
                    f"{v1:.3f}</td></tr>"
                )
            return (
                f"<tr>"
                f"<td>{name}</td>"
                f"<td style='text-align:right; color:{main_color}; font-weight:600;'>"
                f"{v1:.3f}</td>"
                f"<td style='text-align:right; color:{comp_color}; font-weight:600;'>"
                f"{v2:.3f}</td>"
                f"</tr>"
            )

        rows = (
            dim_row("Room for Maneuver", room_norm,
                    comp_room_norm if compare_sector != "None" else None)
            + dim_row("Flexibility", flex_norm,
                    comp_flex_norm if compare_sector != "None" else None)
            + dim_row("Sensitivity", sens_norm,
                    comp_sens_norm if compare_sector != "None" else None)
            + dim_row("Robustness", robust_norm,
                    comp_robust_norm if compare_sector != "None" else None)
        )

        header = (
            "<tr><th>Dimension</th><th style='text-align:right;'>Score</th></tr>"
            if compare_sector == "None"
            else f"<tr><th>Dimension</th><th>{sector_name}</th><th>{compare_sector}</th></tr>"
        )

        st.markdown(
            f"""
            <table style="width:100%; border:1px solid #444; border-radius:6px; padding:6px;">
                {header}
                {rows}
            </table>
            """,
            unsafe_allow_html=True
        )


def render_risk_return_profiles(sector_name: str, period: str):
    st.markdown("## Risk–return profiles")

    # --- Load timeseries ---
    ts_path = "data/robustness/risk_return_timeseries_all_periods.parquet"
    ts = load_parquet(ts_path)
    ts = ts[(ts["Sector"] == sector_name) & (ts["Period"] == period)].copy()

    if ts.empty:
        st.info("No out-of-sample return series found for this sector/period.")
        return

    ts["Date"] = pd.to_datetime(ts["Date"])
    ts = ts.sort_values("Date")

    # Cumulative index (start at 100)
    bench_idx = 100 * (1 + ts["Bench_Return"]).cumprod()
    decarb_idx = 100 * (1 + ts["Decarb_Return"]).cumprod()

    # --- Load summary ---
    summ_path = "data/robustness/risk_return_summary_all_periods.parquet"
    summ = load_parquet(summ_path)
    summ["Period"] = summ["Period"].astype(str)

    row = summ[(summ["Sector"] == sector_name) & (summ["Period"] == period)].iloc[0]

    # Layout: plot left, tables right
    left, right = st.columns([2.2, 1])

    with left:
        fig = go.Figure()
        color_main = sector_colors.get(sector_name, "#6A5AE0")
        plot_title = "Out-of-sample Cumulative Return Index (3 months)"

        tooltip_text = (
            "The cumulative return index shows the compounded path of daily returns over the "
            "out-of-sample window, normalized to 100 at the start of the period, allowing direct "
            "comparison of tracking behavior between the benchmark and the decarbonized portfolio."
        )

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:8px;
                font-size:18px;
                font-weight:600;
                margin-bottom:2px;
            ">
                <span>{plot_title}</span>
                <span class="tooltip-icon">i
                    <span class="tooltip-text">{tooltip_text}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig.add_trace(go.Scatter(
            x=ts["Date"], y=bench_idx, mode="lines",
            name="Benchmark", line=dict(width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Index: %{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=ts["Date"], y=decarb_idx, mode="lines",
            name="Decarbonized", line=dict(width=2, color=color_main),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Index: %{y:.2f}<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Return Index (Base = 100)",
            template="simple_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),margin=dict(
        t=10,   # ↓ this is the key one
        b=40,
        l=50,
        r=20
    
)
        )
        st.plotly_chart(fig, use_container_width=True)

    def tiny_table(title, bench_val, decarb_val, fmt="{:.2%}"):
        df = pd.DataFrame(
            {"Benchmark": [fmt.format(bench_val)], "Decarbonized": [fmt.format(decarb_val)]},
            index=[title]
        )
        st.dataframe(df, use_container_width=True, hide_index=False)

    with right:
        st.markdown("#### Summary (OOS 3 months)")

        # 1) Returns (3m total return)
        tiny_table(
            "Return",
            float(row["Bench_Return_3m"]),
            float(row["Decarb_Return_3m"]),
            fmt="{:.2%}"
        )

        # 2) Volatility (annualized)
        tiny_table(
            "Volatility (ann.)",
            float(row["Bench_Vol_ann"]),
            float(row["Decarb_Vol_ann"]),
            fmt="{:.2%}"
        )

        # TE card (deck-style)
        te_ann = float(row["TE_ann"])
        info_metric(
            "Tracking Error (annualized)",
            f"{te_ann*10000:.1f} bps",
            METRIC_DESCRIPTIONS.get("annualized_te", "")
        )

def compute_concentration_metrics(w: np.ndarray):
    """
    Concentration via HHI = sum(w_i^2). Also report effective number of holdings.
    """
    w = np.asarray(w, float).ravel()
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        return np.nan, np.nan
    w = w / s
    hhi = float(np.sum(w**2))
    n_eff = float(1.0 / hhi) if hhi > 0 else np.nan
    return hhi, n_eff


def load_sector_weights_at_target_te(sector_name: str, period: str, target_te_bps: int = 200):
    """
    Loads w_bench and w_opt at the target TE from your optimal_portfolios_all_te_{period}.pkl.
    Assumes you stored a frontier and can extract the portfolio at ~200 bps.
    If you already have a helper like extract_optimal_portfolios_at_target_te, use that instead.
    """
    pkl = f"data/optimal_portfolios/optimal_portfolios_all_te_{period}.pkl"

    all_data = load_pickle(pkl)

    if sector_name not in all_data:
        st.error(f"No data for sector {sector_name} in period {period}.")
        st.stop()

    d = all_data[sector_name]

    te = np.asarray(d["tracking_errors"], float)  # bps
    weights_by_te = d["weights_by_te"]            # list/array of w vectors
    stock_labels = list(d["stock_labels"])

    # If benchmark weights are stored once:
    w_bench = np.asarray(d.get("w_bench", None), float).ravel() if "w_bench" in d else None

    # Choose index closest to target TE
    idx = int(np.argmin(np.abs(te - target_te_bps)))
    w_opt = np.asarray(weights_by_te[idx], float).ravel()

    if w_bench is None:
        # Fallback: if benchmark is included in weights_by_te at TE=0, adapt as needed
        st.warning("Benchmark weights not found in pickle (d['w_bench']). Please wire them in.")
        w_bench = np.zeros_like(w_opt)

    return stock_labels, w_bench, w_opt, float(te[idx])

def render_composition_section(sector_name: str, period: str, target_te_bps: int = 200):
    section_title(
        "Composition — Benchmark vs Decarbonized",
        sector_name,
        "Compares how concentrated the benchmark and decarbonized portfolios are, and which names drive the active weights."
    )

    stock_labels, w_bench, w_opt, te_used = load_sector_weights_at_target_te(
        sector_name, period, target_te_bps=target_te_bps
    )

    # Concentration metrics
    hhi_b, n_eff_b = compute_concentration_metrics(w_bench)
    hhi_d, n_eff_d = compute_concentration_metrics(w_opt)

    col1, col2 = st.columns(2)

    with col1:
        info_metric(
            "Benchmark concentration (HHI)",
            f"{hhi_b:.4f}",
            "HHI is a simple concentration index: sum of squared weights. Higher means more concentrated."
        )
        info_metric(
            "Benchmark effective holdings",
            f"{n_eff_b:.1f}",
            "Effective number of holdings = 1/HHI. Lower means concentration in fewer names."
        )

    with col2:
        info_metric(
            "Decarbonized concentration (HHI)",
            f"{hhi_d:.4f}",
            "Same concentration measure for the optimized decarbonized portfolio at the selected TE."
        )
        info_metric(
            "Decarbonized effective holdings",
            f"{n_eff_d:.1f}",
            "Effective number of holdings = 1/HHI. Useful to compare diversification vs the benchmark."
        )

    # Weight differences table
    df = pd.DataFrame({
        "Symbol": stock_labels,
        "Benchmark weight": w_bench,
        "Decarbonized weight": w_opt,
    })

    df["Active weight (Δw)"] = df["Decarbonized weight"] - df["Benchmark weight"]
    df["|Δw|"] = df["Active weight (Δw)"].abs()

    # Optional: show only meaningful movers by default
    show_top = st.checkbox("Show only top active movers", value=True, key=f"top_movers_{sector_name}_{period}")
    top_n = st.slider("Top N", min_value=5, max_value=50, value=20, step=5, key=f"topn_{sector_name}_{period}")

    df_show = df.sort_values("|Δw|", ascending=False)
    if show_top:
        df_show = df_show.head(top_n)

    # Formatting for display
    df_disp = df_show.drop(columns=["|Δw|"]).copy()
    for c in ["Benchmark weight", "Decarbonized weight", "Active weight (Δw)"]:
        df_disp[c] = pd.to_numeric(df_disp[c], errors="coerce")

    st.dataframe(
        df_disp.style.format({
            "Benchmark weight": "{:.2%}",
            "Decarbonized weight": "{:.2%}",
            "Active weight (Δw)": "{:+.2%}",
        }),
        use_container_width=True,
        hide_index=True
    )

    explanation_box(
        "Composition",
        f"""
- **Concentration (HHI):** Summarizes how much weight is concentrated in a few names (higher = more concentrated).  
- **Effective holdings (1/HHI):** An intuitive “diversification equivalent”: how many equally-weighted names would give the same concentration.  
- **Active weight (Δw):** The per-stock difference between the decarbonized portfolio and the benchmark. This shows *which names the optimizer tilts up or down* to achieve carbon reduction at about **{te_used:.0f} bps** tracking error.
"""
    )



