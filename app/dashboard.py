import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

from app.stats import (
    load_requests as _load_requests,
    load_pricing as _load_pricing,
    estimate_baseline_cost,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "requests.db"
MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
ROUTING_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "routing.yaml"

TIER_ORDER = ["simple", "moderate", "complex"]
TIER_COLORS = {"simple": "#38bdf8", "moderate": "#8b5cf6", "complex": "#ec4899"}
CARD_BG = "#141928"
RING_TRACK = "#242a3d"
GRID_COLOR = "#232a3d"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
CHART_CONFIG = {"displayModeBar": False}

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide", initial_sidebar_state="expanded")

pio.templates["autopilot"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY, size=13, color="#cbd5e1"),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        colorway=["#38bdf8", "#8b5cf6", "#ec4899", "#22c55e", "#f97316"],
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showline=False),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12), bgcolor="#1e2436", bordercolor=GRID_COLOR),
        margin=dict(t=20, b=10, l=10, r=20),
    )
)
PLOTLY_TEMPLATE = "autopilot"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: {FONT_FAMILY};
    }}
    .main .block-container {{padding-top: 1.5rem; max-width: 1400px;}}
    h1 {{
        color: #f8fafc;
        font-weight: 800;
        letter-spacing: -0.02em;
        font-size: 2.1rem;
    }}
    [data-testid="stCaptionContainer"] {{color: #94a3b8;}}
    [data-testid="stPlotlyChart"] {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #1e2436;
        transition: box-shadow 0.15s ease;
    }}
    [data-testid="stPlotlyChart"]:hover {{
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    }}
    .chart-heading {{
        font-family: {FONT_FAMILY};
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        letter-spacing: -0.01em;
        margin: 2px 0 8px 4px;
    }}
    .ring-wrap {{display: flex; justify-content: center;}}
    div[data-baseweb="tab-list"] {{gap: 4px;}}
    button[data-baseweb="tab"] {{
        background-color: {CARD_BG};
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        font-weight: 500;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #f8fafc;
        border-bottom: 2px solid #8b5cf6;
        font-weight: 600;
    }}
    .filter-chip {{
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid #8b5cf6;
        color: #c4b5fd;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def chart_heading(text: str) -> None:
    """Renders the chart's title as plain Streamlit markup, entirely
    outside the Plotly figure. Plotly's own title + legend positioning
    interact unpredictably (that's what caused the overlap) -- keeping the
    heading out of the figure's coordinate system avoids that class of bug
    for good."""
    st.markdown(f'<div class="chart-heading">{text}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=15)
def load_requests() -> pd.DataFrame:
    return _load_requests()


@st.cache_data(ttl=60)
def load_pricing() -> dict:
    return _load_pricing()


def render_kpi_cards(cards: list[dict]) -> None:
    card_html = ""
    for c in cards:
        card_html += f"""
        <div class="kpi-card" style="--c1:{c['c1']};--c2:{c['c2']}">
            <div class="kpi-label">{c['label']}</div>
            <div class="kpi-value" data-target="{c['value']}"
                 data-prefix="{c.get('prefix', '')}" data-suffix="{c.get('suffix', '')}"
                 data-decimals="{c.get('decimals', 0)}">0</div>
            <div class="kpi-sub">{c.get('sub', '')}</div>
        </div>
        """
    html = f"""
    <div class="kpi-row">{card_html}</div>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        .kpi-row {{display: flex; flex-wrap: wrap; gap: 14px; font-family: {FONT_FAMILY};}}
        .kpi-card {{
            flex: 1; min-width: 150px;
            background: linear-gradient(135deg, var(--c1), var(--c2));
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        }}
        .kpi-label {{
            color: rgba(255,255,255,0.85); font-size: 0.74rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em;
        }}
        .kpi-value {{
            color: #ffffff; font-size: 2rem; font-weight: 800; margin-top: 6px;
            font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
        }}
        .kpi-sub {{color: rgba(255,255,255,0.75); font-size: 0.78rem; margin-top: 2px; font-weight: 500;}}
    </style>
    <script>
        document.querySelectorAll('.kpi-value').forEach(el => {{
            const target = parseFloat(el.dataset.target);
            const decimals = parseInt(el.dataset.decimals || "0");
            const prefix = el.dataset.prefix || "";
            const suffix = el.dataset.suffix || "";
            const start = performance.now();
            const duration = 800;
            function tick(now) {{
                const progress = Math.min((now - start) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
                if (progress < 1) requestAnimationFrame(tick);
            }}
            requestAnimationFrame(tick);
        }});
    </script>
    """
    st.components.v1.html(html, height=150, scrolling=False)


def render_ring(value: float, label: str, color: str, display_text: str = None, max_value: float = 100):
    remainder = max(max_value - value, 0)
    fig = go.Figure(data=[go.Pie(
        values=[value, remainder],
        hole=0.74,
        marker=dict(colors=[color, RING_TRACK], line=dict(color=CARD_BG, width=2)),
        textinfo="none",
        sort=False,
        direction="clockwise",
        rotation=90,
        # Keeps the ring a small centered circle regardless of the column's
        # actual width, instead of fighting Streamlit's container sizing
        # with a fixed-width chart (that's what caused the broken borders).
        domain=dict(x=[0.28, 0.72], y=[0.06, 0.94]),
    )])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=210,
        annotations=[
            dict(text=f"<b>{display_text or f'{value:.1f}%'}</b>", x=0.5, y=0.56,
                 font=dict(size=27, color="#f8fafc", family=FONT_FAMILY), showarrow=False),
            dict(text=label, x=0.5, y=0.32, font=dict(size=12, color="#94a3b8", family=FONT_FAMILY), showarrow=False),
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


df_all = load_requests()
pricing = load_pricing()

st.title("LLM Cost Autopilot")
st.caption("Routing, cost, and quality overview — reads live from data/requests.db")

if df_all.empty:
    st.info("No requests logged yet. Run scripts/replay_traffic.py or send some live traffic first.")
    st.stop()

df_all["baseline_cost"] = estimate_baseline_cost(df_all, pricing)

st.sidebar.header("Controls")
live_mode = st.sidebar.toggle("Live mode (auto-refresh every 1s)", value=False)
if live_mode:
    st_autorefresh(interval=1000, key="live_refresh")

if st.sidebar.button("Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Filters")

min_date = df_all["timestamp"].min().date()
max_date = df_all["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
tiers_selected = st.sidebar.multiselect("Tiers", TIER_ORDER, default=TIER_ORDER)
providers_selected = st.sidebar.multiselect(
    "Providers", sorted(df_all["provider"].unique()), default=sorted(df_all["provider"].unique())
)

start, end = date_range if len(date_range) == 2 else (min_date, max_date)

df = df_all[
    (df_all["timestamp"].dt.date >= start)
    & (df_all["timestamp"].dt.date <= end)
    & (df_all["tier"].isin(tiers_selected))
    & (df_all["provider"].isin(providers_selected))
].copy()

if df.empty:
    st.warning("No requests match the current filters.")
    st.stop()

st.sidebar.caption(f"{len(df):,} of {len(df_all):,} requests match the current filters.")

if "selected_tier" not in st.session_state:
    st.session_state.selected_tier = None

if st.session_state.selected_tier:
    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.markdown(f'<span class="filter-chip">Filtered to tier: {st.session_state.selected_tier}</span>', unsafe_allow_html=True)
    with col_b:
        if st.button("Clear"):
            st.session_state.selected_tier = None
            st.rerun()

active_df = df if not st.session_state.selected_tier else df[df["tier"] == st.session_state.selected_tier]

total_actual = active_df["total_cost"].sum()
total_baseline = active_df["baseline_cost"].sum()
saved = total_baseline - total_actual
saved_pct = (saved / total_baseline * 100) if total_baseline > 0 else 0
verified_df = active_df[active_df["verified"] == 1]
escalation_rate = (verified_df["escalated"].mean() * 100) if len(verified_df) else 0
avg_score = verified_df["judge_score"].dropna().mean() if verified_df["judge_score"].notna().any() else 0

render_kpi_cards([
    {"label": "Requests", "value": len(active_df), "c1": "#38bdf8", "c2": "#6366f1"},
    {"label": "Actual Cost", "value": total_actual, "prefix": "$", "decimals": 4, "c1": "#8b5cf6", "c2": "#d946ef"},
    {"label": "Baseline Cost", "value": total_baseline, "prefix": "$", "decimals": 4, "c1": "#334155", "c2": "#1e293b", "sub": "always top-tier"},
    {"label": "Saved", "value": saved, "prefix": "$", "decimals": 4, "c1": "#22c55e", "c2": "#0ea5e9", "sub": f"{saved_pct:.1f}% reduction"},
    {"label": "Escalation Rate", "value": escalation_rate, "suffix": "%", "decimals": 1, "c1": "#f97316", "c2": "#ef4444"},
    {"label": "Avg Judge Score", "value": avg_score, "suffix": "/5", "decimals": 2, "c1": "#ec4899", "c2": "#f97316"},
])

with st.expander("How is 'Baseline Cost' calculated?"):
    st.write(
        "Baseline cost is what these requests would have cost if every one had been "
        "routed straight to the top-tier model for its provider. Escalated requests "
        "already reflect top-tier pricing exactly. For everything else, actual cost is "
        "scaled by the ratio of per-token prices between the routed model and the "
        "top-tier model, since per-request token counts aren't logged (only a prompt "
        "hash is stored, to keep the log privacy-conscious)."
    )

st.divider()

tab_overview, tab_quality, tab_raw = st.tabs(["Overview", "Quality & Escalation", "Raw Log"])

with tab_overview:
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        render_ring(saved_pct, "Cost savings vs. top-tier", "#22c55e")
    with r1c2:
        render_ring(escalation_rate, "Escalation rate", "#f97316")
    with r1c3:
        score_pct = (avg_score / 5 * 100) if avg_score else 0
        render_ring(score_pct, "Avg judge score", "#ec4899", display_text=f"{avg_score:.2f}/5")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        chart_heading("Routing distribution")
        tier_counts = df["tier"].value_counts().reindex(TIER_ORDER).fillna(0)
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index, values=tier_counts.values, hole=0.58,
            customdata=list(tier_counts.index),
            marker=dict(colors=[TIER_COLORS[t] for t in tier_counts.index], line=dict(color=CARD_BG, width=2)),
            textinfo="label+percent", textfont=dict(color="#f8fafc", family=FONT_FAMILY, size=12.5),
        )])
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, height=340)
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points",
                                 key="tier_donut", config=CHART_CONFIG)
        try:
            points = event["selection"]["points"]
            if points:
                clicked_tier = points[0]["customdata"][0]
                if clicked_tier != st.session_state.selected_tier:
                    st.session_state.selected_tier = clicked_tier
                    st.rerun()
        except Exception:
            pass

    with col2:
        chart_heading("Cumulative cost over time")
        span = df["timestamp"].max() - df["timestamp"].min()
        freq = "h" if span < pd.Timedelta(days=1) else "D"
        over_time = active_df.set_index("timestamp").resample(freq)[["total_cost", "baseline_cost"]].sum().cumsum()

        fig = go.Figure()
        fig.add_scatter(
            x=over_time.index, y=over_time["total_cost"], mode="lines",
            line=dict(color="#38bdf8", width=3, shape="spline", smoothing=0.4),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.14)", showlegend=False,
        )
        fig.add_scatter(
            x=over_time.index, y=over_time["baseline_cost"], mode="lines",
            line=dict(color="#94a3b8", width=2, dash="dot", shape="spline", smoothing=0.4), showlegend=False,
        )
        # Label each line directly at its endpoint instead of using a legend,
        # so there's nothing that can ever collide with anything else.
        last_x = over_time.index[-1]
        fig.add_annotation(x=last_x, y=over_time["total_cost"].iloc[-1], text="Actual", showarrow=False,
                            xanchor="left", xshift=10, font=dict(color="#38bdf8", family=FONT_FAMILY, size=12))
        fig.add_annotation(x=last_x, y=over_time["baseline_cost"].iloc[-1], text="Baseline", showarrow=False,
                            xanchor="left", xshift=10, font=dict(color="#94a3b8", family=FONT_FAMILY, size=12))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=20, b=10, l=10, r=70))
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col3, col4 = st.columns([1, 1])

    with col3:
        chart_heading("Request volume by hour of day")
        heat = active_df.groupby(["tier", "hour"]).size().unstack(fill_value=0).reindex(TIER_ORDER).fillna(0)
        fig = go.Figure(data=go.Heatmap(
            z=heat.values, x=heat.columns, y=heat.index, colorscale="Purples", showscale=False,
            xgap=3, ygap=3,
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, xaxis_title="Hour")
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    with col4:
        chart_heading("Escalation rate by tier")
        esc_by_tier = active_df.groupby("tier").apply(
            lambda g: (g["escalated"].sum() / g["verified"].sum() * 100) if g["verified"].sum() else 0,
            include_groups=False,
        ).reindex(TIER_ORDER).fillna(0)
        fig = go.Figure(data=[go.Bar(
            x=esc_by_tier.index, y=esc_by_tier.values,
            marker_color=[TIER_COLORS[t] for t in esc_by_tier.index],
            text=[f"{v:.0f}%" for v in esc_by_tier.values], textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=13, color="#e2e8f0"),
            width=0.55,
        )])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=30, b=30, l=10, r=10),
                           yaxis_title="% of verified")
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    chart_heading("Cost breakdown: provider → tier → model")
    cost_tree = active_df.groupby(["provider", "tier", "model_id"], as_index=False)["total_cost"].sum()
    fig = px.sunburst(cost_tree, path=["provider", "tier", "model_id"], values="total_cost",
                       color="tier", color_discrete_map={**TIER_COLORS, "(?)": "#242a3d"})
    fig.update_traces(textfont=dict(family=FONT_FAMILY, size=13))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

with tab_quality:
    col1, col2 = st.columns(2)
    with col1:
        chart_heading("Judge score distribution")
        scored = verified_df["judge_score"].dropna()
        if len(scored):
            fig = px.histogram(scored, x="judge_score", nbins=5, color_discrete_sequence=["#8b5cf6"])
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340,
                               xaxis_title="Score (1-5)", yaxis_title="Count", bargap=0.2)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        else:
            st.info("No verified requests in the current filter to score.")

    with col2:
        chart_heading("Escalation rate over time")
        esc_over_time = (
            active_df[active_df["verified"] == 1].set_index("timestamp").resample(freq)
            .agg(escalated=("escalated", "sum"), verified=("verified", "sum"))
        )
        esc_over_time["rate"] = (esc_over_time["escalated"] / esc_over_time["verified"] * 100).fillna(0)
        fig = go.Figure(data=[go.Scatter(
            x=esc_over_time.index, y=esc_over_time["rate"], mode="lines",
            line=dict(color="#ec4899", width=3, shape="spline", smoothing=0.4),
            fill="tozeroy", fillcolor="rgba(236,72,153,0.14)",
        )])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340, yaxis_title="Escalation rate (%)")
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

with tab_raw:
    st.dataframe(
        active_df.sort_values("timestamp", ascending=False)[
            ["timestamp", "provider", "tier", "model_id", "total_cost", "verified", "judge_score", "escalated"]
        ],
        use_container_width=True, height=500,
    )
    st.download_button("Download filtered data as CSV", active_df.to_csv(index=False).encode("utf-8"),
                        file_name="requests_filtered.csv", mime="text/csv")