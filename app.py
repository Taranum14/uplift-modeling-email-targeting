import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(
    page_title="Uplift targeting | Hillstrom email campaign",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Palette and styling
# ----------------------------------------------------------------------------
INK, MUTED, LINE = "#1B1F24", "#6B7280", "#E5E4DE"
ACCENT, AMBER, NEG, GREY = "#0F6B5C", "#C8873A", "#B04A3A", "#A8ABB2"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }

h1 { font-size: 1.65rem !important; font-weight: 600 !important; letter-spacing: -0.02em; padding-bottom: 0.1rem; }
h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; }
.sub { color: #6B7280; font-size: 0.95rem; margin: 0 0 1.4rem 0; max-width: 760px; line-height: 1.5; }

section[data-testid="stSidebar"] { background: #F3F2EC; border-right: 1px solid #E5E4DE; }
section[data-testid="stSidebar"] h2 { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7280; }

.stTabs [data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid #E5E4DE; }
.stTabs [data-baseweb="tab"] { padding: 0.55rem 0; height: auto; font-weight: 500; color: #6B7280; }
.stTabs [aria-selected="true"] { color: #1B1F24; }

.kpi { border: 1px solid #E5E4DE; border-radius: 6px; padding: 14px 16px 12px 16px; background: #FFFFFF; height: 100%; }
.kpi-l { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: #6B7280; margin-bottom: 6px; }
.kpi-v { font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 500; color: #1B1F24; line-height: 1.15; }
.kpi-v.pos { color: #0F6B5C; } .kpi-v.neg { color: #B04A3A; }
.kpi-s { font-size: 0.78rem; color: #6B7280; margin-top: 6px; line-height: 1.35; }

.sec { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7280;
       border-bottom: 1px solid #E5E4DE; padding-bottom: 6px; margin: 2rem 0 0.9rem 0; }
.callout { border-left: 3px solid #0F6B5C; background: #FFFFFF; border-top: 1px solid #E5E4DE;
           border-right: 1px solid #E5E4DE; border-bottom: 1px solid #E5E4DE;
           padding: 14px 18px; border-radius: 0 6px 6px 0; line-height: 1.55; margin: 1.2rem 0 0.4rem 0; }
.callout.warn { border-left-color: #C8873A; }
.note { color: #6B7280; font-size: 0.8rem; line-height: 1.5; margin-top: 0.4rem; }
.stDownloadButton button { border-radius: 4px; border: 1px solid #1B1F24; background: #1B1F24; color: #FFFFFF; font-weight: 500; }
.stDownloadButton button:hover { background: #0F6B5C; border-color: #0F6B5C; color: #FFFFFF; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def money(x, d=0):
    s = f"{abs(x):,.{d}f}"
    return f"\u2212${s}" if x < 0 else f"${s}"


def kpi(label, value, sub="", tone=""):
    return (
        f'<div class="kpi"><div class="kpi-l">{label}</div>'
        f'<div class="kpi-v {tone}">{value}</div><div class="kpi-s">{sub}</div></div>'
    )


def section(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def style(fig, height=380, xtitle=None, ytitle=None):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE, zeroline=False, title=xtitle)
    fig.update_yaxes(gridcolor="#ECEBE5", zeroline=False, title=ytitle)
    return fig


# ----------------------------------------------------------------------------
# Data and derived tables (all cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_scored():
    return pd.read_parquet(BASE / "data" / "test_scored.parquet")


@st.cache_resource
def load_model():
    booster = lgb.Booster(model_file=str(BASE / "models" / "spend_slearner.txt"))
    cols = json.loads((BASE / "models" / "columns.json").read_text())
    return booster, cols


@st.cache_data
def base_curves(steps=40):
    """Observed incremental spend per customer among the top-f share, ranked by model score."""
    d = load_scored()
    order = np.argsort(-d["uplift_spend"].values, kind="stable")
    sp = d["spend"].values[order].astype(float)
    tr = d["treatment"].values[order].astype(float)
    cs_t, cs_c = np.cumsum(sp * tr), np.cumsum(sp * (1 - tr))
    n_t, n_c = np.cumsum(tr), np.cumsum(1 - tr)
    n = len(d)

    fr = np.linspace(0.05, 1.0, steps)
    idx = (fr * n).astype(int) - 1
    inc = cs_t[idx] / n_t[idx] - cs_c[idx] / n_c[idx]

    k = np.linspace(0.01, 1.0, 200)
    kidx = np.maximum((k * n).astype(int) - 1, 0)
    gain = (cs_t[kidx] / n_t[kidx] - cs_c[kidx] / n_c[kidx]) * (kidx + 1)

    ate = sp[tr == 1].mean() - sp[tr == 0].mean()
    return fr, inc, k, gain, float(ate)


@st.cache_data
def decile_table():
    d = load_scored()
    r = d["uplift_spend"].rank(method="first", ascending=False)
    d = d.assign(decile=pd.qcut(r, 10, labels=range(1, 11)))
    rows = []
    for k, g in d.groupby("decile", observed=True):
        t, c = g[g.treatment == 1]["spend"], g[g.treatment == 0]["spend"]
        rows.append(
            dict(decile=int(k), inc=t.mean() - c.mean(),
                 se=np.sqrt(t.var() / len(t) + c.var() / len(c)), n=len(g))
        )
    return pd.DataFrame(rows)


@st.cache_data
def advantage_grid(list_size):
    fr, inc, *_ = base_curves()
    costs = np.round(np.linspace(0.02, 0.50, 25), 3)
    margins = np.round(np.linspace(0.10, 0.70, 25), 3)
    z = np.zeros((len(margins), len(costs)))
    for i, m in enumerate(margins):
        for j, c in enumerate(costs):
            p = (inc * m - c) * fr * list_size
            z[i, j] = max(p.max(), 0) - max(p[-1], 0)
    return costs, margins, z


d = load_scored()
fr, inc, k_gain, gain, ate_spend = base_curves()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("## Assumptions")
margin = st.sidebar.slider("Gross margin on spend", 0.05, 0.80, 0.30, 0.01, format="%.2f")
cost = st.sidebar.slider("Cost per email ($)", 0.01, 0.60, 0.20, 0.01, format="%.2f")
list_size = st.sidebar.number_input("Customer list size", 10_000, 5_000_000, 100_000, 10_000)
st.sidebar.markdown(
    '<div class="note">Margin and cost are assumptions, not measured values. '
    "Every profit figure on this page responds to them.</div>",
    unsafe_allow_html=True,
)

# Policy economics at the chosen assumptions
p_up = (inc * margin - cost) * fr * list_size
p_rd = (ate_spend * margin - cost) * fr * list_size
best_i = int(np.argmax(p_up))
p_all = p_up[-1]
break_even = ate_spend * margin

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Who should receive the email?")
st.markdown(
    '<p class="sub">An uplift analysis of the Hillstrom randomized email campaign '
    "(64,000 customers). Instead of asking who is likely to buy, it asks for whom the email "
    "changes the outcome, then converts that into a profit decision.</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Targeting economics", "Model diagnostics", "Score a customer", "Method"]
)

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Visit rate lift", "+6.1 pts", "10.6% to 16.7%. 95% CI 5.5 to 6.6 pts"), unsafe_allow_html=True)
    c2.markdown(kpi("Extra spend per customer", money(ate_spend, 2), f"Held-out test set, n = {len(d):,}"), unsafe_allow_html=True)
    c3.markdown(kpi("Break-even cost per email", f"${break_even:.3f}", f"Incremental spend x {margin:.0%} margin"), unsafe_allow_html=True)
    tone = "pos" if p_up[best_i] > 0 else "neg"
    c4.markdown(
        kpi("Best targeted profit", money(p_up[best_i]),
            f"Email top {fr[best_i]:.0%} of a {list_size:,} list", tone),
        unsafe_allow_html=True,
    )

    if best_i == len(fr) - 1:
        verdict = (f"At <b>${cost:.2f}</b> per email and a <b>{margin:.0%}</b> margin, the email pays for itself "
                   f"on almost every customer. <b>Email the full list.</b> Ranking customers adds nothing here.")
        cls = "callout"
    elif p_up[best_i] <= 0:
        verdict = (f"At <b>${cost:.2f}</b> per email, no share of the list is profitable on these assumptions. "
                   f"<b>Do not run the campaign</b>, or reduce the cost per contact.")
        cls = "callout warn"
    else:
        verdict = (f"At <b>${cost:.2f}</b> per email, emailing everyone returns <b>{money(p_all)}</b>. "
                   f"Emailing only the top <b>{fr[best_i]:.0%}</b> by predicted uplift returns "
                   f"<b>{money(p_up[best_i])}</b>, against <b>{money(p_rd[best_i])}</b> for the same volume "
                   f"picked at random.")
        cls = "callout"
    st.markdown(f'<div class="{cls}">{verdict}</div>', unsafe_allow_html=True)

    section("What the analysis found")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
**The experiment is sound.** Treated and control groups were balanced on every observed
feature, so differences in outcome can be attributed to the email.

**The average effect is large, the variation is small.** Visit lift sits between 5 and 8 points
across channel, tenure, region and past spend. There are no obvious pockets of persuadables
and no sleeping dogs.
"""
        )
    with b:
        st.markdown(
            """
**Modelling finds modest signal.** Across eight random splits the S-learner was the most
stable model. The T-learner's early lead on a single split did not survive repetition.

**Targeting matters only when contact is costly.** Below the break-even cost, mailing everyone
is optimal. Above it, a small high-uplift slice stays profitable while a blanket send loses money.
"""
        )

# ----------------------------------------------------------------------------
# Targeting economics
# ----------------------------------------------------------------------------
with tab2:
    section("Expected profit by share of list emailed")
    fig = go.Figure()
    fig.add_scatter(x=fr, y=p_up, name="Uplift model (S-learner)", line=dict(color=ACCENT, width=2.5))
    fig.add_scatter(x=fr, y=p_rd, name="Random targeting", line=dict(color=AMBER, width=2, dash="dash"))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    fig.add_scatter(
        x=[fr[best_i]], y=[p_up[best_i]], mode="markers+text", showlegend=False,
        marker=dict(color=ACCENT, size=10, line=dict(color="white", width=2)),
        text=[f"  {fr[best_i]:.0%} of list  {money(p_up[best_i])}"], textposition="top right",
        textfont=dict(size=12, color=INK),
    )
    style(fig, 400, "Share of list emailed, highest predicted uplift first", "Expected profit ($)")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        '<div class="note">Profit uses observed treated-versus-control spend in the held-out test set, '
        "scaled to your list size. The best share is chosen on the same data it is measured on, so the "
        "peak is slightly optimistic. Small shares of the list rest on few customers and are noisy.</div>",
        unsafe_allow_html=True,
    )

    section("Where does targeting beat the simple alternative?")
    costs, margins, z = advantage_grid(list_size)
    hm = go.Figure(
        go.Heatmap(
            x=costs, y=margins, z=z, colorscale=[[0, "#FAFAF7"], [1, ACCENT]],
            colorbar=dict(title="$", thickness=10, len=0.9, outlinewidth=0),
            hovertemplate="Cost $%{x:.2f}<br>Margin %{y:.0%}<br>Gain $%{z:,.0f}<extra></extra>",
        )
    )
    hm.add_scatter(
        x=[cost], y=[margin], mode="markers", showlegend=False,
        marker=dict(color=INK, size=11, symbol="x-thin", line=dict(color=INK, width=2.5)),
    )
    style(hm, 400, "Cost per email ($)", "Gross margin")
    hm.update_yaxes(tickformat=".0%")
    st.plotly_chart(hm, width="stretch")
    st.markdown(
        '<div class="note">Shading shows extra profit from the best targeted policy compared with the '
        "better of emailing everyone or nobody. The pale band along the left is where blanket emailing "
        "is already optimal. The cross marks the current sidebar assumptions.</div>",
        unsafe_allow_html=True,
    )

    section("Export the target list")
    share = st.slider("Share of list to target", 0.05, 1.0, float(fr[best_i]), 0.05, format="%.2f")
    k = int(len(d) * share)
    out = d.sort_values("uplift_spend", ascending=False).head(k).copy()
    out.insert(0, "rank", range(1, k + 1))
    out["expected_margin"] = (out["uplift_spend"] * margin).round(3)
    out["action"] = np.where(out["expected_margin"] > cost, "target", "review")
    show = out.drop(columns=["treatment", "visit", "spend"]).rename(columns={"uplift_spend": "pred_incremental_spend"})
    st.dataframe(show.head(8).round(3), hide_index=True, width="stretch")
    st.download_button("Download CSV", show.to_csv(index=False).encode(), "target_list.csv", "text/csv")
    st.markdown(
        '<div class="note">This list comes from the held-out test customers, as a demonstration of the '
        "output format. In production you would score your own customer file.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Model diagnostics
# ----------------------------------------------------------------------------
with tab3:
    section("Observed uplift by score decile")
    dt = decile_table()
    fig = go.Figure(
        go.Bar(
            x=dt["decile"], y=dt["inc"], marker_color=ACCENT,
            error_y=dict(type="data", array=1.96 * dt["se"], color=GREY, thickness=1.2, width=3),
            hovertemplate="Decile %{x}<br>Incremental spend $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=ate_spend, line_dash="dash", line_color=AMBER,
                  annotation_text=f"Average effect {money(ate_spend, 2)}", annotation_position="top right")
    style(fig, 360, "Score decile (1 = highest predicted uplift)", "Incremental spend per customer ($)")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        '<div class="note">Bars are measured treated-minus-control differences, with 95% intervals. '
        "A working model would slope downward from decile 1 to 10. Wide intervals that overlap the "
        "average line mean individual deciles cannot be told apart reliably.</div>",
        unsafe_allow_html=True,
    )

    section("Cumulative incremental spend (gain curve)")
    n = len(d)
    fig = go.Figure()
    fig.add_scatter(x=k_gain * n, y=gain, name="Uplift model", line=dict(color=ACCENT, width=2.5))
    fig.add_scatter(x=[0, n], y=[0, ate_spend * n], name="Random targeting",
                    line=dict(color=AMBER, width=2, dash="dash"))
    style(fig, 360, "Customers targeted, highest predicted uplift first", "Cumulative incremental spend ($)")
    st.plotly_chart(fig, width="stretch")

    section("Model comparison across 8 random splits")
    comp = pd.DataFrame(
        {
            "Model": ["Response model", "S-learner", "T-learner", "Class transform"],
            "Mean": [0.0176, 0.0296, 0.0158, 0.0138],
            "Std": [0.0099, 0.0050, 0.0091, 0.0062],
            "Min": [0.0060, 0.0214, 0.0020, 0.0037],
            "Max": [0.0333, 0.0383, 0.0270, 0.0227],
        }
    )
    fig = go.Figure(
        go.Bar(
            x=comp["Model"], y=comp["Mean"],
            marker_color=[GREY, ACCENT, GREY, GREY],
            error_y=dict(type="data", array=comp["Std"], color=INK, thickness=1.2, width=4),
            hovertemplate="%{x}<br>Mean Qini AUC %{y:.4f}<extra></extra>",
        )
    )
    style(fig, 320, None, "Qini AUC (mean \u00b1 1 std)")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    c2.dataframe(comp, hide_index=True, width="stretch")
    st.markdown(
        '<div class="note">Higher is better and random targeting is near zero. The S-learner is the most '
        "consistent, but its range still overlaps the response model's, so the improvement is modest and "
        "not statistically conclusive.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Score a customer
# ----------------------------------------------------------------------------
with tab4:
    section("Customer profile")
    booster, cols = load_model()
    a, b = st.columns(2)
    recency = a.slider("Months since last purchase", 1, 12, 6)
    history = a.number_input("Past spend ($)", 0.0, 3000.0, 250.0)
    hist_seg = a.selectbox(
        "Past spend segment",
        ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
         "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"],
    )
    mens = b.checkbox("Bought men's items", True)
    womens = b.checkbox("Bought women's items", False)
    newbie = b.checkbox("New customer", False)
    zip_code = b.selectbox("Zip code type", ["Surburban", "Urban", "Rural"])
    channel = b.selectbox("Channel", ["Web", "Phone", "Multichannel"])

    def clean(s):
        return re.sub(r"[^0-9a-zA-Z_]", "_", s)

    row = dict.fromkeys(cols, 0)
    row.update({"recency": recency, "history": history, "mens": int(mens),
                "womens": int(womens), "newbie": int(newbie)})
    for key in (clean(f"history_segment_{hist_seg}"), clean(f"zip_code_{zip_code}"), clean(f"channel_{channel}")):
        if key in row:
            row[key] = 1
    X1 = pd.DataFrame([row])[cols]
    X0 = X1.copy()
    X1["treat"], X0["treat"] = 1, 0
    lo, hi = d["uplift_spend"].min(), d["uplift_spend"].max()
    pred = float(np.clip(booster.predict(X1)[0] - booster.predict(X0)[0], lo, hi))
    gain_m = pred * margin
    pct = float((d["uplift_spend"] < pred).mean())

    section("Recommendation")
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Predicted extra spend", money(pred, 2), "From receiving the email"), unsafe_allow_html=True)
    c2.markdown(kpi("Expected margin", money(gain_m, 2), f"Versus ${cost:.2f} email cost"), unsafe_allow_html=True)
    c3.markdown(kpi("Uplift percentile", f"{pct:.0%}", "Among held-out customers"), unsafe_allow_html=True)
    if gain_m > cost:
        st.markdown(f'<div class="callout"><b>Target.</b> Expected margin of {money(gain_m, 2)} exceeds the {money(cost, 2)} cost of the email.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="callout warn"><b>Do not target.</b> Expected margin of {money(gain_m, 2)} is below the {money(cost, 2)} cost of the email.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Individual predictions are noisy, especially for customers unlike most of the '
        "training data. Use them to rank customers, not as exact forecasts.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Method
# ----------------------------------------------------------------------------
with tab5:
    section("Approach")
    st.markdown(
        """
1. **Data.** Hillstrom MineThatData email campaign: 64,000 customers randomly assigned to a
   men's email, a women's email, or no email. Both emails are pooled into one treatment arm.
2. **Randomization check.** Treated and control groups compared on recency, past spend, product
   history and customer type. All differences were negligible.
3. **Outcomes.** Website visit was used for model comparison because conversion is too sparse
   (about 1%). Spend was used for the profit analysis because it is measured in dollars.
4. **Models.** A response model baseline and three uplift learners (S, T and class transformation)
   with gradient boosted trees. Compared with Qini AUC over eight random stratified splits.
5. **Profit.** Incremental spend per customer is estimated by an S-learner, ranked, and checked
   against the observed treated-versus-control difference in the held-out set.
"""
    )
    section("Limitations")
    st.markdown(
        """
- Margin and email cost are assumptions. The conclusions depend on them, which is why they are sliders.
- The best share of the list is selected on the same test data used to measure profit.
- Spend has a heavy tail. A few large purchases drive much of the apparent variation between customers.
- One campaign, with outcomes observed over two weeks. Long-term effects are not captured.
- Pooling the men's and women's emails hides any difference between them.
"""
    )
    section("Possible extensions")
    st.markdown(
        """
- Separate analysis of the men's and women's emails as a multi-treatment problem.
- A validation split for choosing the targeted share, with profit reported on untouched data.
- The Criteo uplift dataset, to test whether heterogeneity is stronger at larger scale.
"""
    )