"""
G10 FX Regime Detector
======================
Uses a Gaussian Hidden Markov Model (HMM) to classify the macro environment
into distinct regimes, then shows which G10 FX trades have historically
outperformed in each regime.

Supports two data modes:
  📊 Market Indicators — daily VIX, equities, rates, gold, FX vol (via Yahoo Finance)
  🏛️ Macro Indicators  — monthly CPI, unemployment, NFP, industrial production,
                          Fed Funds, yield curve, consumer sentiment (via FRED API)

Built by Kavish Shetty | August 2026
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmm import GaussianHMM
import plotly.graph_objects as go
import plotly.express as px
import requests as req
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="G10 FX Regime Detector",
    page_icon="🌐",
    layout="wide",
)

# G10 FX pairs and their yfinance tickers
FX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
}

USD_SIGN = {
    "EUR/USD": -1, "GBP/USD": -1, "USD/JPY": 1,
    "AUD/USD": -1, "NZD/USD": -1, "USD/CAD": 1, "USD/CHF": 1,
}

# ── Market-mode tickers ──
MACRO_TICKERS = {
    "VIX": "^VIX", "SPX": "^GSPC", "US10Y": "^TNX", "Gold": "GC=F",
}

# ── FRED series for macro mode ──
FRED_SERIES = {
    "CPI":             "CPIAUCSL",
    "Unemployment":    "UNRATE",
    "NFP":             "PAYEMS",
    "Industrial Prod": "INDPRO",
    "Fed Funds":       "FEDFUNDS",
    "Yield Curve":     "T10Y2Y",
    "Consumer Sent":   "UMCSENT",
}

# Regime colours
REGIME_COLOURS = {
    "Risk-On": "#27ae60", "Carry / Calm": "#2ecc71",
    "Transitional": "#f39c12", "Tightening": "#3498db",
    "Risk-Off": "#e74c3c",
}

REGIME_LABEL_SETS = {
    2: ["Risk-On", "Risk-Off"],
    3: ["Risk-On", "Transitional", "Risk-Off"],
    4: ["Risk-On", "Carry / Calm", "Tightening", "Risk-Off"],
}

# Feature display names
MARKET_FEATURE_NAMES = {
    "VIX_level": "VIX Level",
    "VIX_momentum": "VIX 5d Chg",
    "SPX_return": "S&P 500 21d Ret (%)",
    "rate_change": "10Y Yield 21d Chg (bps)",
    "gold_return": "Gold 21d Ret (%)",
    "usd_return": "USD Index 21d Ret (%)",
    "fx_vol": "G10 FX Realised Vol (%)",
}

MACRO_FEATURE_NAMES = {
    "cpi_yoy": "CPI YoY (%)",
    "unemployment": "Unemployment (%)",
    "nfp_change": "NFP MoM Chg (k)",
    "ip_yoy": "Ind Prod YoY (%)",
    "fed_funds": "Fed Funds (%)",
    "yield_curve": "10Y-2Y Spread (%)",
    "sentiment": "Consumer Sentiment",
}


# ─────────────────────────────────────────────────────────────
# DATA: MARKET MODE
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    all_tickers = list(FX_PAIRS.values()) + list(MACRO_TICKERS.values())
    raw = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw
    ticker_to_name = {v: k for k, v in {**FX_PAIRS, **MACRO_TICKERS}.items()}
    close.columns = [ticker_to_name.get(c, c) for c in close.columns]
    return close.sort_index()


def compute_market_features(data: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    feats = pd.DataFrame(index=data.index)
    feats["VIX_level"] = data["VIX"]
    feats["VIX_momentum"] = data["VIX"].diff(5)
    feats["SPX_return"] = data["SPX"].pct_change(lookback) * 100
    feats["rate_change"] = data["US10Y"].diff(lookback)
    feats["gold_return"] = data["Gold"].pct_change(lookback) * 100

    usd_comps = pd.DataFrame()
    for pair in FX_PAIRS:
        if pair in data.columns:
            usd_comps[pair] = data[pair].pct_change(lookback) * USD_SIGN[pair]
    feats["usd_return"] = usd_comps.mean(axis=1) * 100

    fx_vols = pd.DataFrame()
    for pair in FX_PAIRS:
        if pair in data.columns:
            fx_vols[pair] = data[pair].pct_change().rolling(lookback).std() * np.sqrt(252) * 100
    feats["fx_vol"] = fx_vols.mean(axis=1)

    feats.dropna(inplace=True)
    return feats


# ─────────────────────────────────────────────────────────────
# DATA: MACRO MODE (FRED API)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def download_macro_data(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download monthly macro indicators from FRED."""
    frames = {}
    for name, series_id in FRED_SERIES.items():
        resp = req.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
                "frequency": "m",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            continue
        obs = resp.json().get("observations", [])
        if not obs:
            continue
        s = pd.DataFrame(obs)
        s["date"] = pd.to_datetime(s["date"])
        s["value"] = pd.to_numeric(s["value"], errors="coerce")
        frames[name] = s.set_index("date")["value"]

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames).sort_index().ffill().dropna()
    return df


def compute_macro_features(data: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=data.index)
    feats["cpi_yoy"] = data["CPI"].pct_change(12) * 100          # YoY inflation
    feats["unemployment"] = data["Unemployment"]                   # level
    feats["nfp_change"] = data["NFP"].diff(1)                      # MoM change (thousands)
    feats["ip_yoy"] = data["Industrial Prod"].pct_change(12) * 100 # YoY growth
    feats["fed_funds"] = data["Fed Funds"]                         # level
    feats["yield_curve"] = data["Yield Curve"]                     # 10Y-2Y spread
    feats["sentiment"] = data["Consumer Sent"]                     # level
    feats.dropna(inplace=True)
    return feats


@st.cache_data(ttl=3600, show_spinner=False)
def download_fx_monthly(start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily FX from yfinance and resample to month-end."""
    tickers = list(FX_PAIRS.values())
    raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw
    ticker_to_name = {v: k for k, v in FX_PAIRS.items()}
    close.columns = [ticker_to_name.get(c, c) for c in close.columns]
    return close.resample("ME").last().dropna()


# ─────────────────────────────────────────────────────────────
# MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fit_hmm(
    features_values: np.ndarray,
    features_index: tuple,
    features_columns: tuple,
    n_regimes: int = 3,
    random_state: int = 42,
) -> tuple:
    features = pd.DataFrame(
        features_values,
        index=pd.DatetimeIndex(features_index),
        columns=list(features_columns),
    )
    feat_mean = features.mean()
    feat_std = features.std()
    scaled = (features - feat_mean) / feat_std

    model = GaussianHMM(
        n_components=n_regimes, covariance_type="full",
        n_iter=100, random_state=random_state, tol=0.01,
    )
    model.fit(scaled.values)
    states = model.predict(scaled.values)
    probs = model.predict_proba(scaled.values)
    return model.transmat_, model.means_, states, probs


def label_regimes(
    features: pd.DataFrame, states: np.ndarray, n_regimes: int,
    sort_col: str = "VIX_level",
) -> dict:
    """
    Auto-label regimes.  For market mode, sort by VIX (high VIX = Risk-Off).
    For macro mode, sort by unemployment (high unemployment = Risk-Off).
    """
    col_means = {}
    for i in range(n_regimes):
        mask = states == i
        col_means[i] = features.loc[mask, sort_col].mean()

    # Sort ascending — lowest value gets first label
    ordered = sorted(col_means, key=col_means.get)

    labels = REGIME_LABEL_SETS[n_regimes]
    # Lowest sort_col → Risk-On … Highest → Risk-Off
    return {ordered[i]: labels[i] for i in range(n_regimes)}


def compute_fx_returns(fx_data: pd.DataFrame, fwd_periods: int = 21) -> pd.DataFrame:
    returns = pd.DataFrame(index=fx_data.index)
    for pair in FX_PAIRS:
        if pair in fx_data.columns:
            returns[pair] = fx_data[pair].pct_change(fwd_periods).shift(-fwd_periods) * 100
    returns.dropna(inplace=True)
    return returns


def fx_performance_by_regime(
    fx_returns: pd.DataFrame, features: pd.DataFrame,
    states: np.ndarray, label_map: dict, n_regimes: int,
) -> pd.DataFrame:
    common = features.index.intersection(fx_returns.index)
    aligned_states = pd.Series(states, index=features.index).loc[common]
    aligned_returns = fx_returns.loc[common].copy()
    aligned_returns["Regime"] = [label_map[s] for s in aligned_states]
    perf = aligned_returns.groupby("Regime").mean()
    order = REGIME_LABEL_SETS[n_regimes]
    return perf.reindex([o for o in order if o in perf.index]).round(3)


# ─────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────

def regime_timeline_chart(dates, states, label_map, price_series, series_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=price_series.values, mode="lines",
        line=dict(color="#2c3e50", width=1.5), name=series_name,
    ))
    labels = [label_map[s] for s in states]
    prev, start = labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != prev or i == len(labels) - 1:
            fig.add_vrect(x0=dates[start], x1=dates[i],
                          fillcolor=REGIME_COLOURS.get(prev, "#ccc"), opacity=0.18, line_width=0)
            start, prev = i, labels[i]
    for lab, col in REGIME_COLOURS.items():
        if lab in label_map.values():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(size=10, color=col), name=lab, showlegend=True))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20),
                      xaxis_title="", yaxis_title=series_name,
                      title=dict(text=f"Regime Timeline — {series_name}", font_size=16),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def regime_characteristics_table(features, states, label_map, n_regimes, feat_names):
    fl = features.copy()
    fl["Regime"] = [label_map[s] for s in states]
    summary = fl.groupby("Regime").mean().rename(columns=feat_names)
    counts = fl["Regime"].value_counts(normalize=True) * 100
    summary["Time in Regime (%)"] = counts
    order = REGIME_LABEL_SETS[n_regimes]
    return summary.reindex([o for o in order if o in summary.index]).round(2)


def transition_matrix_df(transmat, label_map, n_regimes):
    order = REGIME_LABEL_SETS[n_regimes]
    l2s = {v: k for k, v in label_map.items()}
    labs = [l for l in order if l in l2s]
    sts = [l2s[l] for l in labs]
    return (pd.DataFrame(transmat[np.ix_(sts, sts)], index=labs, columns=labs) * 100).round(1)


def regime_duration_stats(states, label_map, dates):
    labels = [label_map[s] for s in states]
    durations = {lab: [] for lab in set(labels)}
    cur, run = labels[0], 1
    for i in range(1, len(labels)):
        if labels[i] == cur:
            run += 1
        else:
            durations[cur].append(run)
            cur, run = labels[i], 1
    durations[cur].append(run)
    records = []
    for lab, runs in durations.items():
        records.append({"Regime": lab, "Avg Duration": int(np.mean(runs)),
                        "Max Duration": int(np.max(runs)), "Occurrences": len(runs)})
    return pd.DataFrame(records).set_index("Regime")


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.markdown(
        "<h1 style='margin-bottom:0'>🌐 G10 FX Regime Detector</h1>"
        "<p style='color:grey; margin-top:0; font-size:1.05rem'>"
        "Hidden Markov Model regime classification across G10 currencies</p>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("Parameters")

        mode = st.radio(
            "Data source",
            ["📊 Market Indicators", "🏛️ Macro Indicators"],
            help=(
                "**Market**: daily VIX, equities, rates, gold, FX vol (Yahoo Finance).  \n"
                "**Macro**: monthly CPI, unemployment, NFP, industrial production, "
                "Fed Funds, yield curve, consumer sentiment (FRED)."
            ),
        )
        is_macro = "Macro" in mode

        # FRED API key (macro mode only)
        fred_key = None
        if is_macro:
            try:
                fred_key = st.secrets["FRED_API_KEY"]
            except (KeyError, FileNotFoundError):
                fred_key = st.text_input(
                    "FRED API key",
                    type="password",
                    help="Free — register at fred.stlouisfed.org/docs/api/api_key.html",
                )

        n_regimes = st.selectbox("Number of regimes", [2, 3, 4], index=1,
                                 help="How many distinct states to detect. 3 is a good default.")

        default_start = datetime(2000, 1, 1) if is_macro else datetime(2015, 1, 1)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=default_start, min_value=datetime(1990, 1, 1))
        with col2:
            end_date = st.date_input("End date", value=datetime.today())

        overlay_pair = st.selectbox("FX pair for timeline", list(FX_PAIRS.keys()), index=0)

        if not is_macro:
            lookback = st.slider("Feature lookback (trading days)", 10, 63, 21,
                                 help="Window for trailing returns and vol. 21 ≈ 1 month.")
        else:
            lookback = 1  # monthly — 1-period forward returns

        st.markdown("---")
        if is_macro:
            st.markdown(
                "**How it works**\n\n"
                "The model ingests seven monthly macro indicators from FRED — "
                "CPI, unemployment, NFP, industrial production, Fed Funds rate, "
                "yield curve slope, and consumer sentiment — and infers which "
                "business-cycle regime is most likely.\n\n"
                "Regimes are auto-labelled by average unemployment: "
                "lowest → Risk-On, highest → Risk-Off."
            )
        else:
            st.markdown(
                "**How it works**\n\n"
                "The model observes seven market features — VIX, equity returns, "
                "rate changes, USD strength, gold, and FX vol — and infers which "
                "hidden macro regime is most likely.\n\n"
                "Regimes are auto-labelled by average VIX: lowest → Risk-On, "
                "highest → Risk-Off."
            )

    # ── Load data & compute features ──
    if is_macro:
        if not fred_key:
            st.info(
                "🔑 To use Macro mode, enter a free FRED API key in the sidebar.  \n"
                "Get one in 30 seconds at "
                "[fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)."
            )
            return

        with st.spinner("Downloading macro data from FRED..."):
            macro_data = download_macro_data(fred_key, str(start_date), str(end_date))
        if macro_data.empty or len(macro_data) < 24:
            st.error("Not enough macro data. Check your API key and date range.")
            return

        features = compute_macro_features(macro_data)
        feat_names = MACRO_FEATURE_NAMES
        sort_col = "unemployment"
        freq_label = "months"

        with st.spinner("Downloading FX data..."):
            fx_data = download_fx_monthly(str(start_date), str(end_date))

    else:
        with st.spinner("Downloading market data..."):
            market_data = download_market_data(str(start_date), str(end_date))
        if market_data.empty or len(market_data) < 100:
            st.error("Not enough data. Try a wider date range.")
            return

        features = compute_market_features(market_data, lookback=lookback)
        feat_names = MARKET_FEATURE_NAMES
        sort_col = "VIX_level"
        freq_label = "days"
        fx_data = market_data  # daily, already contains FX columns

    if len(features) < 30:
        st.error("Not enough valid observations after feature computation.")
        return

    # ── Fit HMM ──
    with st.spinner("Fitting Hidden Markov Model..."):
        try:
            transmat, hmm_means, states, probs = fit_hmm(
                features_values=features.values,
                features_index=tuple(features.index),
                features_columns=tuple(features.columns),
                n_regimes=n_regimes,
            )
        except Exception as e:
            st.error(f"HMM failed to converge. Try a different date range. ({e})")
            return

    label_map = label_regimes(features, states, n_regimes, sort_col=sort_col)

    # ── Current regime metrics ──
    current_regime = label_map[states[-1]]
    current_prob = probs[-1][states[-1]] * 100
    periods_in_regime = 1
    for i in range(len(states) - 2, -1, -1):
        if states[i] == states[-1]:
            periods_in_regime += 1
        else:
            break
    switch_prob = (1 - transmat[states[-1], states[-1]]) * 100

    st.markdown("### Current Assessment")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Regime", current_regime)
    m2.metric("Confidence", f"{current_prob:.0f}%")
    m3.metric(f"{freq_label.title()} in Regime", periods_in_regime)
    m4.metric("Switch Probability", f"{switch_prob:.1f}%",
              help=f"Probability of transitioning to a different regime next {freq_label[:-1]}")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Timeline", "📊 Regime Profile", "💱 FX Performance", "🔄 Transitions"]
    )

    # ── Tab 1: Timeline ──
    with tab1:
        if overlay_pair in fx_data.columns:
            if is_macro:
                price = fx_data[overlay_pair].reindex(features.index, method="ffill").dropna()
                common_idx = features.index.intersection(price.index)
                price = price.loc[common_idx]
                timeline_states = pd.Series(states, index=features.index).loc[common_idx].values
            else:
                price = fx_data[overlay_pair].loc[features.index]
                common_idx = features.index
                timeline_states = states

            fig = regime_timeline_chart(common_idx, timeline_states, label_map, price, overlay_pair)
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning(f"No data for {overlay_pair}.")

        # Probability stacked area
        st.markdown("#### Regime Probabilities Over Time")
        prob_df = pd.DataFrame(probs, index=features.index,
                               columns=[label_map[i] for i in range(n_regimes)])
        fig_prob = go.Figure()
        for col in prob_df.columns:
            fig_prob.add_trace(go.Scatter(
                x=prob_df.index, y=prob_df[col], mode="lines", stackgroup="one",
                name=col, line=dict(width=0), fillcolor=REGIME_COLOURS.get(col, "#ccc"),
            ))
        fig_prob.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                               yaxis_title="Probability", yaxis=dict(range=[0, 1]),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_prob, width='stretch')

    # ── Tab 2: Regime Profile ──
    with tab2:
        st.markdown("#### Average Feature Values by Regime")
        char_table = regime_characteristics_table(features, states, label_map, n_regimes, feat_names)
        st.dataframe(char_table, width='stretch')

        # Radar chart
        st.markdown("#### Regime Fingerprints")
        radar_feats = [feat_names[f] for f in feat_names]
        char_normed = char_table[radar_feats].copy()
        for col in char_normed.columns:
            rng = char_normed[col].max() - char_normed[col].min()
            char_normed[col] = ((char_normed[col] - char_normed[col].min()) / rng) if rng > 0 else 0.5

        fig_radar = go.Figure()
        for regime in char_normed.index:
            vals = char_normed.loc[regime].tolist() + [char_normed.loc[regime].iloc[0]]
            cats = list(char_normed.columns) + [char_normed.columns[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, name=regime,
                line=dict(color=REGIME_COLOURS.get(regime, "#555")), fill="toself", opacity=0.5,
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                                height=450, margin=dict(l=60, r=60, t=40, b=40))
        st.plotly_chart(fig_radar, width='stretch')

        # Per-regime correlation matrices
        st.markdown("#### Feature Correlations by Regime")
        st.caption("Cross-asset correlations shift across regimes — compare matrices "
                   "to see which co-movements are regime-dependent.")
        fl = features.copy()
        fl["_regime"] = [label_map[s] for s in states]
        corr_cols = st.columns(n_regimes)
        order = REGIME_LABEL_SETS[n_regimes]
        for idx, regime in enumerate([r for r in order if r in label_map.values()]):
            with corr_cols[idx]:
                st.markdown(f"**{regime}**")
                subset = fl[fl["_regime"] == regime].drop(columns=["_regime"])
                corr = subset.rename(columns=feat_names).corr().round(2)
                short_labels = [c.split(" ")[0] for c in corr.columns]
                fig_c = go.Figure(data=go.Heatmap(
                    z=corr.values, x=short_labels, y=short_labels,
                    colorscale="RdBu_r", zmin=-1, zmax=1,
                    text=corr.values, texttemplate="%{text:.1f}", textfont=dict(size=9),
                    showscale=(idx == n_regimes - 1),
                ))
                fig_c.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    xaxis=dict(tickangle=45))
                st.plotly_chart(fig_c, width='stretch')

    # ── Tab 3: FX Performance ──
    with tab3:
        fwd_label = "1-Month" if is_macro else f"{lookback}-Day"
        st.markdown(f"#### Average {fwd_label} Forward Return by Regime (%)")
        st.caption("Average forward return for each G10 pair conditional on the regime at entry. "
                   "Positive = long the quoted pair was profitable.")

        fwd_periods = 1 if is_macro else lookback
        fx_ret = compute_fx_returns(fx_data, fwd_periods=fwd_periods)

        if not fx_ret.empty:
            perf = fx_performance_by_regime(fx_ret, features, states, label_map, n_regimes)
            st.dataframe(perf, width='stretch')

            fig_bar = go.Figure()
            for regime in perf.index:
                fig_bar.add_trace(go.Bar(
                    x=perf.columns, y=perf.loc[regime], name=regime,
                    marker_color=REGIME_COLOURS.get(regime, "#555"),
                ))
            fig_bar.update_layout(barmode="group", height=400,
                                  margin=dict(l=20, r=20, t=40, b=20),
                                  yaxis_title="Avg Fwd Return (%)",
                                  title=f"G10 FX {fwd_label} Returns by Regime",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_bar, width='stretch')
        else:
            st.warning("Insufficient data for forward returns.")

    # ── Tab 4: Transitions ──
    with tab4:
        period_word = "month" if is_macro else "day"
        st.markdown("#### Transition Probability Matrix (%)")
        st.caption(f"Each row shows the probability of moving to each regime next {period_word}, "
                   "given today's regime. Diagonal = probability of staying.")
        trans = transition_matrix_df(transmat, label_map, n_regimes)
        st.dataframe(trans, width='stretch')

        st.markdown("#### Regime Duration Statistics")
        dur = regime_duration_stats(states, label_map, features.index)
        dur = dur.rename(columns={
            "Avg Duration": f"Avg Duration ({freq_label})",
            "Max Duration": f"Max Duration ({freq_label})",
        })
        order = REGIME_LABEL_SETS[n_regimes]
        dur = dur.reindex([o for o in order if o in dur.index])
        st.dataframe(dur, width='stretch')

    # ── Footer ──
    st.markdown("---")
    if is_macro:
        st.caption(
            "**Methodology:** Gaussian Hidden Markov Model (Hamilton, 1989) fitted to "
            "seven monthly macro indicators from FRED.  Regimes auto-labelled by average "
            "unemployment rate.  Model re-fits on each parameter change.  "
            "This is an analytical tool, not investment advice."
        )
    else:
        st.caption(
            "**Methodology:** Gaussian Hidden Markov Model (Hamilton, 1989) fitted to "
            "seven daily cross-asset features via Yahoo Finance.  Regimes auto-labelled "
            "by average VIX.  Model re-fits on each parameter change.  "
            "This is an analytical tool, not investment advice."
        )


if __name__ == "__main__":
    main()
