"""
G10 FX Regime Detector
======================
Uses a Gaussian Hidden Markov Model (HMM) to classify the macro environment
into distinct regimes based on observable market indicators, then shows
which G10 FX trades have historically outperformed in each regime.

Built by Kavish Shetty
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Sign convention for USD index:
#   +1 → pair is quoted as USD/X, so a rise = USD strengthening
#   -1 → pair is quoted as X/USD, so a rise = USD weakening (invert)
USD_SIGN = {
    "EUR/USD": -1,
    "GBP/USD": -1,
    "USD/JPY": 1,
    "AUD/USD": -1,
    "NZD/USD": -1,
    "USD/CAD": 1,
    "USD/CHF": 1,
}

# Macro indicators
MACRO_TICKERS = {
    "VIX": "^VIX",
    "SPX": "^GSPC",
    "US10Y": "^TNX",
    "Gold": "GC=F",
}

# Regime colour palette (consistent across all charts)
REGIME_COLOURS = {
    "Risk-On": "#27ae60",
    "Carry / Calm": "#2ecc71",
    "Transitional": "#f39c12",
    "Tightening": "#3498db",
    "Risk-Off": "#e74c3c",
}

# How to label regimes depending on how many the user picks
REGIME_LABEL_SETS = {
    2: ["Risk-On", "Risk-Off"],
    3: ["Risk-On", "Transitional", "Risk-Off"],
    4: ["Risk-On", "Carry / Calm", "Tightening", "Risk-Off"],
}

# Feature display names (for charts and tables)
FEATURE_NAMES = {
    "VIX_level": "VIX Level",
    "VIX_momentum": "VIX 5d Chg",
    "SPX_return": "S&P 500 21d Ret (%)",
    "rate_change": "10Y Yield 21d Chg (bps)",
    "gold_return": "Gold 21d Ret (%)",
    "usd_return": "USD Index 21d Ret (%)",
    "fx_vol": "G10 FX Realised Vol (%)",
}


# ─────────────────────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def download_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily close prices for all FX pairs and macro indicators.
    Returns a single DataFrame with readable column names.
    """
    all_tickers = list(FX_PAIRS.values()) + list(MACRO_TICKERS.values())
    raw = yf.download(all_tickers, start=start_date, end=end_date, progress=False)

    # yfinance returns MultiIndex columns: (Price, Ticker).  Grab Close prices.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw

    # Build a reverse map: ticker → readable name
    ticker_to_name = {}
    for name, ticker in {**FX_PAIRS, **MACRO_TICKERS}.items():
        ticker_to_name[ticker] = name

    close.columns = [ticker_to_name.get(c, c) for c in close.columns]
    close = close.sort_index()
    return close


def compute_features(data: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """
    Derive the seven observable features the HMM will use to infer regimes.
    Each feature is computed from trailing windows so the model only uses
    information available on each date (no look-ahead).
    """
    feats = pd.DataFrame(index=data.index)

    # 1. VIX level — raw gauge of implied equity volatility / fear
    feats["VIX_level"] = data["VIX"]

    # 2. VIX 5-day change — is fear rising or falling?
    feats["VIX_momentum"] = data["VIX"].diff(5)

    # 3. S&P 500 trailing return — equity risk appetite over the past month
    feats["SPX_return"] = data["SPX"].pct_change(lookback) * 100

    # 4. 10-year yield change — direction of rates
    feats["rate_change"] = data["US10Y"].diff(lookback)

    # 5. Gold trailing return — safe-haven demand
    feats["gold_return"] = data["Gold"].pct_change(lookback) * 100

    # 6. USD basket return — computed from the G10 crosses, sign-adjusted
    #    so that positive = USD strengthening
    usd_components = pd.DataFrame()
    for pair in FX_PAIRS:
        if pair in data.columns:
            ret = data[pair].pct_change(lookback)
            usd_components[pair] = ret * USD_SIGN[pair]
    feats["usd_return"] = usd_components.mean(axis=1) * 100

    # 7. Average G10 FX realised volatility — how choppy are currencies?
    fx_vols = pd.DataFrame()
    for pair in FX_PAIRS:
        if pair in data.columns:
            fx_vols[pair] = (
                data[pair].pct_change().rolling(lookback).std() * np.sqrt(252) * 100
            )
    feats["fx_vol"] = fx_vols.mean(axis=1)

    feats.dropna(inplace=True)
    return feats


# ─────────────────────────────────────────────────────────────
# MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────


def fit_hmm(
    features: pd.DataFrame, n_regimes: int = 3, random_state: int = 42
) -> tuple:
    """
    Fit a Gaussian HMM to the standardised feature matrix.

    Returns
    -------
    model       : fitted GaussianHMM
    states      : array of regime labels per date
    probs       : array of regime probabilities per date (n_dates × n_regimes)
    feat_mean   : Series of feature means (for un-standardising)
    feat_std    : Series of feature stds
    """
    feat_mean = features.mean()
    feat_std = features.std()
    scaled = (features - feat_mean) / feat_std

    # Try fitting with several seeds; pick the best log-likelihood
    best_score = -np.inf
    best_model = None

    for seed in [random_state, random_state + 1, random_state + 7]:
        try:
            m = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=500,
                random_state=seed,
                tol=0.01,
            )
            m.fit(scaled.values)
            score = m.score(scaled.values)
            if score > best_score:
                best_score = score
                best_model = m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM failed to converge.  Try a different date range.")

    states = best_model.predict(scaled.values)
    probs = best_model.predict_proba(scaled.values)

    return best_model, states, probs, feat_mean, feat_std


def label_regimes(
    features: pd.DataFrame, states: np.ndarray, n_regimes: int
) -> dict:
    """
    Automatically label each hidden state by sorting on average VIX.
    Lowest VIX → Risk-On, highest VIX → Risk-Off.
    """
    vix_means = {}
    for i in range(n_regimes):
        mask = states == i
        vix_means[i] = features.loc[mask, "VIX_level"].mean()

    # Sort states from lowest average VIX to highest
    ordered = sorted(vix_means, key=vix_means.get)

    labels = REGIME_LABEL_SETS[n_regimes]
    return {state_id: labels[rank] for rank, state_id in enumerate(ordered)}


def compute_fx_returns(data: pd.DataFrame, fwd_days: int = 21) -> pd.DataFrame:
    """
    Compute forward returns for each FX pair (used to evaluate regime
    performance — i.e. if you entered a position at the start of each
    regime, what would your next-month return have been?).
    """
    returns = pd.DataFrame(index=data.index)
    for pair in FX_PAIRS:
        if pair in data.columns:
            returns[pair] = data[pair].pct_change(fwd_days).shift(-fwd_days) * 100
    returns.dropna(inplace=True)
    return returns


# ─────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────


def regime_timeline_chart(
    dates: pd.DatetimeIndex,
    states: np.ndarray,
    label_map: dict,
    price_series: pd.Series,
    series_name: str = "EUR/USD",
) -> go.Figure:
    """Overlay regime-coloured bands on an FX price chart."""
    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=price_series.values,
            mode="lines",
            line=dict(color="#2c3e50", width=1.5),
            name=series_name,
        )
    )

    # Regime bands as vertical rectangles
    labels = [label_map[s] for s in states]
    prev_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != prev_label or i == len(labels) - 1:
            fig.add_vrect(
                x0=dates[start_idx],
                x1=dates[i],
                fillcolor=REGIME_COLOURS.get(prev_label, "#cccccc"),
                opacity=0.18,
                line_width=0,
            )
            start_idx = i
            prev_label = labels[i]

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title=series_name,
        title=dict(text=f"Regime Timeline — {series_name}", font_size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def regime_characteristics_table(
    features: pd.DataFrame, states: np.ndarray, label_map: dict, n_regimes: int
) -> pd.DataFrame:
    """Average feature values per regime."""
    features_labelled = features.copy()
    features_labelled["Regime"] = [label_map[s] for s in states]

    summary = features_labelled.groupby("Regime").mean()
    summary = summary.rename(columns=FEATURE_NAMES)

    # Add regime frequency
    counts = features_labelled["Regime"].value_counts(normalize=True) * 100
    summary["Time in Regime (%)"] = counts

    # Order by risk level
    order = REGIME_LABEL_SETS[n_regimes]
    summary = summary.reindex([o for o in order if o in summary.index])
    return summary.round(2)


def fx_performance_by_regime(
    fx_returns: pd.DataFrame,
    features: pd.DataFrame,
    states: np.ndarray,
    label_map: dict,
    n_regimes: int,
) -> pd.DataFrame:
    """
    Average forward FX returns per regime.  Shows which pairs historically
    outperform in each macro environment.
    """
    # Align dates
    common = features.index.intersection(fx_returns.index)
    aligned_states = pd.Series(states, index=features.index).loc[common]
    aligned_returns = fx_returns.loc[common]

    aligned_returns["Regime"] = [label_map[s] for s in aligned_states]
    perf = aligned_returns.groupby("Regime").mean()

    order = REGIME_LABEL_SETS[n_regimes]
    perf = perf.reindex([o for o in order if o in perf.index])
    return perf.round(3)


def transition_matrix(model, label_map: dict, n_regimes: int) -> pd.DataFrame:
    """Format the HMM transition matrix with readable labels."""
    order = REGIME_LABEL_SETS[n_regimes]
    state_to_label = label_map
    label_to_state = {v: k for k, v in state_to_label.items()}

    labels_ordered = [l for l in order if l in label_to_state]
    states_ordered = [label_to_state[l] for l in labels_ordered]

    trans = pd.DataFrame(
        model.transmat_[np.ix_(states_ordered, states_ordered)],
        index=labels_ordered,
        columns=labels_ordered,
    )
    return (trans * 100).round(1)


def regime_duration_stats(
    states: np.ndarray, label_map: dict, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Average and max consecutive days spent in each regime."""
    labels = [label_map[s] for s in states]
    durations = {lab: [] for lab in set(labels)}

    current_label = labels[0]
    run_length = 1

    for i in range(1, len(labels)):
        if labels[i] == current_label:
            run_length += 1
        else:
            durations[current_label].append(run_length)
            current_label = labels[i]
            run_length = 1
    durations[current_label].append(run_length)

    records = []
    for lab, runs in durations.items():
        records.append(
            {
                "Regime": lab,
                "Avg Duration (days)": int(np.mean(runs)),
                "Max Duration (days)": int(np.max(runs)),
                "Occurrences": len(runs),
            }
        )
    return pd.DataFrame(records).set_index("Regime")


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────


def main():
    # ── Header ──
    st.markdown(
        """
        <h1 style='margin-bottom:0'>🌐 G10 FX Regime Detector</h1>
        <p style='color:grey; margin-top:0; font-size:1.05rem'>
        Hidden Markov Model regime classification across G10 currencies
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("Parameters")

        n_regimes = st.selectbox(
            "Number of regimes",
            options=[2, 3, 4],
            index=1,
            help="How many distinct market states to detect. 3 is a good default.",
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start date",
                value=datetime(2007, 1, 1),
                min_value=datetime(2003, 1, 1),
            )
        with col2:
            end_date = st.date_input(
                "End date",
                value=datetime.today(),
            )

        overlay_pair = st.selectbox(
            "FX pair for timeline overlay",
            options=list(FX_PAIRS.keys()),
            index=0,
        )

        lookback = st.slider(
            "Feature lookback (trading days)",
            min_value=10,
            max_value=63,
            value=21,
            help="Window for computing trailing returns and vol.  21 ≈ 1 month.",
        )

        st.markdown("---")
        st.markdown(
            """
            **How it works**

            The model observes seven market features — VIX, equity returns,
            rate changes, USD strength, gold, and FX vol — and infers which
            hidden macro regime is most likely producing those observations.

            Regimes are auto-labelled by average VIX: lowest VIX → Risk-On,
            highest → Risk-Off.
            """,
        )

    # ── Load data ──
    with st.spinner("Downloading market data..."):
        data = download_data(str(start_date), str(end_date))

    if data.empty or len(data) < 100:
        st.error("Not enough data for the selected range.  Try a wider window.")
        return

    # ── Compute features ──
    features = compute_features(data, lookback=lookback)
    if len(features) < 60:
        st.error("Not enough valid observations after feature computation.")
        return

    # ── Fit HMM ──
    with st.spinner("Fitting Hidden Markov Model..."):
        try:
            model, states, probs, feat_mean, feat_std = fit_hmm(
                features, n_regimes=n_regimes
            )
        except RuntimeError as e:
            st.error(str(e))
            return

    label_map = label_regimes(features, states, n_regimes)

    # ── Current regime headline ──
    current_regime = label_map[states[-1]]
    current_colour = REGIME_COLOURS.get(current_regime, "#555")
    current_prob = probs[-1][states[-1]] * 100

    # How long have we been in this regime?
    days_in_regime = 1
    for i in range(len(states) - 2, -1, -1):
        if states[i] == states[-1]:
            days_in_regime += 1
        else:
            break

    # Probability of switching away (1 − self-transition probability)
    switch_prob = (1 - model.transmat_[states[-1], states[-1]]) * 100

    # ── Key metrics row ──
    st.markdown("### Current Assessment")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Regime", current_regime)
    m2.metric("Confidence", f"{current_prob:.0f}%")
    m3.metric("Days in Regime", days_in_regime)
    m4.metric("Switch Probability", f"{switch_prob:.1f}%",
              help="Daily probability of transitioning to a different regime")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Timeline", "📊 Regime Profile", "💱 FX Performance", "🔄 Transitions"]
    )

    # ── Tab 1: Regime timeline ──
    with tab1:
        if overlay_pair in data.columns:
            price = data[overlay_pair].loc[features.index]
            fig = regime_timeline_chart(
                features.index, states, label_map, price, overlay_pair
            )

            # Add legend entries for regimes
            for label, colour in REGIME_COLOURS.items():
                if label in label_map.values():
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(size=10, color=colour),
                            name=label,
                            showlegend=True,
                        )
                    )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data for {overlay_pair}.")

        # Regime probability over time
        st.markdown("#### Regime Probabilities Over Time")
        prob_df = pd.DataFrame(probs, index=features.index)
        prob_df.columns = [label_map[i] for i in range(n_regimes)]

        fig_prob = go.Figure()
        for col in prob_df.columns:
            fig_prob.add_trace(
                go.Scatter(
                    x=prob_df.index,
                    y=prob_df[col],
                    mode="lines",
                    stackgroup="one",
                    name=col,
                    line=dict(width=0),
                    fillcolor=REGIME_COLOURS.get(col, "#ccc"),
                )
            )
        fig_prob.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Tab 2: Regime characteristics ──
    with tab2:
        st.markdown("#### Average Feature Values by Regime")
        char_table = regime_characteristics_table(
            features, states, label_map, n_regimes
        )
        st.dataframe(char_table, use_container_width=True)

        # Radar chart of regime profiles
        st.markdown("#### Regime Fingerprints")
        radar_feats = [FEATURE_NAMES[f] for f in FEATURE_NAMES]
        fig_radar = go.Figure()

        char_normed = char_table[radar_feats].copy()
        for col in char_normed.columns:
            rng = char_normed[col].max() - char_normed[col].min()
            if rng > 0:
                char_normed[col] = (char_normed[col] - char_normed[col].min()) / rng
            else:
                char_normed[col] = 0.5

        for regime in char_normed.index:
            vals = char_normed.loc[regime].tolist()
            vals.append(vals[0])  # close the polygon
            cats = list(char_normed.columns) + [char_normed.columns[0]]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=cats,
                    name=regime,
                    line=dict(color=REGIME_COLOURS.get(regime, "#555")),
                    fill="toself",
                    opacity=0.5,
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            height=450,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 3: FX performance ──
    with tab3:
        st.markdown("#### Average 21-Day Forward Return by Regime (%)")
        st.caption(
            "Shows the average next-month return for each G10 pair, conditional "
            "on the regime at entry.  Positive = long the quoted pair was profitable."
        )

        fx_ret = compute_fx_returns(data, fwd_days=lookback)
        if not fx_ret.empty:
            perf = fx_performance_by_regime(
                fx_ret, features, states, label_map, n_regimes
            )
            st.dataframe(
                perf.style.background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )

            # Bar chart
            fig_bar = go.Figure()
            for regime in perf.index:
                fig_bar.add_trace(
                    go.Bar(
                        x=perf.columns,
                        y=perf.loc[regime],
                        name=regime,
                        marker_color=REGIME_COLOURS.get(regime, "#555"),
                    )
                )
            fig_bar.update_layout(
                barmode="group",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Avg Fwd Return (%)",
                title="G10 FX Returns by Regime",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Insufficient data for forward returns.")

    # ── Tab 4: Transitions ──
    with tab4:
        st.markdown("#### Transition Probability Matrix (%)")
        st.caption(
            "Each row shows the probability of moving to each regime tomorrow, "
            "given today's regime.  Diagonal = probability of staying."
        )
        trans = transition_matrix(model, label_map, n_regimes)
        st.dataframe(
            trans.style.background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )

        st.markdown("#### Regime Duration Statistics")
        dur = regime_duration_stats(states, label_map, features.index)
        order = REGIME_LABEL_SETS[n_regimes]
        dur = dur.reindex([o for o in order if o in dur.index])
        st.dataframe(dur, use_container_width=True)

    # ── Footer ──
    st.markdown("---")
    st.caption(
        "**Methodology:** Gaussian Hidden Markov Model (Hamilton, 1989) fitted to "
        "seven cross-asset features.  Regimes auto-labelled by average VIX.  "
        "Data sourced from Yahoo Finance.  Model re-fits on each run; "
        "regime boundaries may shift slightly with new data.  "
        "This is an analytical tool, not investment advice."
    )


if __name__ == "__main__":
    main()
