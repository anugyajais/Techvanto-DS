# streamlit_app/app.py
import streamlit as st
import pandas as pd
import numpy as np

from components import (
    plot_class_distribution,
    plot_amount_overview,
    plot_time_kde,
    plot_fraud_share_by_amount_bin_simple,
    plot_fraud_rate_vs_total_by_amount_bin,
    plot_top_features_kde
)

st.set_page_config(layout="wide", page_title="FraudDetect-AI EDA Dashboard")

# Caching / helpers
@st.cache_data(show_spinner=False)
def load_data(path="../data/creditcard.csv"):
    df = pd.read_csv(path)
    # ensure Class is int and numeric columns are numeric
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(int)
    return df

@st.cache_data
def precompute_amount_bin_counts(df, bins):
    """
    Return a small verification table with totals and fraud counts for given bins.
    Uses a local copy so original df is not mutated (prevents Interval columns leaking into df).
    """
    dfc = df.copy()
    cat = pd.cut(dfc['Amount'], bins=bins, include_lowest=True)
    dfc['AmountBin_temp'] = cat  # temporary column name
    bin_index = dfc['AmountBin_temp'].cat.categories

    total_counts = dfc['AmountBin_temp'].value_counts().sort_index().reindex(bin_index, fill_value=0)
    fraud_counts = dfc[dfc['Class'] == 1]['AmountBin_temp'].value_counts().sort_index().reindex(bin_index, fill_value=0)
    fraud_rate_in_bin = (fraud_counts / total_counts.replace(0, np.nan)).fillna(0) * 100.0
    fraud_share_of_total = (fraud_counts / fraud_counts.sum()).fillna(0) * 100.0

    table = pd.DataFrame({
        "AmountBin": [str(iv) for iv in bin_index],
        "total_txns": total_counts.astype(int).values,
        "fraud_count": fraud_counts.astype(int).values,
        "fraud_rate_pct_in_bin": fraud_rate_in_bin.values,
        "fraud_share_of_total_frauds_pct": fraud_share_of_total.values
    })
    return table

def stratified_sample(df, n, balanced=False, random_state=42):
    """Sample n rows. If balanced True, aim for half fraud / half non-fraud."""
    if n is None or n >= len(df):
        return df.copy()
    if not balanced:
        return df.sample(n=n, random_state=random_state)
    fraud = df[df['Class'] == 1]
    nonfraud = df[df['Class'] == 0]
    n_half = max(1, n // 2)
    n_f = min(len(fraud), n_half)
    n_nf = min(len(nonfraud), n - n_f)
    samp = pd.concat([
        fraud.sample(n=n_f, random_state=random_state) if n_f > 0 else pd.DataFrame(columns=df.columns),
        nonfraud.sample(n=n_nf, random_state=random_state) if n_nf > 0 else pd.DataFrame(columns=df.columns),
    ])
    if len(samp) < n:
        remaining = df.drop(index=samp.index)
        add = remaining.sample(n=(n - len(samp)), random_state=random_state)
        samp = pd.concat([samp, add])
    return samp.sample(frac=1, random_state=random_state)  # shuffle

# -------------------------
# Main app
# -------------------------
def main():
    st.title("💳 Credit Card Fraud Detection — Interactive EDA")
    df = load_data()
    st.markdown("This dashboard provides an interactive way to explore the credit card fraud dataset. Use the sidebar controls to customize your analysis.\n\n")

    # Safety check
    if df is None or df.shape[0] == 0:
        st.error("Failed to load data or dataset is empty.")
        return

    # Sidebar controls
    st.sidebar.header("Controls")
    st.sidebar.subheader("Amount bins (comma-separated)")
    default_bins = [0, 1, 5, 10, 25, 50, 100, 500, 1000, 5000, 30000]
    bins_input = st.sidebar.text_input("Bins", value=",".join(str(x) for x in default_bins))
    # parse bins safely
    try:
        bins = [float(x.strip()) for x in bins_input.split(",") if x.strip() != ""]
        # ensure bins are sorted and cover data range
        bins = sorted(bins)
        if bins[0] > 0:
            bins[0] = 0.0
        # make last bin large enough to include max if not already
        max_amt = float(df['Amount'].max())
        if bins[-1] <= max_amt:
            bins[-1] = max_amt + 1.0
    except Exception:
        st.sidebar.error("Couldn't parse bins. Reverting to default.")
        bins = default_bins

    st.sidebar.subheader("Which plots to display")
    show_class = st.sidebar.checkbox("Class distribution", value=True)
    show_amount_overview = st.sidebar.checkbox("Amount overview (hist + KDE)", value=True)
    show_amount_bin_rate = st.sidebar.checkbox("Transaction count + fraud rate per amount bin", value=True)
    show_fraud_share = st.sidebar.checkbox("Share of all frauds per bin (fraud mass)", value=True)
    show_top_features = st.sidebar.checkbox("Top feature KDEs", value=True)

    st.sidebar.subheader("Top feature settings")
    top_n = st.sidebar.slider("Top-N features (by corr/mutual info)", min_value=3, max_value=15, value=5, step=1)

    # Quick dataset summary
    st.header("Dataset overview")
    st.markdown("This dataset contains credit card transactions labeled as fraudulent or not. The `Class` column indicates fraud (1) or non-fraud (0).")
    st.write(df.head())

    # Basic summary metrics
    dup_count = int(df.duplicated().sum())
    rows = len(df)
    fraud_pct = df['Class'].mean() * 100.0
    st.markdown(f"- Rows: **{rows:,}**  \n- Duplicate rows: **{dup_count}**  \n- Fraud fraction: **{fraud_pct:.5f}%**")
    st.markdown("**Quick note:** The dataset is *very* imbalanced (frauds are a tiny fraction). When you interpret accuracy-based metrics keep this in mind; prioritize precision/recall or PR-AUC for model evaluation.")

    # Create sampled df for heavy plots if requested (currently using full df by default)
    sampled_df = df.copy()

    # -------------------------
    # Plots area
    # -------------------------
    st.markdown("---")
    st.header("Distributions & Correlations")

    # CLASS DISTRIBUTION
    if show_class:
        st.subheader("Class distribution")
        left, space, right = st.columns([2.5, 0.5, 2])
        with left:
            try:
                plot_class_distribution(df)
            except Exception as e:
                st.error(f"plot_class_distribution failed: {e}")
        with right:
            # Metrics stacked vertically
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Fraud %", f"{(df['Class'].mean()*100):.5f}%")
            st.markdown("**Interpretation / Notes**")
            st.markdown(
                "- The left plot uses a **log y-scale** so the tiny fraud bar is visible next to the large non-fraud bar.  \n"
                "- The annotation above each bar shows the raw count and *percentage of total transactions*.  \n"
                "- Because frauds are extremely rare, the raw count is more useful than accuracy for model evaluation; prefer precision/recall or PR-AUC when benchmarking models."
            )

    # AMOUNT OVERVIEW (hist + KDE)
    if show_amount_overview:
        st.subheader("Amount overview")
        try:
            plot_amount_overview(sampled_df)
            # Add a short interpretation
            st.markdown("**Interpretation / Notes**")
            st.markdown(
                "- Left: histogram of **log1p(Amount)** (the x-axis tick labels are shown in original amount units). "
                "We log-transform because transaction amounts are long-tailed.  \n"
                "- Right: KDE of log-amount split by class (Non-Fraud vs Fraud). Look for **shifts** between the two curves — "
                "if fraud density peaks at higher log-amounts than non-fraud, amount may be predictive.  \n"
                "- Beware: `Amount` alone is insufficient; it should be combined with other features and careful sampling/thresholding because of class imbalance."
            )
        except Exception as e:
            st.error(f"plot_amount_overview failed: {e}")

    # AMOUNT-BIN ANALYSIS
    if show_amount_bin_rate or show_fraud_share:
        st.header("Amount bin analysis")
        st.markdown("&nbsp;")
        left, right = st.columns([2, 1])

        with left:
            if show_amount_bin_rate:
                st.subheader("Transaction count and fraud rate per amount bin")
                try:
                    try:
                        plot_fraud_rate_vs_total_by_amount_bin(df, bins=bins)
                    except TypeError:
                        plot_fraud_rate_vs_total_by_amount_bin(df)
                    # add interpretation below the chart
                    st.markdown("**Interpretation / Notes**")
                    st.markdown(
                        "- Bars show **total transactions per amount bin** (log scale). The red line shows **fraud rate** in each bin (percent of transactions in that bin that are fraud).  \n"
                        "- If a bin has a high fraud rate, it means *transactions within that amount range* are relatively more likely to be fraudulent.  \n"
                        "- However, a high fraud rate in a tiny bin with few transactions may still represent few actual frauds; check the fraud-count and fraud-share table to verify absolute numbers."
                    )
                except Exception as e:
                    st.error(f"plot_fraud_rate_vs_total_by_amount_bin failed: {e}")

            if show_fraud_share:
                st.subheader("Share of all frauds per bin")
                try:
                    plot_fraud_share_by_amount_bin_simple(df)
                    st.markdown("**Interpretation / Notes**")
                    st.markdown(
                        "- This plot shows the **share of the dataset's total frauds** that fall into each bin (i.e., treat all frauds as 100%).  \n"
                        "- Use this to answer: *which amount ranges contain most fraud cases in absolute terms?* — complements the fraud-rate plot which is a per-bin relative rate."
                    )
                except Exception as e:
                    st.error(f"plot_fraud_share_by_amount_bin_simple failed: {e}")

        with right:
            # small verification table (cached)
            try:
                st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)  # add vertical space
                table = precompute_amount_bin_counts(df, bins=bins)
                st.markdown("**Bin summary (quick verification)**")
                st.dataframe(table, use_container_width=True)
                st.markdown(
                    "- `total_txns`: how many transactions fall in this bin.  \n"
                    "- `fraud_count`: how many frauds in this bin (absolute number).  \n"
                    "- `fraud_rate_pct_in_bin`: frauds / total txns in bin (the red line on the chart).  \n"
                    "- `fraud_share_of_total_frauds_pct`: percent of *all frauds* located in this bin (sums to ~100% across bins)."
                )
            except Exception as e:
                st.error(f"precompute_amount_bin_counts failed: {e}")

    # TOP FEATURES (KDEs)
    if show_top_features:
        st.header("Top feature distributions (by correlation + mutual info)")
        numeric_df = df.select_dtypes(include=[np.number])
        if 'Class' not in numeric_df.columns:
            st.warning("No numeric 'Class' column found for feature ranking.")
            top_features_suggested = []
        else:
            corr_series = numeric_df.corr()['Class'].abs().drop('Class').sort_values(ascending=False)
            top_features_suggested = corr_series.head(30).index.tolist()

        left, right = st.columns([3, 1])
        with left:
            # Plot area
            selected = st.session_state.get('selected_top_features', top_features_suggested[:top_n])
            selected_numeric = [c for c in selected if c in numeric_df.columns]
            if len(selected_numeric) == 0:
                st.info("Select at least one numeric feature to plot.")
            else:
                try:
                    plot_top_features_kde(df, selected_numeric)
                    st.markdown("**Interpretation / Notes**")
                    st.markdown(
                        "- The features V1..V28 are PCA components (not directly interpretable). We still plot them because they may be predictive.  \n"
                        "- `Correlation` vs `Mutual Information`: correlation captures linear association; mutual information finds non-linear relevance. Consider both when selecting features.  \n"
                        "- Use these KDEs to see whether the feature distributions differ across classes — separation suggests predictive power."
                    )
                except Exception as e:
                    st.error(f"plot_top_features_kde failed: {e}")
        with right:
            # Selection and metrics stacked vertically
            selected = st.multiselect("Pick features to plot (suggested)", options=top_features_suggested, default=top_features_suggested[:top_n], key='selected_top_features')
            st.metric("Numeric features", f"{len(numeric_df.columns)}")
            st.metric("Top-N suggested", f"{len(top_features_suggested)}")
            st.markdown(
                "- Tip: If a KDE shows the fraud and non-fraud lines largely overlapping, that feature alone is weak — combine features or use model-based feature importance (e.g., SHAP) to rank."
            )

    # Footer / final notes
    st.markdown("---")
    st.markdown("**Important modeling notes & next steps**")
    st.markdown(
        "- The dataset is heavily imbalanced — when training models prefer metrics like **Precision @ X, Recall, F1 (class-specific)** and **Area under Precision-Recall curve (PR-AUC)** over raw accuracy.  \n"
        "- Consider resampling (SMOTE) or class-weighted models, but validate results on a holdout set.  \n"
        "- For model interpretability, add SHAP explanations and display example-level explanations in the dashboard.  \n"
        "- Suggested next moves: feature selection (use mutual information + model importance), try tree-based models (XGBoost/LightGBM), track PR-AUC and confusion matrix for chosen thresholds."
    )

if __name__ == "__main__":
    main()
