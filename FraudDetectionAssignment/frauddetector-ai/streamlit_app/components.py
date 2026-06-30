# streamlit_app/components.py
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style='whitegrid')

def plot_class_distribution(df: pd.DataFrame):
    class_counts = df['Class'].value_counts()
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    ax.set_yscale('log')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_title('Class Distribution (Log Scale)')
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = f'{100 * count / total:.5f}%'
        ax.annotate(f'{count}\n({percentage})', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)

def plot_amount_overview(df):
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
    # a) Histogram (log-scale)
    logamt = np.log1p(df['Amount'])
    sns.histplot(logamt, bins=50, kde=False, color='gray', ax=ax1)
    ax1.set_title("Amount Histogram (Log-Scaled)")
    ax1.set_xticklabels([f"{np.expm1(t):.0f}" for t in ax1.get_xticks()])
    # b) KDE by Class
    for cls, label, color in [(0,'Non-Fraud','#E74C3C'), (1,'Fraud','#3498DB')]:
        sns.kdeplot(np.log1p(df[df.Class==cls].Amount),
                    fill=True, common_norm=False, alpha=0.5,
                    label=label, color=color, ax=ax2)
    ax2.set_title("Amount by Class (Log-KDE)")
    ax2.legend()
    st.pyplot(fig)
    
def plot_time_kde(df):
    fig, ax = plt.subplots(figsize=(6,4))
    for cls, label, color in [(0,'Non-Fraud','#E74C3C'), (1,'Fraud','#3498DB')]:
        sns.kdeplot(df[df.Class==cls].Time,
                    fill=True, common_norm=False, alpha=0.5,
                    label=label, color=color, ax=ax)
    ax.set_title("Transaction Time by Class (KDE)")
    ax.legend()
    st.pyplot(fig)

import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Colors (consistent with your other plots)
COLOR_FRAUD = "#3498DB"
COLOR_TOTAL = "#16a085"

def _make_amount_bins(df, bins):
    """Helper: returns the categorical index for bins and ensures consistent ordering."""
    cat = pd.cut(df['Amount'], bins=bins)
    idx = pd.IntervalIndex(cat.cat.categories) if hasattr(cat, 'cat') else cat.index
    return cat, cat.cat.categories

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

def plot_fraud_share_by_amount_bin_simple(df):
    """
    Hard-coded bins. Bars show percent of ALL frauds that fall in each bin (sums ~100%).
    Annotates each bar with percent and the absolute fraud count.
    """
    # hard-coded bins (based on dataset percentiles)
    bins = [0, 1, 5, 10, 25, 50, 100, 500, 1000, 5000, 30000]

    if 'Amount' not in df.columns or 'Class' not in df.columns:
        st.error("DataFrame must contain 'Amount' and 'Class' columns.")
        return

    dfc = df.copy()
    # create bins and ensure ordered index
    dfc['AmountBin'] = pd.cut(dfc['Amount'], bins=bins, include_lowest=True)
    bin_idx = dfc['AmountBin'].cat.categories

    # fraud counts per bin (reindex to keep empty bins)
    fraud_counts = dfc[dfc['Class'] == 1]['AmountBin'].value_counts().sort_index().reindex(bin_idx, fill_value=0).astype(int)
    if fraud_counts.sum() == 0:
        st.write("No frauds found.")
        return

    # percent of total frauds in each bin
    fraud_pct = fraud_counts / fraud_counts.sum() * 100.0

    # plotting
    x = np.arange(len(fraud_pct))
    labels = [str(iv) for iv in fraud_pct.index]

    fig, ax = plt.subplots(figsize=(11,4.5))
    bars = ax.bar(x, fraud_pct.values, color="#c0392b", edgecolor="#0b3d3b", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel("Share of total frauds (%)")
    ax.set_title("Share of ALL Frauds by Amount Bin (total frauds = 100%)")

    # format y axis as percent (0..100)
    ax.set_ylim(0, max(8, fraud_pct.max() * 1.15))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=2))

    # annotate percent and absolute count
    for xi, bar, pct, cnt in zip(x, bars, fraud_pct.values, fraud_counts.values):
        ax.annotate(f"{pct:.2f}%", (xi, pct),
                    textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
        # count inside if bar big, else below percent
        if pct > 8:
            ax.annotate(f"{int(cnt)}", (xi, pct/2), ha='center', va='center', color='white', fontsize=9)
        else:
            ax.annotate(f"{int(cnt)}", (xi, pct), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

def plot_fraud_share_by_amount_bin(df, bins=[0,10,50,100,500,1000,5000,10000,25000], annotate=True):
    """
    Plot the share of ALL frauds that fall in each amount bin.
    Bars = percent of total frauds contained in that bin.
    Optionally annotate with absolute fraud counts.
    """
    df = df.copy()
    df['AmountBin'] = pd.cut(df['Amount'], bins=bins)
    # total frauds
    total_frauds = df.loc[df['Class'] == 1].shape[0]
    if total_frauds == 0:
        st.write("No frauds in dataset.")
        return

    # fraud counts per bin; align to full bin index so empty bins are shown
    fraud_counts = df[df['Class'] == 1]['AmountBin'].value_counts().sort_index()
    # ensure index order = bin order:
    bin_index = pd.IntervalIndex(pd.cut(df['Amount'], bins=bins).cat.categories)
    fraud_counts = fraud_counts.reindex(bin_index, fill_value=0)

    # percent of all frauds in each bin
    fraud_pct = fraud_counts / fraud_counts.sum() * 100

    x = np.arange(len(fraud_pct))
    labels = [str(iv) for iv in fraud_pct.index]

    fig, ax = plt.subplots(figsize=(12,5))
    bars = ax.bar(x, fraud_pct.values, color=COLOR_FRAUD, alpha=0.85)
    ax.set_ylabel("Share of total frauds (%)")
    ax.set_title("Share of ALL Frauds by Amount Bin (total frauds = 100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')

    # percent-format y axis
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))

    # annotate with absolute fraud counts (optional)
    if annotate:
        for xi, bar, cnt in zip(x, bars, fraud_counts.values):
            height = bar.get_height()
            ax.annotate(f"{cnt}\n{height:.2f}%", (xi, height),
                        textcoords="offset points", xytext=(0,6), ha='center', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)


# import numpy as np
# import matplotlib.ticker as mtick

# def plot_amount_bin_rate(df, bins=[0,10,50,100,500,1000,5000,10000,25000]):
#     # create bins and counts
#     df = df.copy()
#     df['AmountBin'] = pd.cut(df['Amount'], bins=bins)
#     bin_counts = df['AmountBin'].value_counts().sort_index()
    
#     # ensure fraud counts align to same index (avoid NaNs)
#     fraud_counts = df[df['Class'] == 1]['AmountBin'].value_counts().reindex(bin_counts.index, fill_value=0)
#     fraud_rate = (fraud_counts / bin_counts).fillna(0) * 100  # percent

#     # numeric x positions for perfect alignment
#     x = np.arange(len(bin_counts))

#     fig, ax1 = plt.subplots(figsize=(12,5))

#     # Bars (log scale)
#     bars = ax1.bar(x, bin_counts.values, color='teal', alpha=0.6)
#     ax1.set_yscale('log')
#     ax1.set_ylabel('Total Transactions (log scale)')
#     ax1.set_ylim(bottom=1)  # avoid log(0) issues

#     # Secondary axis for fraud rate
#     ax2 = ax1.twinx()
#     ax2.plot(x, fraud_rate.values, color='red', marker='o', linewidth=2, label='Fraud Rate (%)')
#     ax2.set_ylabel('Fraud Rate (%)')
#     ax2.yaxis.set_major_formatter(mtick.PercentFormatter())  # shows as % (optionally use .1f)

#     # Ticks and labels
#     labels = [str(interval) for interval in bin_counts.index]
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(labels, rotation=35, ha='right')

#     # Annotate fraud-rate values above the points for clarity
#     for xi, pct in zip(x, fraud_rate.values):
#         ax2.annotate(f"{pct:.3f}%", (xi, pct), textcoords="offset points", xytext=(0,6), ha='center', color='red', fontsize=9)

#     ax1.set_title('Transactions & Fraud Rate by Amount Bin')
#     ax2.legend(loc='upper right')
#     fig.tight_layout()
#     st.pyplot(fig)

# def plot_amount_bin_rate(df, bins=[0,10,50,100,500,1000,5000,10000,25000]):
#     df['AmountBin'] = pd.cut(df.Amount, bins=bins)
#     total = df.AmountBin.value_counts().sort_index()
#     rate  = df[df.Class==1].AmountBin.value_counts().sort_index() / total * 100
#     fig, ax1 = plt.subplots(figsize=(8,4))
#     ax1.bar(total.index.astype(str), total.values, color='teal', alpha=0.5)
#     ax1.set_yscale('log'); ax1.set_ylabel("Total Txns (log)")
#     ax2 = ax1.twinx()
#     ax2.plot(rate.index.astype(str), rate.values, '-o', color='red')
#     ax2.set_ylabel("Fraud Rate (%)")
#     plt.xticks(rotation=45)
#     ax1.set_title("Transactions & Fraud Rate by Amount Bin")
#     st.pyplot(fig)

def plot_fraud_rate_vs_total_by_amount_bin(df, bins=[0, 10, 50, 100, 500, 1000, 5000, 10000, 25000]):
    df['AmountBin'] = pd.cut(df['Amount'], bins=bins)

    bin_counts = df['AmountBin'].value_counts().sort_index()
    fraud_counts = df[df['Class'] == 1]['AmountBin'].value_counts().sort_index()
    
    fraud_rate = (fraud_counts / bin_counts).fillna(0) * 100

    fig, ax1 = plt.subplots(figsize=(14,8))

    ax1.bar(bin_counts.index.astype(str), bin_counts.values, color='teal', alpha=0.7)
    ax1.set_ylabel('Total Transaction Count (log scale)')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(bin_counts.index.astype(str), fraud_rate.values, color='red', marker='o', linewidth=2, label='Fraud Rate (%)')
    ax2.set_ylabel('Fraud Rate (%)')
    
    ax1.set_title('Transaction Count and Fraud Rate per Amount Bin')
    ax1.set_xlabel('Transaction Amount Bins')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)

def plot_amount_distribution(df):
    log_amount = np.log1p(df['Amount'])
    fig, ax = plt.subplots(figsize=(12,6))
    sns.histplot(log_amount, bins=50, kde=True, ax=ax)
    ax.set_title('Transaction Amount Distribution (Log Scale)')
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{np.expm1(t):.0f}' for t in ticks])
    ax.set_xlabel('Transaction Amount')
    ax.set_ylabel('Count')
    fig.tight_layout()
    st.pyplot(fig)

# def plot_amount_distribution_by_class(df):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.kdeplot(data=df, x='Amount', hue='Class', fill=True, common_norm=False, ax=ax, palette='Set2')
    ax.set_title('Transaction Amount Distribution by Class')
    st.pyplot(fig)

# def plot_log_amount_distribution_by_class(df: pd.DataFrame):
    df = df.copy()
    df['LogAmount'] = np.log1p(df['Amount'])

    fig, ax = plt.subplots(figsize=(12,6))
    sns.kdeplot(data=df, x='LogAmount', hue='Class', fill=True, common_norm=False, ax=ax, palette='Set2')
    ax.set_title('Log-Scaled Transaction Amount Distribution by Class')
    ax.set_xlabel('Transaction Amount')

    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{np.expm1(t):.0f}' for t in ticks])

    st.pyplot(fig)

# def plot_time_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['Time'], bins=50, kde=True, ax=ax)
    ax.set_title('Transaction Time Distribution')
    st.pyplot(fig)

def plot_time_kde_by_class(df):
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x='Time', hue='Class', fill=True, common_norm=False, ax=ax, palette='husl')
    ax.set_title('KDE Plot of Transaction Time by Class')
    st.pyplot(fig)

# def plot_correlation_heatmap(df: pd.DataFrame):
    st.subheader("Correlation Heatmap")
    # Filter only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, ax=ax, annot=True, fmt=".2f")
    ax.set_title('Correlation Matrix Heatmap (Numeric Features Only)')
    st.pyplot(fig)

# def plot_filtered_correlation_heatmap(df: pd.DataFrame, threshold: float = 0.00001):
    st.subheader("Correlation Heatmap (Filtered by Correlation with Class)")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Filter columns where correlation with Class is above threshold
    corr_target = corr['Class'].abs()
    relevant_features = corr_target[corr_target > threshold].index
    filtered_corr = corr.loc[relevant_features, relevant_features]

    if len(filtered_corr.columns) <= 1:
        st.write("No features have correlation above the threshold with Class.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(filtered_corr, cmap='coolwarm', linewidths=0.5, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f'Correlation Heatmap (Features with |Correlation| > {threshold})')
    st.pyplot(fig)

def plot_top_features_kde(df, top_features):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,12))
    axs = axs.flatten()

    for idx, feature in enumerate(top_features):
        sns.kdeplot(data=df, x=feature, hue='Class', fill=True, common_norm=False, ax=axs[idx], palette='Set2', alpha=0.5)
        axs[idx].set_title(f'KDE of {feature} by Class')

    for j in range(len(top_features), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    st.pyplot(fig)
