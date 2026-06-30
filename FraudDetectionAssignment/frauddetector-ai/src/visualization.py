# src/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style='whitegrid')

def plot_class_distribution(df, save_path=None):
    class_counts = df['Class'].value_counts()
    print("\nClass Distribution:\n", class_counts)
    print("\nPercentage Distribution:\n", class_counts / len(df) * 100)
    ax = sns.countplot(x='Class', data=df)
    ax.set_yscale('log')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    plt.title('Class Distribution (Log Scale with Annotations)')
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = f'{100 * count / total:.5f}%'
        ax.annotate(f'{count}\n({percentage})', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    plt.show()


def plot_amount_distribution(dataframe):
    log_amount = np.log1p(dataframe['Amount'])
    plt.figure(figsize=(12,6))
    ax = sns.histplot(log_amount, bins=50, kde=True)
    plt.title('Transaction Amount Distribution (Log Scaled with Actual Amount Labels)')
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=8)
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{np.expm1(t):.0f}' for t in ticks])
    plt.xlabel('Transaction Amount')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_amount_distribution_by_class(dataframe):
    dataframe = dataframe.copy()
    dataframe['LogAmount'] = np.log1p(dataframe['Amount'])
    dataframe['ClassLabel'] = dataframe['Class'].map({0: 'Non-Fraud', 1: 'Fraud'})

    plt.figure(figsize=(12,6))
    ax = sns.kdeplot(
        data=dataframe, 
        x='LogAmount', 
        hue='ClassLabel', 
        fill=True, 
        common_norm=False, 
        palette='Set1', 
        alpha=0.5
    )
    plt.title('Log-Scaled Transaction Amount Distribution by Class (Fraud vs Non-Fraud)')

    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{np.expm1(t):.0f}' for t in ticks])

    plt.xlabel('Transaction Amount')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()


def plot_top_features_kde(df, features, ncols=3):
    rows = -(-len(features) // ncols)
    fig, axes = plt.subplots(rows, ncols, figsize=(6 * ncols, 4 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.kdeplot(data=df, x=feature, hue='Class', fill=True,
                    common_norm=False, palette='Set2', alpha=0.5, ax=axes[i])
        axes[i].set_title(f'KDE of {feature} by Class')

    # Turn off unused subplots
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_time_distribution(df, save_path=None):
    plt.figure(figsize=(8,6))
    sns.histplot(df['Time'], bins=50, kde=True)
    plt.title('Transaction Time Distribution')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    if save_path:
        plt.savefig(save_path)
    plt.show()
