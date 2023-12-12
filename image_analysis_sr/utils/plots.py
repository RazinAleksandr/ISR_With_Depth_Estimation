import seaborn as sns
import matplotlib.pyplot as plt


def plot_metrics_distribution_grid(df, figsize=(25, 10)):
    """
    Plot the distribution of each metric in the DataFrame in a grid.

    Args:
    df (pandas.DataFrame): DataFrame with metrics, indexed by image name.
    """
    sns.set(style="whitegrid")

    num_metrics = len(df.columns)
    num_rows = (num_metrics + 2) // 4

    fig, axes = plt.subplots(num_rows, 4, figsize=figsize)
    axes = axes.flatten()

    for idx, column in enumerate(df.columns):
        sns.distplot(df[column], ax=axes[idx], label=column, hist=False)
        axes[idx].legend(loc="upper right")
        axes[idx].set_title(f'Distribution of {column}')
        axes[idx].set_xlabel('Metric Value')
        axes[idx].set_ylabel('Density')

    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    return fig