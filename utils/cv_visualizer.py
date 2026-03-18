

# =============================================================================
# 7. Cross-Validation History / Training History
# =============================================================================

def plot_cv_history(
    cv_scores: dict | list,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize cross-validation fold scores as a bar chart.
    
    Useful for visualizing training history / performance across CV folds.
    For deep learning models, can be adapted to show epoch-wise loss/accuracy.

    Parameters
    ----------
    cv_scores : dict | list
        Either:
        - dict: {'fold_1': 0.95, 'fold_2': 0.93, ...}
        - list: [fold_scores, ...]
    model_name : str
        Name of the model (for title)
    save_path : str | None
        Optional output file path

    Returns
    -------
    plt.Figure
    """
    if isinstance(cv_scores, dict):
        folds = list(cv_scores.keys())
        scores = list(cv_scores.values())
    else:
        folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
        scores = list(cv_scores) if isinstance(cv_scores, (list, np.ndarray)) else [cv_scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#3498DB' if i % 2 == 0 else '#2ECC71' for i in range(len(folds))]
    bars = ax.bar(folds, scores, color=colors, edgecolor='white', linewidth=1.5, alpha=0.8)
    
    # Add value labels on top of bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    
    # Add mean line
    mean_score = np.mean(scores)
    ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_score:.4f}', alpha=0.7)
    
    ax.set_title(f"{model_name} – Cross-Validation Scores", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("F1 Score (Macro)", fontsize=12)
    ax.set_ylim(0, max(scores) * 1.15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=15 if len(folds) > 5 else 0)
    plt.tight_layout()
    
    if save_path:
        _savefig(fig, save_path)
    return fig


def plot_training_history(
    history: dict,
    metrics: list = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history for deep learning models.
    
    Shows loss and optional metrics (e.g., accuracy) over epochs.

    Parameters
    ----------
    history : dict
        Keys: 'loss', 'val_loss', 'accuracy', 'val_accuracy', etc.
        Values: lists of metric values per epoch
    metrics : list | None
        Which metrics to plot besides 'loss'. Default: ['accuracy']
    save_path : str | None
        Optional output file path

    Returns
    -------
    plt.Figure
    """
    if metrics is None:
        metrics = ['accuracy']
    
    n_metrics = 1 + len(metrics)  # loss + additional metrics
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(history.get('loss', [])) + 1)
    
    # Plot loss
    if 'loss' in history:
        axes[0].plot(epochs, history['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
        axes[0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    # Plot other metrics
    for i, metric in enumerate(metrics, start=1):
        if i < len(axes) and metric in history:
            axes[i].plot(epochs, history[metric], 'g-o', label=f'Train {metric}', linewidth=2, markersize=4)
            if f'val_{metric}' in history:
                axes[i].plot(epochs, history[f'val_{metric}'], 'orange', marker='s', 
                           label=f'Val {metric}', linewidth=2, markersize=4)
            axes[i].set_title(metric.capitalize(), fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(alpha=0.3)
    
    fig.suptitle('Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        _savefig(fig, save_path)
    return fig
