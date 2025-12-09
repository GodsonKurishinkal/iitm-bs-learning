"""Generate visualizations for Statistics II course notes.

This script creates educational charts and diagrams for inferential statistics,
hypothesis testing, regression, and ANOVA topics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Constants
FIGURE_DPI: int = 150
COLORS: dict[str, str] = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#28A745",
    "reject": "#DC3545",
    "neutral": "#3B3B3B",
}

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_week01_joint_distributions() -> None:
    """Create visualizations for Week 01: Joint Distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Joint PMF as heatmap
    ax1 = axes[0]
    x_vals = [0, 1, 2]
    y_vals = [0, 1, 2]
    joint_pmf = np.array([
        [0.10, 0.15, 0.05],
        [0.08, 0.20, 0.12],
        [0.02, 0.10, 0.18]
    ])

    im = ax1.imshow(joint_pmf, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(len(x_vals)))
    ax1.set_xticklabels(x_vals)
    ax1.set_yticks(range(len(y_vals)))
    ax1.set_yticklabels(y_vals)
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_title("Joint PMF: P(X=x, Y=y)", fontsize=14, fontweight="bold")

    # Add values to cells
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            ax1.text(j, i, f"{joint_pmf[i, j]:.2f}", ha="center", va="center",
                     fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax1, label="Probability")

    # Plot 2: Marginal distributions
    ax2 = axes[1]
    marginal_x = joint_pmf.sum(axis=0)
    marginal_y = joint_pmf.sum(axis=1)

    x_pos = np.arange(len(x_vals))
    width = 0.35

    ax2.bar(x_pos - width/2, marginal_x, width, color=COLORS["primary"],
            label="P(X=x) [Marginal X]")
    ax2.bar(x_pos + width/2, marginal_y, width, color=COLORS["secondary"],
            label="P(Y=y) [Marginal Y]")

    ax2.set_xlabel("Value", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_title("Marginal Distributions", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["0", "1", "2"])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-01-joint-distributions.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-01-joint-distributions.png")


def create_week02_conditional_distributions() -> None:
    """Create visualizations for Week 02: Conditional Distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Conditional distribution visualization
    ax1 = axes[0]
    x = np.linspace(-3, 5, 200)

    # Joint normal visualization
    y1 = stats.norm.pdf(x, 0, 1)  # Marginal Y
    y2 = stats.norm.pdf(x, 1, 0.7)  # Y|X=1
    y3 = stats.norm.pdf(x, 2, 0.7)  # Y|X=2

    ax1.plot(x, y1, color=COLORS["primary"], linewidth=2.5, label="Marginal f(y)")
    ax1.plot(x, y2, color=COLORS["secondary"], linewidth=2.5, linestyle="--",
             label="f(y|X=1)")
    ax1.plot(x, y3, color=COLORS["accent"], linewidth=2.5, linestyle=":",
             label="f(y|X=2)")

    ax1.set_xlabel("y", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Conditional vs Marginal Distribution", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: Independence illustration
    ax2 = axes[1]
    ax2.axis("off")

    text_content = """
    Independence: X and Y are independent if

        P(X=x, Y=y) = P(X=x) × P(Y=y)

    Equivalently:

        P(Y=y | X=x) = P(Y=y)

        (knowing X doesn't change probability of Y)

    Covariance of Independent Variables:

        Cov(X, Y) = 0

    But Cov(X,Y) = 0 does NOT imply independence!
    """

    ax2.text(0.5, 0.5, text_content, transform=ax2.transAxes, fontsize=12,
             verticalalignment="center", horizontalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax2.set_title("Independence & Covariance", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-02-conditional-distributions.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-02-conditional-distributions.png")


def create_week03_covariance() -> None:
    """Create visualizations for Week 03: Covariance and Correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Covariance quadrants
    ax1 = axes[0]

    # Generate correlated data
    mean = [0, 0]
    cov_matrix = [[1, 0.7], [0.7, 1]]
    data = np.random.multivariate_normal(mean, cov_matrix, 200)

    ax1.scatter(data[:, 0], data[:, 1], alpha=0.5, color=COLORS["primary"], s=40)
    ax1.axhline(0, color=COLORS["neutral"], linewidth=1.5, linestyle="--")
    ax1.axvline(0, color=COLORS["neutral"], linewidth=1.5, linestyle="--")

    # Label quadrants
    ax1.text(1.5, 1.5, "+,+\n(positive)", ha="center", fontsize=10,
             color=COLORS["success"])
    ax1.text(-1.5, 1.5, "-,+\n(negative)", ha="center", fontsize=10,
             color=COLORS["reject"])
    ax1.text(-1.5, -1.5, "-,-\n(positive)", ha="center", fontsize=10,
             color=COLORS["success"])
    ax1.text(1.5, -1.5, "+,-\n(negative)", ha="center", fontsize=10,
             color=COLORS["reject"])

    ax1.set_xlabel("X - E[X]", fontsize=12)
    ax1.set_ylabel("Y - E[Y]", fontsize=12)
    ax1.set_title("Covariance: Quadrant Signs", fontsize=14, fontweight="bold")

    # Plot 2: Correlation scale
    ax2 = axes[1]

    # Draw correlation scale
    scale_y = 0.5
    ax2.axhline(scale_y, color=COLORS["neutral"], linewidth=3, xmin=0.1, xmax=0.9)

    # Add tick marks and labels
    for r, pos in [(-1, 0.1), (-0.5, 0.3), (0, 0.5), (0.5, 0.7), (1, 0.9)]:
        ax2.plot(pos, scale_y, "|", color=COLORS["neutral"], markersize=20,
                 markeredgewidth=3)
        ax2.text(pos, scale_y - 0.15, f"r = {r}", ha="center", fontsize=11)

    # Labels for regions
    ax2.text(0.2, scale_y + 0.15, "Strong\nNegative", ha="center", fontsize=10,
             color=COLORS["reject"])
    ax2.text(0.5, scale_y + 0.15, "No\nCorrelation", ha="center", fontsize=10,
             color=COLORS["neutral"])
    ax2.text(0.8, scale_y + 0.15, "Strong\nPositive", ha="center", fontsize=10,
             color=COLORS["success"])

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Correlation Coefficient Scale", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-03-covariance.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-03-covariance.png")


def create_week05_clt() -> None:
    """Create visualizations for Week 05: Central Limit Theorem."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Original distribution (uniform)
    ax_orig = axes[0, 0]
    x_unif = np.linspace(-0.5, 1.5, 100)
    ax_orig.fill_between([0, 1], [1, 1], color=COLORS["primary"], alpha=0.7)
    ax_orig.axhline(0, color="black", linewidth=0.5)
    ax_orig.set_xlim(-0.5, 1.5)
    ax_orig.set_ylim(0, 1.5)
    ax_orig.set_title("Original: Uniform(0,1)", fontsize=12, fontweight="bold")
    ax_orig.set_xlabel("x")
    ax_orig.set_ylabel("f(x)")

    # Sampling distributions for different n
    sample_sizes = [2, 5, 30]
    n_simulations = 10000

    for idx, n in enumerate(sample_sizes):
        ax = axes[0 if idx == 0 else 1, idx if idx == 0 else idx - 1] if idx < 2 else axes[1, 1]
        ax = axes[0, idx + 1] if idx < 2 else axes[1, 0]

        # Simulate sample means
        sample_means = [np.mean(np.random.uniform(0, 1, n)) for _ in range(n_simulations)]

        ax.hist(sample_means, bins=40, density=True, color=COLORS["secondary"],
                alpha=0.7, edgecolor="white")

        # Overlay normal curve
        x_norm = np.linspace(min(sample_means), max(sample_means), 100)
        mu = 0.5
        sigma = (1/12)**0.5 / np.sqrt(n)
        ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma),
                color=COLORS["accent"], linewidth=2.5, label="Normal approx")

        ax.set_title(f"Sample Means (n={n})", fontsize=12, fontweight="bold")
        ax.set_xlabel("x̄")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    # CLT formula
    ax_formula = axes[1, 1]
    ax_formula.axis("off")
    text_content = """
    Central Limit Theorem

    If X₁, X₂, ..., Xₙ are i.i.d. with
    mean μ and variance σ²:

         X̄ₙ → N(μ, σ²/n)

    as n → ∞

    Standard Error: SE = σ/√n
    """
    ax_formula.text(0.5, 0.5, text_content, transform=ax_formula.transAxes,
                    fontsize=12, verticalalignment="center", horizontalalignment="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # SE decreasing with n
    ax_se = axes[1, 2]
    n_vals = np.arange(1, 101)
    se_vals = 1 / np.sqrt(n_vals)
    ax_se.plot(n_vals, se_vals, color=COLORS["primary"], linewidth=2.5)
    ax_se.set_xlabel("Sample Size (n)", fontsize=12)
    ax_se.set_ylabel("Standard Error (σ/√n)", fontsize=12)
    ax_se.set_title("SE Decreases with n", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-05-clt.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-05-clt.png")


def create_week07_confidence_intervals() -> None:
    """Create visualizations for Week 07: Confidence Intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Multiple confidence intervals
    ax1 = axes[0]
    true_mu = 50
    n_samples = 20
    sample_size = 30
    sigma = 10

    np.random.seed(42)

    for i in range(n_samples):
        sample = np.random.normal(true_mu, sigma, sample_size)
        x_bar = np.mean(sample)
        se = sigma / np.sqrt(sample_size)
        ci_lower = x_bar - 1.96 * se
        ci_upper = x_bar + 1.96 * se

        # Color based on whether CI contains true mean
        contains_mean = ci_lower <= true_mu <= ci_upper
        color = COLORS["success"] if contains_mean else COLORS["reject"]

        ax1.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=2)
        ax1.plot(x_bar, i, "o", color=color, markersize=5)

    ax1.axvline(true_mu, color=COLORS["neutral"], linewidth=2, linestyle="--",
                label=f"True μ = {true_mu}")
    ax1.set_xlabel("Value", fontsize=12)
    ax1.set_ylabel("Sample Number", fontsize=12)
    ax1.set_title("20 Confidence Intervals (95%)", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: CI width comparison
    ax2 = axes[1]
    conf_levels = [0.90, 0.95, 0.99]
    z_values = [1.645, 1.96, 2.576]
    se = 10 / np.sqrt(30)

    for i, (conf, z) in enumerate(zip(conf_levels, z_values)):
        width = 2 * z * se
        ax2.barh(i, width, height=0.5, color=COLORS["primary"], alpha=0.7)
        ax2.text(width + 0.2, i, f"Width = {width:.2f}", va="center", fontsize=11)

    ax2.set_yticks(range(len(conf_levels)))
    ax2.set_yticklabels([f"{int(c*100)}% CI" for c in conf_levels])
    ax2.set_xlabel("CI Width", fontsize=12)
    ax2.set_title("Higher Confidence → Wider Interval", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-07-confidence-intervals.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-07-confidence-intervals.png")


def create_week08_hypothesis_testing() -> None:
    """Create visualizations for Week 08: Hypothesis Testing."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Critical regions
    ax1 = axes[0]
    x = np.linspace(-4, 4, 200)
    y = stats.norm.pdf(x, 0, 1)

    ax1.plot(x, y, color=COLORS["primary"], linewidth=2.5)
    ax1.fill_between(x, 0, y, where=(x <= -1.96), alpha=0.5, color=COLORS["reject"],
                      label="Reject H₀ (α/2)")
    ax1.fill_between(x, 0, y, where=(x >= 1.96), alpha=0.5, color=COLORS["reject"])
    ax1.fill_between(x, 0, y, where=(np.abs(x) < 1.96), alpha=0.3,
                      color=COLORS["success"], label="Fail to Reject H₀")

    ax1.axvline(-1.96, color=COLORS["neutral"], linestyle="--", linewidth=1.5)
    ax1.axvline(1.96, color=COLORS["neutral"], linestyle="--", linewidth=1.5)

    ax1.text(-1.96, -0.05, "-1.96", ha="center", fontsize=10)
    ax1.text(1.96, -0.05, "1.96", ha="center", fontsize=10)

    ax1.set_xlabel("z-score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Two-Tailed Test (α = 0.05)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")

    # Plot 2: Type I and Type II errors
    ax2 = axes[1]
    ax2.axis("off")

    # Draw confusion matrix
    cell_text = [
        ["Correct\n(True Negative)", "Type I Error\n(False Positive)"],
        ["Type II Error\n(False Negative)", "Correct\n(True Positive)"]
    ]
    row_labels = ["H₀ True", "H₀ False"]
    col_labels = ["Fail to Reject H₀", "Reject H₀"]

    table = ax2.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                       cellLoc="center", loc="center", cellColours=[
                           [COLORS["success"] + "40", COLORS["reject"] + "40"],
                           [COLORS["accent"] + "40", COLORS["success"] + "40"]
                       ])
    table.scale(1.5, 2.5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    ax2.set_title("Decision Matrix", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-08-hypothesis-testing.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-08-hypothesis-testing.png")


def create_week09_proportions() -> None:
    """Create visualizations for Week 09: Inference for Proportions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Sampling distribution of proportion
    ax1 = axes[0]
    n = 100
    p = 0.4
    se = np.sqrt(p * (1 - p) / n)

    x = np.linspace(0.2, 0.6, 200)
    y = stats.norm.pdf(x, p, se)

    ax1.plot(x, y, color=COLORS["primary"], linewidth=2.5)
    ax1.fill_between(x, 0, y, where=(np.abs(x - p) <= 1.96 * se),
                      alpha=0.3, color=COLORS["primary"])

    ax1.axvline(p, color=COLORS["accent"], linewidth=2, linestyle="--",
                label=f"True p = {p}")
    ax1.axvline(p - 1.96 * se, color=COLORS["secondary"], linestyle=":", linewidth=1.5)
    ax1.axvline(p + 1.96 * se, color=COLORS["secondary"], linestyle=":", linewidth=1.5)

    ax1.set_xlabel("Sample Proportion (p̂)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(f"Sampling Distribution (n={n})", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: Sample size effect
    ax2 = axes[1]
    n_values = [25, 50, 100, 200, 500]
    se_values = [np.sqrt(0.5 * 0.5 / n) for n in n_values]

    ax2.bar(range(len(n_values)), se_values, color=COLORS["primary"],
            edgecolor="white", width=0.6)
    ax2.set_xticks(range(len(n_values)))
    ax2.set_xticklabels([f"n={n}" for n in n_values])
    ax2.set_xlabel("Sample Size", fontsize=12)
    ax2.set_ylabel("Standard Error", fontsize=12)
    ax2.set_title("SE Decreases with Larger n", fontsize=14, fontweight="bold")

    for i, (n, se) in enumerate(zip(n_values, se_values)):
        ax2.text(i, se + 0.005, f"{se:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-09-proportions.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-09-proportions.png")


def create_week10_regression() -> None:
    """Create visualizations for Week 10: Simple Linear Regression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate regression data
    np.random.seed(42)
    x = np.linspace(10, 50, 50)
    y = 5 + 0.8 * x + np.random.normal(0, 5, 50)

    # Fit regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = intercept + slope * x

    # Plot 1: Scatter with regression line
    ax1 = axes[0]
    ax1.scatter(x, y, color=COLORS["primary"], alpha=0.6, s=50, label="Data points")
    ax1.plot(x, y_pred, color=COLORS["accent"], linewidth=2.5,
             label=f"ŷ = {intercept:.2f} + {slope:.2f}x")

    # Show one residual
    idx = 25
    ax1.plot([x[idx], x[idx]], [y[idx], y_pred[idx]], color=COLORS["reject"],
             linewidth=2, linestyle="--", label="Residual")

    ax1.set_xlabel("X (Advertising Spend)", fontsize=12)
    ax1.set_ylabel("Y (Sales)", fontsize=12)
    ax1.set_title(f"Linear Regression (R² = {r_value**2:.3f})", fontsize=14,
                  fontweight="bold")
    ax1.legend()

    # Plot 2: Residual plot
    ax2 = axes[1]
    residuals = y - y_pred

    ax2.scatter(y_pred, residuals, color=COLORS["secondary"], alpha=0.6, s=50)
    ax2.axhline(0, color=COLORS["neutral"], linewidth=2, linestyle="--")

    ax2.set_xlabel("Fitted Values (ŷ)", fontsize=12)
    ax2.set_ylabel("Residuals (y - ŷ)", fontsize=12)
    ax2.set_title("Residual Plot", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-10-regression.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-10-regression.png")


def create_week11_anova() -> None:
    """Create visualizations for Week 11: ANOVA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate group data
    np.random.seed(42)
    group_a = np.random.normal(50, 8, 30)
    group_b = np.random.normal(55, 8, 30)
    group_c = np.random.normal(60, 8, 30)

    all_data = [group_a, group_b, group_c]
    group_labels = ["Group A", "Group B", "Group C"]

    # Plot 1: Box plots
    ax1 = axes[0]
    bp = ax1.boxplot(all_data, labels=group_labels, patch_artist=True)
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Grand mean line
    grand_mean = np.mean([np.mean(g) for g in all_data])
    ax1.axhline(grand_mean, color=COLORS["neutral"], linewidth=2, linestyle="--",
                label=f"Grand Mean = {grand_mean:.1f}")

    ax1.set_ylabel("Value", fontsize=12)
    ax1.set_title("One-Way ANOVA: Group Comparison", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: F-distribution
    ax2 = axes[1]
    df1, df2 = 2, 87  # k-1, n-k
    x = np.linspace(0, 5, 200)
    y = stats.f.pdf(x, df1, df2)

    ax2.plot(x, y, color=COLORS["primary"], linewidth=2.5)

    # Critical value
    f_crit = stats.f.ppf(0.95, df1, df2)
    ax2.fill_between(x, 0, y, where=(x >= f_crit), alpha=0.5,
                      color=COLORS["reject"], label=f"Reject H₀ (α=0.05)")
    ax2.axvline(f_crit, color=COLORS["neutral"], linewidth=1.5, linestyle="--",
                label=f"F_crit = {f_crit:.2f}")

    ax2.set_xlabel("F-statistic", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(f"F-Distribution (df1={df1}, df2={df2})", fontsize=14,
                  fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-11-anova.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-11-anova.png")


def create_week12_chi_square() -> None:
    """Create visualizations for Week 12: Chi-Square Tests."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Chi-square distribution
    ax1 = axes[0]
    x = np.linspace(0, 25, 200)

    for df, color in [(3, COLORS["primary"]), (5, COLORS["secondary"]),
                      (10, COLORS["accent"])]:
        y = stats.chi2.pdf(x, df)
        ax1.plot(x, y, color=color, linewidth=2.5, label=f"df = {df}")

    ax1.set_xlabel("χ²", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Chi-Square Distribution", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: Contingency table visualization
    ax2 = axes[1]
    ax2.axis("off")

    # Example contingency table
    cell_text = [
        ["150", "100", "250"],
        ["80", "170", "250"],
        ["230", "270", "500"]
    ]
    row_labels = ["Male", "Female", "Total"]
    col_labels = ["Product A", "Product B", "Total"]

    table = ax2.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                       cellLoc="center", loc="center")
    table.scale(1.3, 2.0)
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    ax2.set_title("Contingency Table: χ² Test of Independence", fontsize=14,
                  fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-12-chi-square.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-12-chi-square.png")


def main() -> None:
    """Generate all Statistics II visualizations."""
    print("Generating Statistics II visualizations...")
    print("-" * 40)

    create_week01_joint_distributions()
    create_week02_conditional_distributions()
    create_week03_covariance()
    create_week05_clt()
    create_week07_confidence_intervals()
    create_week08_hypothesis_testing()
    create_week09_proportions()
    create_week10_regression()
    create_week11_anova()
    create_week12_chi_square()

    print("-" * 40)
    print(f"All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
