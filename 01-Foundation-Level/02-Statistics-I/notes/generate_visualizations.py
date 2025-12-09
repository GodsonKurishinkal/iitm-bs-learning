"""Generate visualizations for Statistics I course notes.

This script creates educational charts and diagrams for descriptive statistics,
probability, and distributions topics.
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
    "success": "#C73E1D",
    "neutral": "#3B3B3B",
}

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_week01_data_types() -> None:
    """Create visualizations for Week 01: Data Types."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # Create a hierarchical diagram
    text_content = """
                            ┌─────────────────┐
                            │    VARIABLES    │
                            └────────┬────────┘
                    ┌───────────────┴───────────────┐
            ┌───────┴───────┐               ┌───────┴───────┐
            │  CATEGORICAL  │               │   NUMERICAL   │
            │  (Qualitative)│               │ (Quantitative)│
            └───────┬───────┘               └───────┬───────┘
        ┌───────────┴───────────┐       ┌───────────┴───────────┐
    ┌───┴───┐           ┌───────┴───┐ ┌───┴───┐           ┌─────┴─────┐
    │NOMINAL│           │  ORDINAL  │ │DISCRETE│           │CONTINUOUS │
    └───────┘           └───────────┘ └────────┘           └───────────┘

    No order            Has order      Countable            Measurable
    Categories only     Rankings       Whole numbers        Any value

    Examples:           Examples:      Examples:            Examples:
    • Color             • Ratings      • # of orders        • Weight
    • Product type      • Grades       • # of customers     • Temperature
    • Region            • Satisfaction • Defect count       • Revenue
    """

    ax.text(0.5, 0.5, text_content, transform=ax.transAxes, fontsize=11,
            verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace")
    ax.set_title("Classification of Variables", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-01-data-types.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-01-data-types.png")


def create_week02_categorical_data() -> None:
    """Create visualizations for Week 02: Categorical Data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Bar chart
    ax1 = axes[0]
    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]
    values = [450, 320, 280, 190, 160]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["success"], COLORS["neutral"]]

    bars = ax1.bar(categories, values, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_xlabel("Product Category", fontsize=12)
    ax1.set_ylabel("Sales (units)", fontsize=12)
    ax1.set_title("Bar Chart: Sales by Category", fontsize=14, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(val), ha="center", fontsize=10)
    ax1.set_ylim(0, 520)

    # Plot 2: Pie chart
    ax2 = axes[1]
    ax2.pie(values, labels=categories, colors=colors, autopct="%1.1f%%",
            startangle=90, explode=[0.05, 0, 0, 0, 0])
    ax2.set_title("Pie Chart: Sales Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-02-categorical-data.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-02-categorical-data.png")


def create_week03_numerical_data() -> None:
    """Create visualizations for Week 03: Numerical Data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate sample data
    data = np.concatenate([
        np.random.normal(50, 10, 80),
        np.random.normal(70, 5, 20)
    ])

    # Plot 1: Histogram with central tendency
    ax1 = axes[0]
    ax1.hist(data, bins=15, color=COLORS["primary"], edgecolor="white", alpha=0.7)

    mean_val = np.mean(data)
    median_val = np.median(data)

    ax1.axvline(mean_val, color=COLORS["accent"], linewidth=2, linestyle="-",
                label=f"Mean = {mean_val:.1f}")
    ax1.axvline(median_val, color=COLORS["secondary"], linewidth=2, linestyle="--",
                label=f"Median = {median_val:.1f}")

    ax1.set_xlabel("Value", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Histogram with Central Tendency", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: Box plot with IQR explanation
    ax2 = axes[1]
    bp = ax2.boxplot(data, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor(COLORS["primary"])
    bp["boxes"][0].set_alpha(0.7)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Add annotations
    ax2.annotate("Q3 (75th)", xy=(1.15, q3), fontsize=10)
    ax2.annotate("Q1 (25th)", xy=(1.15, q1), fontsize=10)
    ax2.annotate("Median", xy=(1.15, median_val), fontsize=10)
    ax2.annotate(f"IQR = {iqr:.1f}", xy=(0.6, (q1 + q3) / 2), fontsize=11,
                 fontweight="bold", color=COLORS["accent"])

    # Draw IQR bracket
    ax2.plot([0.75, 0.75], [q1, q3], color=COLORS["accent"], linewidth=3)

    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_title("Box Plot (5-Number Summary)", fontsize=14, fontweight="bold")
    ax2.set_xlim(0.5, 1.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-03-numerical-data.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-03-numerical-data.png")


def create_week04_correlation() -> None:
    """Create visualizations for Week 04: Association and Correlation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Positive correlation
    ax1 = axes[0]
    x1 = np.random.normal(50, 15, 100)
    y1 = 0.8 * x1 + np.random.normal(0, 8, 100)
    ax1.scatter(x1, y1, color=COLORS["primary"], alpha=0.6, s=50)
    z = np.polyfit(x1, y1, 1)
    p = np.poly1d(z)
    ax1.plot(sorted(x1), p(sorted(x1)), color=COLORS["accent"], linewidth=2)
    r1 = np.corrcoef(x1, y1)[0, 1]
    ax1.set_title(f"Positive Correlation\nr = {r1:.2f}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X", fontsize=11)
    ax1.set_ylabel("Y", fontsize=11)

    # Plot 2: Negative correlation
    ax2 = axes[1]
    x2 = np.random.normal(50, 15, 100)
    y2 = -0.7 * x2 + 100 + np.random.normal(0, 10, 100)
    ax2.scatter(x2, y2, color=COLORS["secondary"], alpha=0.6, s=50)
    z = np.polyfit(x2, y2, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(x2), p(sorted(x2)), color=COLORS["accent"], linewidth=2)
    r2 = np.corrcoef(x2, y2)[0, 1]
    ax2.set_title(f"Negative Correlation\nr = {r2:.2f}", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X", fontsize=11)
    ax2.set_ylabel("Y", fontsize=11)

    # Plot 3: No correlation
    ax3 = axes[2]
    x3 = np.random.normal(50, 15, 100)
    y3 = np.random.normal(50, 15, 100)
    ax3.scatter(x3, y3, color=COLORS["neutral"], alpha=0.6, s=50)
    r3 = np.corrcoef(x3, y3)[0, 1]
    ax3.set_title(f"No Correlation\nr = {r3:.2f}", fontsize=13, fontweight="bold")
    ax3.set_xlabel("X", fontsize=11)
    ax3.set_ylabel("Y", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-04-correlation.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-04-correlation.png")


def create_week07_probability() -> None:
    """Create visualizations for Week 07: Introduction to Probability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Venn diagram for probability
    ax1 = axes[0]

    # Draw circles for events A and B
    circle_a = plt.Circle((0.35, 0.5), 0.25, color=COLORS["primary"], alpha=0.5,
                           label="P(A)")
    circle_b = plt.Circle((0.55, 0.5), 0.25, color=COLORS["secondary"], alpha=0.5,
                           label="P(B)")
    ax1.add_patch(circle_a)
    ax1.add_patch(circle_b)

    ax1.text(0.25, 0.5, "A only", ha="center", va="center", fontsize=11)
    ax1.text(0.65, 0.5, "B only", ha="center", va="center", fontsize=11)
    ax1.text(0.45, 0.5, "A∩B", ha="center", va="center", fontsize=11, fontweight="bold")

    # Sample space rectangle
    rect = plt.Rectangle((0, 0.15), 0.9, 0.7, fill=False, edgecolor=COLORS["neutral"],
                          linewidth=2)
    ax1.add_patch(rect)
    ax1.text(0.85, 0.8, "S", fontsize=12, fontweight="bold")

    ax1.set_xlim(-0.1, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Venn Diagram: Events A and B", fontsize=14, fontweight="bold")

    # Plot 2: Probability rules summary
    ax2 = axes[1]
    ax2.axis("off")

    text_content = """
    Probability Axioms:

    1. 0 ≤ P(A) ≤ 1

    2. P(S) = 1  (sample space)

    3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

    Key Formulas:

    • Complement: P(A') = 1 - P(A)

    • Addition Rule:
      P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

    • If mutually exclusive:
      P(A ∪ B) = P(A) + P(B)
    """

    ax2.text(0.5, 0.5, text_content, transform=ax2.transAxes, fontsize=12,
             verticalalignment="center", horizontalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax2.set_title("Probability Rules", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-07-probability.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-07-probability.png")


def create_week08_conditional_probability() -> None:
    """Create visualizations for Week 08: Conditional Probability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tree diagram for Bayes' theorem
    ax.axis("off")

    # Draw tree
    # Level 0 (root)
    ax.plot([0.5], [0.9], "o", color=COLORS["primary"], markersize=15)

    # Level 1 (A and A')
    ax.plot([0.5, 0.25], [0.9, 0.6], "-", color=COLORS["neutral"], linewidth=2)
    ax.plot([0.5, 0.75], [0.9, 0.6], "-", color=COLORS["neutral"], linewidth=2)
    ax.plot([0.25], [0.6], "o", color=COLORS["primary"], markersize=12)
    ax.plot([0.75], [0.6], "o", color=COLORS["secondary"], markersize=12)
    ax.text(0.25, 0.55, "A", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.75, 0.55, "A'", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.35, 0.78, "P(A)", ha="center", fontsize=10)
    ax.text(0.65, 0.78, "P(A')", ha="center", fontsize=10)

    # Level 2 (B|A and B|A')
    ax.plot([0.25, 0.1], [0.6, 0.3], "-", color=COLORS["neutral"], linewidth=2)
    ax.plot([0.25, 0.35], [0.6, 0.3], "-", color=COLORS["neutral"], linewidth=2)
    ax.plot([0.75, 0.65], [0.6, 0.3], "-", color=COLORS["neutral"], linewidth=2)
    ax.plot([0.75, 0.9], [0.6, 0.3], "-", color=COLORS["neutral"], linewidth=2)

    ax.plot([0.1], [0.3], "o", color=COLORS["accent"], markersize=10)
    ax.plot([0.35], [0.3], "o", color=COLORS["neutral"], markersize=10)
    ax.plot([0.65], [0.3], "o", color=COLORS["accent"], markersize=10)
    ax.plot([0.9], [0.3], "o", color=COLORS["neutral"], markersize=10)

    ax.text(0.1, 0.22, "B", ha="center", fontsize=11)
    ax.text(0.35, 0.22, "B'", ha="center", fontsize=11)
    ax.text(0.65, 0.22, "B", ha="center", fontsize=11)
    ax.text(0.9, 0.22, "B'", ha="center", fontsize=11)

    ax.text(0.15, 0.47, "P(B|A)", ha="center", fontsize=9)
    ax.text(0.32, 0.47, "P(B'|A)", ha="center", fontsize=9)
    ax.text(0.68, 0.47, "P(B|A')", ha="center", fontsize=9)
    ax.text(0.85, 0.47, "P(B'|A')", ha="center", fontsize=9)

    # Bayes formula
    ax.text(0.5, 0.08, "Bayes' Theorem:", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.02, "P(A|B) = P(B|A) × P(A) / P(B)", ha="center", fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Probability Tree & Bayes' Theorem", fontsize=16, fontweight="bold",
                 pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-08-conditional-probability.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-08-conditional-probability.png")


def create_week09_random_variables() -> None:
    """Create visualizations for Week 09: Random Variables."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: PMF (discrete)
    ax1 = axes[0]
    x_discrete = np.arange(0, 6)
    pmf = stats.binom.pmf(x_discrete, n=5, p=0.5)

    ax1.bar(x_discrete, pmf, color=COLORS["primary"], edgecolor="white", width=0.6)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("P(X = x)", fontsize=12)
    ax1.set_title("PMF: Binomial(n=5, p=0.5)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_discrete)

    # Plot 2: PDF and CDF (continuous)
    ax2 = axes[1]
    x_cont = np.linspace(-4, 4, 200)
    pdf = stats.norm.pdf(x_cont, 0, 1)
    cdf = stats.norm.cdf(x_cont, 0, 1)

    ax2.plot(x_cont, pdf, color=COLORS["primary"], linewidth=2.5, label="PDF f(x)")
    ax2.plot(x_cont, cdf, color=COLORS["accent"], linewidth=2.5, linestyle="--",
             label="CDF F(x)")
    ax2.fill_between(x_cont, 0, pdf, where=(x_cont < 1), alpha=0.3,
                      color=COLORS["primary"], label="P(X < 1)")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_title("PDF & CDF: Normal(0, 1)", fontsize=14, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-09-random-variables.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-09-random-variables.png")


def create_week10_expectation() -> None:
    """Create visualizations for Week 10: Expectation and Variance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Mean and variance visualization
    ax1 = axes[0]

    # Two distributions with same mean, different variance
    x = np.linspace(-6, 10, 200)
    y1 = stats.norm.pdf(x, 2, 1)  # Low variance
    y2 = stats.norm.pdf(x, 2, 2)  # High variance

    ax1.plot(x, y1, color=COLORS["primary"], linewidth=2.5,
             label="σ² = 1 (concentrated)")
    ax1.plot(x, y2, color=COLORS["secondary"], linewidth=2.5,
             label="σ² = 4 (spread out)")
    ax1.axvline(2, color=COLORS["accent"], linewidth=2, linestyle="--",
                label="Mean μ = 2")

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title("Same Mean, Different Variance", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: Standard deviation regions
    ax2 = axes[1]
    x = np.linspace(-4, 4, 200)
    y = stats.norm.pdf(x, 0, 1)

    ax2.plot(x, y, color=COLORS["primary"], linewidth=2.5)
    ax2.fill_between(x, 0, y, where=(np.abs(x) <= 1), alpha=0.4,
                      color=COLORS["primary"], label="μ ± 1σ (68%)")
    ax2.fill_between(x, 0, y, where=(np.abs(x) <= 2) & (np.abs(x) > 1),
                      alpha=0.3, color=COLORS["secondary"], label="μ ± 2σ (95%)")
    ax2.fill_between(x, 0, y, where=(np.abs(x) <= 3) & (np.abs(x) > 2),
                      alpha=0.2, color=COLORS["accent"], label="μ ± 3σ (99.7%)")

    ax2.axvline(0, color=COLORS["neutral"], linewidth=1, linestyle="--")
    ax2.set_xlabel("x (standard deviations from mean)", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Empirical Rule (68-95-99.7)", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-10-expectation.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-10-expectation.png")


def create_week11_discrete_distributions() -> None:
    """Create visualizations for Week 11: Binomial and Poisson."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Binomial distributions
    ax1 = axes[0]
    x = np.arange(0, 21)

    for p, color, label in [(0.2, COLORS["primary"], "p=0.2"),
                            (0.5, COLORS["secondary"], "p=0.5"),
                            (0.8, COLORS["accent"], "p=0.8")]:
        pmf = stats.binom.pmf(x, n=20, p=p)
        ax1.plot(x, pmf, "o-", color=color, linewidth=2, markersize=5, label=label)

    ax1.set_xlabel("k (number of successes)", fontsize=12)
    ax1.set_ylabel("P(X = k)", fontsize=12)
    ax1.set_title("Binomial(n=20, p)", fontsize=14, fontweight="bold")
    ax1.legend()

    # Plot 2: Poisson distributions
    ax2 = axes[1]
    x = np.arange(0, 20)

    for lam, color, label in [(2, COLORS["primary"], "λ=2"),
                              (5, COLORS["secondary"], "λ=5"),
                              (10, COLORS["accent"], "λ=10")]:
        pmf = stats.poisson.pmf(x, lam)
        ax2.plot(x, pmf, "o-", color=color, linewidth=2, markersize=5, label=label)

    ax2.set_xlabel("k (number of events)", fontsize=12)
    ax2.set_ylabel("P(X = k)", fontsize=12)
    ax2.set_title("Poisson(λ)", fontsize=14, fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-11-discrete-distributions.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-11-discrete-distributions.png")


def create_week12_continuous_distributions() -> None:
    """Create visualizations for Week 12: Continuous Distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Normal distribution
    ax1 = axes[0]
    x = np.linspace(-4, 4, 200)

    for mu, sigma, color in [(0, 1, COLORS["primary"]),
                             (0, 0.5, COLORS["secondary"]),
                             (1, 1, COLORS["accent"])]:
        y = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, y, color=color, linewidth=2.5,
                 label=f"μ={mu}, σ={sigma}")

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title("Normal Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    # Plot 2: Uniform distribution
    ax2 = axes[1]
    x = np.linspace(-0.5, 5.5, 200)

    for a, b, color in [(0, 2, COLORS["primary"]),
                        (1, 4, COLORS["secondary"])]:
        y = stats.uniform.pdf(x, a, b - a)
        ax2.plot(x, y, color=color, linewidth=2.5, label=f"U({a}, {b})")
        ax2.fill_between(x, 0, y, alpha=0.2, color=color)

    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Uniform Distribution", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.set_ylim(0, 0.7)

    # Plot 3: Exponential distribution
    ax3 = axes[2]
    x = np.linspace(0, 5, 200)

    for lam, color in [(0.5, COLORS["primary"]),
                       (1, COLORS["secondary"]),
                       (2, COLORS["accent"])]:
        y = stats.expon.pdf(x, scale=1/lam)
        ax3.plot(x, y, color=color, linewidth=2.5, label=f"λ={lam}")

    ax3.set_xlabel("x", fontsize=12)
    ax3.set_ylabel("f(x)", fontsize=12)
    ax3.set_title("Exponential Distribution", fontsize=13, fontweight="bold")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-12-continuous-distributions.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-12-continuous-distributions.png")


def main() -> None:
    """Generate all Statistics I visualizations."""
    print("Generating Statistics I visualizations...")
    print("-" * 40)

    create_week01_data_types()
    create_week02_categorical_data()
    create_week03_numerical_data()
    create_week04_correlation()
    create_week07_probability()
    create_week08_conditional_probability()
    create_week09_random_variables()
    create_week10_expectation()
    create_week11_discrete_distributions()
    create_week12_continuous_distributions()

    print("-" * 40)
    print(f"All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
