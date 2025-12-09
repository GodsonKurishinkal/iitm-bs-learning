"""Generate visualizations for Mathematics I course notes.

This script creates educational charts and diagrams for each week's topics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Constants
FIGURE_DPI: int = 150
FIGURE_SIZE: tuple[int, int] = (10, 6)
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


def create_week02_coordinate_geometry() -> None:
    """Create visualizations for Week 02: Coordinate Systems."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Distance and Midpoint
    ax1 = axes[0]
    p1, p2 = (1, 2), (5, 6)
    midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], "o-", color=COLORS["primary"],
             linewidth=2, markersize=10, label="Line segment")
    ax1.plot(*midpoint, "s", color=COLORS["accent"], markersize=12, label="Midpoint")
    ax1.annotate(f"P₁({p1[0]}, {p1[1]})", p1, textcoords="offset points",
                 xytext=(10, -15), fontsize=11)
    ax1.annotate(f"P₂({p2[0]}, {p2[1]})", p2, textcoords="offset points",
                 xytext=(10, 5), fontsize=11)
    ax1.annotate(f"M({midpoint[0]:.1f}, {midpoint[1]:.1f})", midpoint,
                 textcoords="offset points", xytext=(10, 10), fontsize=11)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Distance and Midpoint Formula", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 8)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Different Line Forms
    ax2 = axes[1]
    x = np.linspace(-2, 6, 100)

    # y = mx + c (slope-intercept)
    ax2.plot(x, 2 * x + 1, label="y = 2x + 1 (slope-intercept)",
             color=COLORS["primary"], linewidth=2)
    # Point-slope form visualization
    ax2.plot(x, 0.5 * (x - 2) + 3, label="y - 3 = 0.5(x - 2) (point-slope)",
             color=COLORS["secondary"], linewidth=2)
    ax2.plot(2, 3, "o", color=COLORS["secondary"], markersize=10)

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Line Equation Forms", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.set_xlim(-2, 6)
    ax2.set_ylim(-2, 8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-02-coordinate-geometry.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 02 visualization saved")


def create_week03_quadratic_functions() -> None:
    """Create visualizations for Week 03: Quadratic Functions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Parabola with vertex
    ax1 = axes[0]
    x = np.linspace(-4, 6, 200)
    a, b, c = 1, -2, -3
    y = a * x**2 + b * x + c
    vertex_x = -b / (2 * a)
    vertex_y = a * vertex_x**2 + b * vertex_x + c

    ax1.plot(x, y, color=COLORS["primary"], linewidth=2.5, label=f"f(x) = x² - 2x - 3")
    ax1.plot(vertex_x, vertex_y, "o", color=COLORS["accent"], markersize=12,
             label=f"Vertex ({vertex_x:.0f}, {vertex_y:.0f})")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.axvline(x=vertex_x, color=COLORS["accent"], linestyle="--", alpha=0.5,
                label="Axis of symmetry")

    # Mark roots
    roots = np.roots([a, b, c])
    for root in roots:
        if np.isreal(root):
            ax1.plot(root.real, 0, "x", color=COLORS["success"], markersize=12,
                     markeredgewidth=3)

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title("Quadratic Function with Vertex & Roots", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-6, 10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Discriminant cases
    ax2 = axes[1]
    x = np.linspace(-3, 3, 200)

    # Two real roots (Δ > 0)
    ax2.plot(x, x**2 - 1, color=COLORS["primary"], linewidth=2,
             label="Δ > 0: Two roots")
    # One real root (Δ = 0)
    ax2.plot(x, x**2, color=COLORS["secondary"], linewidth=2,
             label="Δ = 0: One root")
    # No real roots (Δ < 0)
    ax2.plot(x, x**2 + 1, color=COLORS["accent"], linewidth=2,
             label="Δ < 0: No real roots")

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Discriminant Cases", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_ylim(-2, 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-03-quadratic-functions.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 03 visualization saved")


def create_week04_polynomials() -> None:
    """Create visualizations for Week 04: Polynomials."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Multiplicities
    ax1 = axes[0]
    x = np.linspace(-2, 4, 500)

    # (x-1)^1 * (x-2)^2 * (x-3)^3 scaled
    y = 0.2 * (x - 0) * (x - 2)**2 * (x - 3)

    ax1.plot(x, y, color=COLORS["primary"], linewidth=2.5)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)

    # Mark roots with different behaviors
    ax1.plot(0, 0, "o", color=COLORS["secondary"], markersize=10,
             label="Mult. 1: Crosses")
    ax1.plot(2, 0, "s", color=COLORS["accent"], markersize=10,
             label="Mult. 2: Bounces")
    ax1.plot(3, 0, "^", color=COLORS["success"], markersize=10,
             label="Mult. 1: Crosses")

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title("Root Multiplicities & Graph Behavior", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.set_ylim(-3, 5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: End Behavior
    ax2 = axes[1]
    x = np.linspace(-2, 2, 200)

    ax2.plot(x, x**3, color=COLORS["primary"], linewidth=2,
             label="Odd degree, a > 0")
    ax2.plot(x, -x**3, color=COLORS["secondary"], linewidth=2,
             label="Odd degree, a < 0")
    ax2.plot(x, x**4, color=COLORS["accent"], linewidth=2, linestyle="--",
             label="Even degree, a > 0")
    ax2.plot(x, -x**4, color=COLORS["success"], linewidth=2, linestyle="--",
             label="Even degree, a < 0")

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Polynomial End Behavior", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylim(-5, 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-04-polynomials.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 04 visualization saved")


def create_week05_transformations() -> None:
    """Create visualizations for Week 05: Functions and Transformations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Exponential functions
    ax1 = axes[0]
    x = np.linspace(-2, 3, 200)

    ax1.plot(x, np.exp(x), color=COLORS["primary"], linewidth=2.5, label="eˣ (growth)")
    ax1.plot(x, 2**x, color=COLORS["secondary"], linewidth=2, label="2ˣ (growth)")
    ax1.plot(x, (0.5)**x, color=COLORS["accent"], linewidth=2, label="(1/2)ˣ (decay)")
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="y = 1 (all pass through)")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title("Exponential Functions", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_xlim(-2, 3)
    ax1.set_ylim(-0.5, 8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Function transformations
    ax2 = axes[1]
    x = np.linspace(-4, 4, 200)

    ax2.plot(x, x**2, color=COLORS["neutral"], linewidth=2, label="f(x) = x²")
    ax2.plot(x, (x - 2)**2, color=COLORS["primary"], linewidth=2,
             label="f(x-2): Shift right 2")
    ax2.plot(x, x**2 + 3, color=COLORS["secondary"], linewidth=2,
             label="f(x)+3: Shift up 3")
    ax2.plot(x, 2 * x**2, color=COLORS["accent"], linewidth=2, linestyle="--",
             label="2f(x): Vertical stretch")

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Function Transformations", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-1, 12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-05-transformations.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 05 visualization saved")


def create_week06_logarithms() -> None:
    """Create visualizations for Week 06: Logarithmic Functions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Log functions comparison
    ax1 = axes[0]
    x = np.linspace(0.01, 10, 500)

    ax1.plot(x, np.log(x), color=COLORS["primary"], linewidth=2.5, label="ln(x)")
    ax1.plot(x, np.log10(x), color=COLORS["secondary"], linewidth=2, label="log₁₀(x)")
    ax1.plot(x, np.log2(x), color=COLORS["accent"], linewidth=2, label="log₂(x)")

    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    ax1.plot(1, 0, "o", color=COLORS["success"], markersize=10,
             label="(1, 0): All logs pass through")

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title("Logarithmic Functions", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-3, 4)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Exponential vs Logarithm (inverse relationship)
    ax2 = axes[1]
    x = np.linspace(-2, 2.5, 200)
    x_log = np.linspace(0.01, 8, 200)

    ax2.plot(x, np.exp(x), color=COLORS["primary"], linewidth=2.5, label="y = eˣ")
    ax2.plot(x_log, np.log(x_log), color=COLORS["secondary"], linewidth=2.5,
             label="y = ln(x)")
    ax2.plot(x, x, color="gray", linestyle="--", linewidth=1.5, label="y = x (mirror line)")

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Exponential & Log are Inverses", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_xlim(-2, 8)
    ax2.set_ylim(-2, 8)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-06-logarithms.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 06 visualization saved")


def create_week07_limits() -> None:
    """Create visualizations for Week 07: Sequences, Limits, and Continuity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Sequence convergence
    ax1 = axes[0]
    n = np.arange(1, 20)

    seq1 = 1 / n
    seq2 = (n + 1) / n
    seq3 = (-1)**n / n

    ax1.scatter(n, seq1, color=COLORS["primary"], s=50, label="aₙ = 1/n → 0")
    ax1.scatter(n, seq2, color=COLORS["secondary"], s=50, label="aₙ = (n+1)/n → 1")
    ax1.scatter(n, seq3, color=COLORS["accent"], s=50, label="aₙ = (-1)ⁿ/n → 0")

    ax1.axhline(y=0, color=COLORS["primary"], linestyle="--", alpha=0.5)
    ax1.axhline(y=1, color=COLORS["secondary"], linestyle="--", alpha=0.5)

    ax1.set_xlabel("n", fontsize=12)
    ax1.set_ylabel("aₙ", fontsize=12)
    ax1.set_title("Sequence Convergence", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Limit of a function
    ax2 = axes[1]
    x = np.linspace(-2, 2, 500)
    x_hole = np.linspace(-2, 2, 500)
    x_hole = x_hole[np.abs(x_hole) > 0.05]

    # sin(x)/x which has limit 1 at x=0
    y = np.sin(x) / np.where(np.abs(x) > 0.001, x, np.nan)

    ax2.plot(x, y, color=COLORS["primary"], linewidth=2.5)
    ax2.plot(0, 1, "o", color="white", markersize=10, markeredgecolor=COLORS["primary"],
             markeredgewidth=2)
    ax2.plot(0, 1, "o", color=COLORS["accent"], markersize=6,
             label="lim(x→0) sin(x)/x = 1")

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Limit: sin(x)/x as x → 0", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_ylim(-0.5, 1.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-07-limits.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 07 visualization saved")


def create_week08_derivatives() -> None:
    """Create visualizations for Week 08: Derivatives and Critical Points."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Function and its derivative
    ax1 = axes[0]
    x = np.linspace(-2, 4, 200)
    f = x**3 - 3 * x**2 - 9 * x + 5
    f_prime = 3 * x**2 - 6 * x - 9

    ax1.plot(x, f, color=COLORS["primary"], linewidth=2.5, label="f(x) = x³ - 3x² - 9x + 5")
    ax1.plot(x, f_prime, color=COLORS["secondary"], linewidth=2, linestyle="--",
             label="f'(x) = 3x² - 6x - 9")

    # Critical points where f'(x) = 0
    crit_points = [-1, 3]
    for cp in crit_points:
        y_val = cp**3 - 3 * cp**2 - 9 * cp + 5
        ax1.plot(cp, y_val, "o", color=COLORS["accent"], markersize=10)
    ax1.plot([], [], "o", color=COLORS["accent"], markersize=10, label="Critical points")

    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Function and Its Derivative", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(-30, 20)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tangent line visualization
    ax2 = axes[1]
    x = np.linspace(-1, 5, 200)
    f = x**2

    # Tangent at x=2
    x0 = 2
    f_x0 = x0**2
    slope = 2 * x0
    tangent = slope * (x - x0) + f_x0

    ax2.plot(x, f, color=COLORS["primary"], linewidth=2.5, label="f(x) = x²")
    ax2.plot(x, tangent, color=COLORS["secondary"], linewidth=2, linestyle="--",
             label=f"Tangent at x={x0}")
    ax2.plot(x0, f_x0, "o", color=COLORS["accent"], markersize=12)
    ax2.annotate(f"Point ({x0}, {f_x0})\nSlope = {slope}", (x0, f_x0),
                 textcoords="offset points", xytext=(15, 15), fontsize=10)

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Tangent Line = Derivative at a Point", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_ylim(-2, 20)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-08-derivatives.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 08 visualization saved")


def create_week09_integration() -> None:
    """Create visualizations for Week 09: Integration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Riemann sum approximation
    ax1 = axes[0]

    def f(x: "NDArray[np.float64]") -> "NDArray[np.float64]":
        return x**2

    a, b, n = 0, 2, 8
    x_fine = np.linspace(a, b, 200)
    ax1.plot(x_fine, f(x_fine), color=COLORS["primary"], linewidth=2.5, label="f(x) = x²")

    # Draw rectangles (left Riemann sum)
    dx = (b - a) / n
    for i in range(n):
        x_left = a + i * dx
        height = f(np.array([x_left]))[0]
        rect = plt.Rectangle((x_left, 0), dx, height,
                              facecolor=COLORS["secondary"], edgecolor="white",
                              alpha=0.6)
        ax1.add_patch(rect)

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title(f"Riemann Sum (n={n} rectangles)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_xlim(-0.2, 2.2)
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Definite integral as area
    ax2 = axes[1]
    x = np.linspace(0, 3, 200)
    y = np.sin(x) + 1

    ax2.plot(x, y, color=COLORS["primary"], linewidth=2.5, label="f(x) = sin(x) + 1")
    ax2.fill_between(x, 0, y, alpha=0.3, color=COLORS["secondary"],
                      label="∫₀³ f(x)dx (Area)")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("f(x)", fontsize=12)
    ax2.set_title("Definite Integral = Area Under Curve", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_ylim(-0.2, 2.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-09-integration.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 09 visualization saved")


def create_week10_graphs() -> None:
    """Create visualizations for Week 10: Graph Theory Basics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Graph representation example
    ax1 = axes[0]

    # Define graph vertices and edges
    vertices = {0: (0, 1), 1: (1, 2), 2: (2, 1), 3: (1, 0)}
    edges = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Draw edges
    for u, v in edges:
        ax1.plot([vertices[u][0], vertices[v][0]],
                 [vertices[u][1], vertices[v][1]],
                 color=COLORS["neutral"], linewidth=2, zorder=1)

    # Draw vertices
    for v, (x, y) in vertices.items():
        ax1.scatter(x, y, s=500, color=COLORS["primary"], zorder=2, edgecolors="white",
                    linewidth=2)
        ax1.annotate(str(v), (x, y), ha="center", va="center", fontsize=14,
                     fontweight="bold", color="white")

    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect("equal")
    ax1.set_title("Graph G = (V, E)", fontsize=14, fontweight="bold")
    ax1.axis("off")

    # Add adjacency matrix
    adj_matrix = "Adjacency Matrix:\n    0 1 2 3\n0 [ 0 1 0 1 ]\n1 [ 1 0 1 1 ]\n2 [ 0 1 0 1 ]\n3 [ 1 1 1 0 ]"
    ax1.text(2.8, 1, adj_matrix, fontsize=10, fontfamily="monospace",
             verticalalignment="center")

    # Plot 2: BFS vs DFS traversal
    ax2 = axes[1]
    ax2.text(0.5, 0.9, "BFS (Breadth-First Search)", fontsize=14, fontweight="bold",
             ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.75, "Level by level exploration\nUses: Queue (FIFO)\nFinds: Shortest path",
             fontsize=11, ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.55, "Order: 0 → 1, 3 → 2", fontsize=11, ha="center",
             transform=ax2.transAxes, color=COLORS["primary"])

    ax2.axhline(y=0.45, color="gray", linewidth=1, linestyle="--")

    ax2.text(0.5, 0.35, "DFS (Depth-First Search)", fontsize=14, fontweight="bold",
             ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.2, "Go deep before backtracking\nUses: Stack (LIFO)\nFinds: Cycles, paths",
             fontsize=11, ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.05, "Order: 0 → 1 → 2 → 3", fontsize=11, ha="center",
             transform=ax2.transAxes, color=COLORS["secondary"])

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("BFS vs DFS Comparison", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-10-graph-basics.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 10 visualization saved")


def create_week11_graph_algorithms() -> None:
    """Create visualizations for Week 11: Graph Algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Dijkstra's algorithm concept
    ax1 = axes[0]

    # Weighted graph
    vertices = {
        "A": (0, 1), "B": (1, 2), "C": (2, 2),
        "D": (1, 0), "E": (2, 0)
    }
    edges = [
        ("A", "B", 4), ("A", "D", 2), ("B", "C", 3),
        ("B", "D", 1), ("D", "E", 5), ("C", "E", 1)
    ]

    # Draw edges with weights
    for u, v, w in edges:
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        ax1.plot([x1, x2], [y1, y2], color=COLORS["neutral"], linewidth=2, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax1.annotate(str(w), (mx, my), fontsize=10, ha="center",
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    # Highlight shortest path A → E (A→D→B→C→E)
    shortest = [("A", "D"), ("D", "B"), ("B", "C"), ("C", "E")]
    for u, v in shortest:
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        ax1.plot([x1, x2], [y1, y2], color=COLORS["accent"], linewidth=4, zorder=1,
                 alpha=0.7)

    # Draw vertices
    for v, (x, y) in vertices.items():
        ax1.scatter(x, y, s=600, color=COLORS["primary"], zorder=2, edgecolors="white",
                    linewidth=2)
        ax1.annotate(v, (x, y), ha="center", va="center", fontsize=14,
                     fontweight="bold", color="white")

    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_aspect("equal")
    ax1.set_title("Dijkstra's Shortest Path (A→E)", fontsize=14, fontweight="bold")
    ax1.axis("off")
    ax1.text(1, -0.3, "Shortest: A→D→B→C→E = 7", fontsize=11, ha="center",
             color=COLORS["accent"])

    # Plot 2: MST concept
    ax2 = axes[1]

    # Same vertices, different visualization
    for u, v, w in edges:
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        ax2.plot([x1, x2], [y1, y2], color=COLORS["neutral"], linewidth=1.5,
                 linestyle="--", alpha=0.4, zorder=1)

    # MST edges
    mst_edges = [("A", "D", 2), ("B", "D", 1), ("B", "C", 3), ("C", "E", 1)]
    for u, v, w in mst_edges:
        x1, y1 = vertices[u]
        x2, y2 = vertices[v]
        ax2.plot([x1, x2], [y1, y2], color=COLORS["success"], linewidth=3, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax2.annotate(str(w), (mx, my), fontsize=10, ha="center",
                     bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    # Draw vertices
    for v, (x, y) in vertices.items():
        ax2.scatter(x, y, s=600, color=COLORS["primary"], zorder=2, edgecolors="white",
                    linewidth=2)
        ax2.annotate(v, (x, y), ha="center", va="center", fontsize=14,
                     fontweight="bold", color="white")

    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_aspect("equal")
    ax2.set_title("Minimum Spanning Tree (MST)", fontsize=14, fontweight="bold")
    ax2.axis("off")
    ax2.text(1, -0.3, "MST Total Weight = 7", fontsize=11, ha="center",
             color=COLORS["success"])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-11-graph-algorithms.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Week 11 visualization saved")


def main() -> None:
    """Generate all Mathematics I visualizations."""
    print("Generating Mathematics I visualizations...")
    print("-" * 40)

    create_week02_coordinate_geometry()
    create_week03_quadratic_functions()
    create_week04_polynomials()
    create_week05_transformations()
    create_week06_logarithms()
    create_week07_limits()
    create_week08_derivatives()
    create_week09_integration()
    create_week10_graphs()
    create_week11_graph_algorithms()

    print("-" * 40)
    print(f"All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
