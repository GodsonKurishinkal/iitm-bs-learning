"""Generate visualizations for Mathematics II course notes.

This script creates educational charts and diagrams for linear algebra
and multivariable calculus topics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def create_week01_vectors_matrices() -> None:
    """Create visualizations for Week 01: Vectors and Matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Vector operations
    ax1 = axes[0]
    origin = np.array([0, 0])
    v1 = np.array([3, 1])
    v2 = np.array([1, 2])
    v_sum = v1 + v2

    ax1.quiver(*origin, *v1, angles="xy", scale_units="xy", scale=1,
               color=COLORS["primary"], width=0.02, label="v = (3, 1)")
    ax1.quiver(*origin, *v2, angles="xy", scale_units="xy", scale=1,
               color=COLORS["secondary"], width=0.02, label="w = (1, 2)")
    ax1.quiver(*origin, *v_sum, angles="xy", scale_units="xy", scale=1,
               color=COLORS["accent"], width=0.02, label="v + w = (4, 3)")

    # Parallelogram
    ax1.plot([v1[0], v_sum[0]], [v1[1], v_sum[1]], "--", color=COLORS["secondary"],
             alpha=0.5)
    ax1.plot([v2[0], v_sum[0]], [v2[1], v_sum[1]], "--", color=COLORS["primary"],
             alpha=0.5)

    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 4)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Vector Addition", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Matrix transformation
    ax2 = axes[1]

    # Original unit square
    square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    ax2.plot(square[0], square[1], color=COLORS["primary"], linewidth=2,
             label="Original")
    ax2.fill(square[0], square[1], color=COLORS["primary"], alpha=0.2)

    # Transformation matrix (shear)
    A = np.array([[1, 0.5], [0, 1]])
    transformed = A @ square
    ax2.plot(transformed[0], transformed[1], color=COLORS["accent"], linewidth=2,
             label="After shear")
    ax2.fill(transformed[0], transformed[1], color=COLORS["accent"], alpha=0.2)

    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Matrix Transformation (Shear)", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-01-vectors-matrices.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-01-vectors-matrices.png")


def create_week02_linear_equations() -> None:
    """Create visualizations for Week 02: Solving Linear Equations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: System of 2 equations (intersecting lines)
    ax1 = axes[0]
    x = np.linspace(-1, 5, 200)

    # x + y = 4 => y = 4 - x
    # 2x - y = 2 => y = 2x - 2
    y1 = 4 - x
    y2 = 2 * x - 2

    ax1.plot(x, y1, color=COLORS["primary"], linewidth=2, label="x + y = 4")
    ax1.plot(x, y2, color=COLORS["secondary"], linewidth=2, label="2x - y = 2")
    ax1.plot(2, 2, "o", color=COLORS["accent"], markersize=12,
             label="Solution (2, 2)")

    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Unique Solution (Intersecting Lines)", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-2, 5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Three cases
    ax2 = axes[1]
    x = np.linspace(-1, 4, 200)

    # Case 1: Unique solution (intersecting)
    ax2.plot(x, 2 - x, color=COLORS["primary"], linewidth=2, alpha=0.7)
    ax2.plot(x, x, color=COLORS["primary"], linewidth=2, alpha=0.7)
    ax2.plot(1, 1, "o", color=COLORS["primary"], markersize=10)

    # Case 2: No solution (parallel) - shifted up
    ax2.plot(x, 3 - x + 3, color=COLORS["secondary"], linewidth=2, alpha=0.7)
    ax2.plot(x, 5 - x + 3, color=COLORS["secondary"], linewidth=2, alpha=0.7)

    # Case 3: Infinite solutions (same line) - shifted down
    ax2.plot(x, -1 - 0.5 * x, color=COLORS["accent"], linewidth=4, alpha=0.7)

    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)

    # Legend manually
    ax2.plot([], [], color=COLORS["primary"], linewidth=2, label="Unique solution")
    ax2.plot([], [], color=COLORS["secondary"], linewidth=2, label="No solution")
    ax2.plot([], [], color=COLORS["accent"], linewidth=4, label="Infinite solutions")

    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Three Cases for Linear Systems", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-3, 9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-02-linear-equations.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-02-linear-equations.png")


def create_week03_vector_spaces() -> None:
    """Create visualizations for Week 03: Introduction to Vector Spaces."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Span of vectors
    ax1 = axes[0]

    # Vectors v1 and v2
    v1 = np.array([2, 1])
    v2 = np.array([1, 2])

    # Draw span (all linear combinations) as shaded region
    # For visualization, show some linear combinations
    for a in np.linspace(-1, 1.5, 6):
        for b in np.linspace(-1, 1.5, 6):
            point = a * v1 + b * v2
            ax1.plot(point[0], point[1], ".", color=COLORS["secondary"], alpha=0.3,
                     markersize=8)

    # Draw basis vectors
    ax1.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["primary"], width=0.03, label="v1 = (2, 1)")
    ax1.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["accent"], width=0.03, label="v2 = (1, 2)")

    ax1.set_xlim(-2, 4)
    ax1.set_ylim(-2, 4)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Span of Two Vectors = R²", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)

    # Plot 2: Subspace (line through origin)
    ax2 = axes[1]

    t = np.linspace(-2, 2, 100)
    v = np.array([1, 2])

    # The subspace (line through origin)
    line_x = t * v[0]
    line_y = t * v[1]
    ax2.plot(line_x, line_y, color=COLORS["primary"], linewidth=3,
             label="Subspace: Span{(1,2)}")
    ax2.fill_between(line_x, line_y - 0.1, line_y + 0.1, alpha=0.2,
                      color=COLORS["primary"])

    # Some vectors in the subspace
    for s in [-1, 0.5, 1, 1.5]:
        point = s * v
        ax2.quiver(0, 0, point[0], point[1], angles="xy", scale_units="xy", scale=1,
                   color=COLORS["secondary"], width=0.02, alpha=0.7)

    ax2.plot(0, 0, "o", color=COLORS["accent"], markersize=10,
             label="Origin (must be in subspace)")

    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("1D Subspace of R²", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-03-vector-spaces.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-03-vector-spaces.png")


def create_week05_rank_nullity() -> None:
    """Create visualizations for Week 05: Rank-Nullity Theorem."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Visualization of rank-nullity theorem
    # dim(V) = rank(T) + nullity(T)

    # Draw domain V
    circle_v = plt.Circle((2, 3), 1.5, color=COLORS["primary"], alpha=0.3)
    ax.add_patch(circle_v)
    ax.text(2, 3, "V\ndim = n", ha="center", va="center", fontsize=14,
            fontweight="bold")
    ax.text(2, 4.8, "Domain", ha="center", fontsize=12)

    # Draw codomain W
    circle_w = plt.Circle((8, 3), 1.5, color=COLORS["secondary"], alpha=0.3)
    ax.add_patch(circle_w)
    ax.text(8, 3, "W", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(8, 4.8, "Codomain", ha="center", fontsize=12)

    # Draw image (inside W)
    circle_img = plt.Circle((8, 3), 0.8, color=COLORS["accent"], alpha=0.5)
    ax.add_patch(circle_img)
    ax.text(8, 3, "Im(T)\nrank", ha="center", va="center", fontsize=11)

    # Draw kernel (inside V)
    circle_ker = plt.Circle((2, 3), 0.6, color=COLORS["success"], alpha=0.5)
    ax.add_patch(circle_ker)
    ax.text(2, 3, "Ker(T)\nnullity", ha="center", va="center", fontsize=10)

    # Arrow for transformation
    ax.annotate("", xy=(6.3, 3), xytext=(3.7, 3),
                arrowprops=dict(arrowstyle="->", color=COLORS["neutral"], lw=2))
    ax.text(5, 3.3, "T: V → W", ha="center", fontsize=12, fontweight="bold")

    # Rank-Nullity formula
    ax.text(5, 0.8, "Rank-Nullity Theorem:", ha="center", fontsize=14,
            fontweight="bold")
    ax.text(5, 0.3, "dim(V) = rank(T) + nullity(T)", ha="center", fontsize=14,
            style="italic")
    ax.text(5, -0.2, "n = dim(Im(T)) + dim(Ker(T))", ha="center", fontsize=12)

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rank-Nullity Theorem Visualization", fontsize=16, fontweight="bold",
                 pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-05-rank-nullity.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-05-rank-nullity.png")


def create_week07_eigenvalues() -> None:
    """Create visualizations for Week 07: Similar Matrices and Inner Products."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Eigenvector visualization
    ax1 = axes[0]

    # Matrix A = [[2, 1], [1, 2]] has eigenvalues 3 and 1
    # Eigenvector for lambda=3: (1, 1)
    # Eigenvector for lambda=1: (1, -1)

    # Draw original vectors and their transformations
    A = np.array([[2, 1], [1, 2]])

    # Random vector (gets rotated and scaled)
    v_random = np.array([1, 0.3])
    Av_random = A @ v_random
    ax1.quiver(0, 0, v_random[0], v_random[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["neutral"], width=0.02, alpha=0.5)
    ax1.quiver(0, 0, Av_random[0], Av_random[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["neutral"], width=0.02, alpha=0.5, linestyle="--")

    # Eigenvector (1, 1) - only gets scaled
    v_eigen = np.array([1, 1]) / np.sqrt(2)
    Av_eigen = A @ v_eigen
    ax1.quiver(0, 0, v_eigen[0], v_eigen[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["primary"], width=0.03, label="Eigenvector v")
    ax1.quiver(0, 0, Av_eigen[0], Av_eigen[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["accent"], width=0.03, label="Av = 3v (scaled only)")

    ax1.set_xlim(-0.5, 3)
    ax1.set_ylim(-0.5, 3)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Eigenvectors: Direction Preserved", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)

    # Plot 2: Inner product / orthogonality
    ax2 = axes[1]

    # Orthogonal vectors
    v1 = np.array([2, 0])
    v2 = np.array([0, 1.5])

    ax2.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["primary"], width=0.03, label="u = (2, 0)")
    ax2.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["secondary"], width=0.03, label="v = (0, 1.5)")

    # Right angle indicator
    rect = plt.Rectangle((0, 0), 0.2, 0.2, fill=False,
                          edgecolor=COLORS["neutral"], linewidth=1.5)
    ax2.add_patch(rect)

    # Non-orthogonal vectors
    v3 = np.array([1, 1])
    ax2.quiver(0, 0, v3[0], v3[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["accent"], width=0.03, label="w = (1, 1)")

    ax2.text(2.2, 1.2, "u · v = 0\n(orthogonal)", fontsize=11, color=COLORS["primary"])
    ax2.text(1.2, 0.5, "u · w = 2\n(not orthogonal)", fontsize=11, color=COLORS["accent"])

    ax2.set_xlim(-0.5, 3)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Inner Product & Orthogonality", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-07-eigenvalues-inner-products.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-07-eigenvalues-inner-products.png")


def create_week08_gram_schmidt() -> None:
    """Create visualizations for Week 08: Orthogonality and Gram-Schmidt."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Gram-Schmidt process
    ax1 = axes[0]

    # Original vectors
    v1 = np.array([2, 1])
    v2 = np.array([1, 2])

    # Gram-Schmidt
    u1 = v1 / np.linalg.norm(v1)  # Normalize v1
    proj = np.dot(v2, u1) * u1  # Projection of v2 onto u1
    u2_unnorm = v2 - proj
    u2 = u2_unnorm / np.linalg.norm(u2_unnorm)

    # Draw original vectors
    ax1.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["neutral"], width=0.02, alpha=0.5, label="Original v1, v2")
    ax1.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["neutral"], width=0.02, alpha=0.5)

    # Draw orthonormal vectors (scaled up for visibility)
    ax1.quiver(0, 0, u1[0] * 1.5, u1[1] * 1.5, angles="xy", scale_units="xy", scale=1,
               color=COLORS["primary"], width=0.03, label="u1 (normalized v1)")
    ax1.quiver(0, 0, u2[0] * 1.5, u2[1] * 1.5, angles="xy", scale_units="xy", scale=1,
               color=COLORS["accent"], width=0.03, label="u2 (orthogonal)")

    # Right angle indicator
    rect = plt.Rectangle((0, 0), 0.15, 0.15, fill=False,
                          edgecolor=COLORS["success"], linewidth=2,
                          transform=ax1.transData)
    angle = np.arctan2(u1[1], u1[0])
    t = plt.matplotlib.transforms.Affine2D().rotate(angle) + ax1.transData
    rect.set_transform(t)
    ax1.add_patch(rect)

    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Gram-Schmidt Orthonormalization", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axvline(x=0, color="black", linewidth=0.5)

    # Plot 2: Projection onto subspace
    ax2 = axes[1]

    # Vector to project
    b = np.array([1, 2])
    # Subspace direction
    a = np.array([2, 0.5])
    a_unit = a / np.linalg.norm(a)

    # Projection
    proj_len = np.dot(b, a_unit)
    proj = proj_len * a_unit

    # Orthogonal component
    orth = b - proj

    ax2.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["primary"], width=0.03, label="b (vector to project)")
    ax2.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1,
               color=COLORS["accent"], width=0.03, label="proj_a(b)")

    # Draw the subspace (line)
    t = np.linspace(-0.5, 3, 100)
    ax2.plot(t * a_unit[0], t * a_unit[1], "--", color=COLORS["secondary"],
             linewidth=2, alpha=0.5, label="Subspace (span{a})")

    # Orthogonal drop line
    ax2.plot([b[0], proj[0]], [b[1], proj[1]], ":", color=COLORS["neutral"],
             linewidth=2)

    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Orthogonal Projection", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-08-gram-schmidt.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-08-gram-schmidt.png")


def create_week09_multivariable() -> None:
    """Create visualizations for Week 09: Multivariable Functions."""
    fig = plt.figure(figsize=(12, 5))

    # Plot 1: 3D surface
    ax1 = fig.add_subplot(121, projection="3d")

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("y", fontsize=11)
    ax1.set_zlabel("z", fontsize=11)
    ax1.set_title("f(x,y) = x² + y² (Paraboloid)", fontsize=13, fontweight="bold")

    # Plot 2: Contour plot
    ax2 = fig.add_subplot(122)

    contour = ax2.contour(X, Y, Z, levels=10, cmap="viridis")
    ax2.clabel(contour, inline=True, fontsize=9)
    ax2.plot(0, 0, "o", color=COLORS["accent"], markersize=10, label="Minimum (0,0)")

    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Contour Plot (Level Curves)", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-09-multivariable.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-09-multivariable.png")


def create_week10_gradient() -> None:
    """Create visualizations for Week 10: Gradient and Critical Points."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Gradient vectors on contour plot
    ax1 = axes[0]

    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Contour
    contour = ax1.contour(X, Y, Z, levels=8, cmap="Blues", alpha=0.7)
    ax1.clabel(contour, inline=True, fontsize=8)

    # Gradient vectors at selected points
    points = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0.5, 0.5)]
    for px, py in points:
        grad_x = 2 * px  # df/dx = 2x
        grad_y = 2 * py  # df/dy = 2y
        ax1.quiver(px, py, grad_x * 0.3, grad_y * 0.3,
                   color=COLORS["accent"], width=0.01, scale=1, scale_units="xy")

    ax1.plot(0, 0, "o", color=COLORS["success"], markersize=10,
             label="Minimum (gradient = 0)")

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Gradient Points Uphill", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Saddle point visualization
    ax2 = axes[1]

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2  # Saddle function

    contour = ax2.contour(X, Y, Z, levels=15, cmap="RdBu", alpha=0.7)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(0, 0, "o", color=COLORS["accent"], markersize=12, label="Saddle point")

    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("y", fontsize=12)
    ax2.set_title("Saddle Point: f(x,y) = x² - y²", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-10-gradient.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-10-gradient.png")


def create_week11_optimization() -> None:
    """Create visualizations for Week 11: Hessian and Optimization."""
    fig = plt.figure(figsize=(12, 5))

    # Plot 1: 3D surface with min/max/saddle
    ax1 = fig.add_subplot(121, projection="3d")

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Bowl (minimum)

    ax1.plot_surface(X, Y, Z, cmap="Greens", alpha=0.7, edgecolor="none")
    ax1.scatter([0], [0], [0], color=COLORS["accent"], s=100, label="Minimum")

    ax1.set_xlabel("x", fontsize=10)
    ax1.set_ylabel("y", fontsize=10)
    ax1.set_zlabel("z", fontsize=10)
    ax1.set_title("Minimum: H positive definite", fontsize=12, fontweight="bold")

    # Plot 2: Hessian test summary
    ax2 = fig.add_subplot(122)
    ax2.axis("off")

    # Create a table-like visualization
    text_content = """
    Second Derivative Test (Hessian):

    At critical point where ∇f = 0:

    ┌─────────────────────────────────────┐
    │  det(H) > 0 and f_xx > 0            │
    │  → LOCAL MINIMUM                    │
    │  (H is positive definite)           │
    ├─────────────────────────────────────┤
    │  det(H) > 0 and f_xx < 0            │
    │  → LOCAL MAXIMUM                    │
    │  (H is negative definite)           │
    ├─────────────────────────────────────┤
    │  det(H) < 0                         │
    │  → SADDLE POINT                     │
    │  (H is indefinite)                  │
    ├─────────────────────────────────────┤
    │  det(H) = 0                         │
    │  → TEST INCONCLUSIVE                │
    └─────────────────────────────────────┘

    where H = | f_xx  f_xy |
              | f_yx  f_yy |
    """

    ax2.text(0.5, 0.5, text_content, transform=ax2.transAxes, fontsize=11,
             verticalalignment="center", horizontalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax2.set_title("Hessian Test for Critical Points", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "week-11-optimization.png", dpi=FIGURE_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Created: week-11-optimization.png")


def main() -> None:
    """Generate all Mathematics II visualizations."""
    print("Generating Mathematics II visualizations...")
    print("-" * 40)

    create_week01_vectors_matrices()
    create_week02_linear_equations()
    create_week03_vector_spaces()
    create_week05_rank_nullity()
    create_week07_eigenvalues()
    create_week08_gram_schmidt()
    create_week09_multivariable()
    create_week10_gradient()
    create_week11_optimization()

    print("-" * 40)
    print(f"All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
