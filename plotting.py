
# =====================================================================
# FILE: plotting.py
# Matplotlib figure builders and the download helper.
# =====================================================================
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse
import streamlit as st


def download_fig(fig, filename="plot.png", text="Download Plot"):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)
    img_buffer.seek(0)
    st.download_button(label=text, data=img_buffer.getvalue(),
                       file_name=filename, mime="image/png")


def dollar_formatter(x, pos):
    return f"${x:,.0f}"

def plot_ce_plane(dalys, costs, region, perspective, sim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dalys, costs, s=0.5, color="g", alpha=.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    hi_d, lo_d = np.max(dalys), np.min(dalys)
    for cet, color in [(sim.threshold, "blue"), (sim.gdp_ppp, "red"), (sim.gdp_ppp_3, "k")]:
        ax.plot([lo_d, hi_d], [lo_d * cet, hi_d * cet],
                label=f"CET: ${cet:,.0f}", ls="--", color=color)
    ax.grid(axis="both", alpha=.2, linewidth=.3)
    
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
    fig.suptitle(f"Cost-effectiveness Plane: {region}")
    ax.set_title(perspective)
    ci_ellipse(dalys, costs)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_ylabel(r"$\Delta$ Cost", fontweight="bold")
    ax.set_xlabel(r"$\Delta$ DALYs", fontweight="bold")
    return fig

def ci_ellipse(x, y, ax=None, edgecolor='red', facecolor='none', linestyle='-', linewidth=2):
    """Plot confidence interval ellipse for cost-effectiveness plane."""
    if ax is None:
        ax = plt.gca()
    
    # Handle cases with insufficient data
    if len(x) < 3 or len(y) < 3:
        return
    
    mean = np.array([np.mean(x), np.mean(y)])
    cov = np.cov(x, y)
    
    # Check for valid covariance
    if np.isnan(cov).any() or np.isinf(cov).any():
        return
    
    # Use eigh for symmetric matrices (more stable)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return
    
    # Ensure eigenvalues are positive (add small regularization)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # Use first eigenvector for angle
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    width = 2 * np.sqrt(eigenvalues[0]) * np.sqrt(5.991)
    height = 2 * np.sqrt(eigenvalues[1]) * np.sqrt(5.991)
    
    ellipse = Ellipse(xy=mean, width=width, height=height, 
                     angle=np.degrees(angle),
                     edgecolor=edgecolor, facecolor=facecolor,
                     linestyle=linestyle, linewidth=linewidth)
    ax.add_patch(ellipse)
def plot_tornado_diagram(tornado_results, base_nmb, region_name):
    tornado_results.sort(key=lambda x: x['nmb_range'], reverse=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    parameters = [r['parameter'] for r in tornado_results]
    y_pos = np.arange(len(parameters))

    left_bars, right_bars = [], []
    for r in tornado_results:
        low, high = r['low_nmb'], r['high_nmb']
        if low < high:
            left_bars.append(low - base_nmb)
            right_bars.append(high - base_nmb)
        else:
            left_bars.append(high - base_nmb)
            right_bars.append(low - base_nmb)

    ax.barh(y_pos, left_bars, align='center', color='lightcoral', alpha=0.7, label='Unfavorable')
    ax.barh(y_pos, right_bars, align='center', color='lightgreen', alpha=0.7, label='Favorable')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parameters)
    ax.invert_yaxis()
    ax.set_xlabel('Change in Net Monetary Benefit ($)')
    ax.set_title(f'Tornado Diagram - Sensitivity Analysis\n{region_name}',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(dollar_formatter))
    ax.grid(axis='x', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(loc='lower right')
    for i, r in enumerate(tornado_results):
        ax.text(left_bars[i] - abs(left_bars[i]) * 0.1, i, f"{r['low_value']:.3f}",
                ha='right', va='center', fontsize=8)
        ax.text(right_bars[i] + abs(right_bars[i]) * 0.1, i, f"{r['high_value']:.3f}",
                ha='left', va='center', fontsize=8)
    plt.tight_layout()
    return fig

