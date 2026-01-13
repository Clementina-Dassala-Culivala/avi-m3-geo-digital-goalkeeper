# =====================================================
# VISUALIZAÇÕES — DIGITAL GOALKEEPER
# =====================================================

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


# =====================================================
# PI 1 — Distribuição Posicional do Guarda-Redes (STREAMLIT SAFE)
# =====================================================
def plot_pi1_positional_distribution(positions, mean_position):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    x = positions["#x0"].dropna().values
    y = positions["#y0"].dropna().values

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0E0E0E")
    ax.set_facecolor("#0E0E0E")

    # --------------------------------------------------
    # 🔹 SUBAMOSTRAGEM (CRÍTICO)
    # --------------------------------------------------
    MAX_POINTS = 3000
    if len(x) > MAX_POINTS:
        idx = np.random.choice(len(x), MAX_POINTS, replace=False)
        x = x[idx]
        y = y[idx]

    # --------------------------------------------------
    # 🔹 KDE ROBUSTO
    # --------------------------------------------------
    try:
        values = np.vstack([x, y])
        kde = gaussian_kde(values, bw_method=0.25)

        xi, yi = np.mgrid[
            x.min():x.max():120j,
            y.min():y.max():120j
        ]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
        zi = zi.reshape(xi.shape)

        ax.imshow(
            zi.T,
            origin="lower",
            cmap="turbo",
            extent=[x.min(), x.max(), y.min(), y.max()],
            alpha=0.95
        )

    except Exception:
        # Fallback seguro
        ax.scatter(x, y, s=4, alpha=0.35, color="#4C78A8")

    # --------------------------------------------------
    # 🔹 POSIÇÃO MÉDIA
    # --------------------------------------------------
    ax.scatter(
        mean_position[0],
        mean_position[1],
        s=120,
        color="#FFD700",
        edgecolor="black",
        zorder=10,
        label="Posição Média"
    )

    ax.set_title(
        "PI 1 — Distribuição Posicional do Guarda-Redes",
        color="white",
        fontsize=14,
        pad=12
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    ax.legend(
        loc="upper right",
        facecolor="#1C1F26",
        edgecolor="none",
        labelcolor="white"
    )

    plt.tight_layout()
    return fig





# =====================================================
# PI 2 — Distância Percorrida
# =====================================================
def plot_pi2_distance_travelled(distances):
    """PI 2 — Distância Percorrida (acumulada)"""

    cumulative_distance = np.cumsum(distances)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cumulative_distance, color="blue")

    ax.set_title("PI 2 – Distância Percorrida pelo Guarda-Redes")
    ax.set_xlabel("Instante (frames)")
    ax.set_ylabel("Distância acumulada")

    return fig


# =====================================================
# PI 3 — Frequência de Ameaças por Zona (ESTÁTICO – LEGADO)
# =====================================================
def plot_pi3_threat_frequency(heatmap: np.ndarray):
    """PI 3 — Frequência de Ameaças por Zona (matplotlib)"""

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap="hot"
    )

    ax.set_title("PI 3 – Frequência de Ameaças por Zona")
    ax.set_xlabel("Eixo X (zonas)")
    ax.set_ylabel("Eixo Y (zonas)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Frequência de ameaças")

    return fig


# =====================================================
# PI 3 — Origem Espacial das Ameaças Ofensivas (INTERATIVO)
# =====================================================
def plot_pi3_threat_frequency_interactive(heatmap: np.ndarray):
    """
    PI 3 — Origem Espacial das Ameaças Ofensivas (Campo)
    Persona: Treinador Principal
    Contexto: Pós-Jogo
    """

    # Evitar log(0)
    heatmap_safe = np.where(heatmap <= 0, 1, heatmap)
    heatmap_log = np.log10(heatmap_safe)

    df = pd.DataFrame(
        heatmap_log,
        columns=[f"X{i}" for i in range(heatmap.shape[1])],
        index=[f"Y{i}" for i in range(heatmap.shape[0])]
    )

    fig = px.imshow(
        df,
        color_continuous_scale="viridis",
        aspect="equal",
        labels=dict(color="log10(Frequência de ameaças)")
    )

    # ---------- Escala de cores (consolidada) ----------
    fig.update_coloraxes(
        cmin=heatmap_log.min(),
        cmax=np.percentile(heatmap_log, 95),
        colorbar=dict(
            title="Frequência de ameaças",
            tickmode="array",
            tickvals=[1.3, 1.6, 2.0],
            ticktext=[
                "Baixa (~20 ameaças)",
                "Média (~40–50 ameaças)",
                "Alta (≥100 ameaças)"
            ]
        )
    )

    # ---------- Baliza ----------
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=0.5,
        y0=3.5, y1=5.5,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    fig.add_annotation(
        x=0,
        y=5.7,
        text="Baliza",
        showarrow=False,
        font=dict(color="white")
    )

    # ---------- Hover ----------
    fig.update_traces(
        hovertemplate=(
            "Zona X: %{x}<br>"
            "Zona Y: %{y}<br>"
            "log10(Ameaças): %{z:.2f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        title="PI 3 — Origem Espacial das Ameaças Ofensivas (Campo)",
        xaxis_title="Eixo X (zonas do campo)",
        yaxis_title="Eixo Y (zonas do campo)",
        template="plotly_dark"
    )

    return fig


# =====================================================
# PI 4 — Intensidade de Reação
# =====================================================
def plot_pi4_reaction_intensity(
    speeds,
    mean_speed,
    max_speed
):
    """PI 4 — Intensidade de Reação do Guarda-Redes"""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(speeds, alpha=0.6, label="Velocidade instantânea")

    ax.axhline(
        mean_speed,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Velocidade média ({mean_speed:.2f})"
    )

    ax.axhline(
        max_speed,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Velocidade máxima ({max_speed:.2f})"
    )

    ax.set_title("PI 4 – Intensidade de Reação do Guarda-Redes")
    ax.set_xlabel("Instante (frames)")
    ax.set_ylabel("Velocidade")
    ax.legend()

    return fig


# =====================================================
# PI 5 — CANAL DE PROGRESSÃO DAS AMEAÇAS (FINAL)
# =====================================================
def plot_pi5_threat_progression_channels(pi5_data: dict):
    """
    PI 5 — Canal de Progressão das Ameaças Ofensivas
    Persona: Treinador Principal
    Contexto: Pós-Jogo
    """

    channels = ["Esquerdo", "Central", "Direito"]
    counts = [pi5_data["counts"][c] for c in channels]
    percentages = [pi5_data["percentages"][c] for c in channels]

    colors = ["#4C78A8", "#54A24B", "#4C78A8"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=channels,
                y=counts,
                text=[f"{p:.1f}%" for p in percentages],
                textposition="auto",
                marker_color=colors
            )
        ]
    )

    fig.update_layout(
        title="PI 5 — Canal de Progressão das Ameaças Ofensivas",
        xaxis_title="Canal do Campo",
        yaxis_title="Número de Ameaças",
        showlegend=False,
        template="plotly_dark",
        margin=dict(t=60, b=40, l=40, r=40)
    )

    return fig


# =====================================================
# 🔁 ALIASES (COMPATIBILIDADE)
# =====================================================
def plot_pi3_threat_frequency_by_zone(heatmap):
    return plot_pi3_threat_frequency(heatmap)


def plot_pi4_reaction_intensity_by_time(
    speeds,
    mean_speed,
    max_speed
):
    return plot_pi4_reaction_intensity(
        speeds,
        mean_speed,
        max_speed
    )
