import pandas as pd


def pi1_positional_distribution(X: pd.DataFrame):
    """
    PI 1 — Distribuição Posicional e Posição Média do Guarda-Redes
    Assume o guarda-redes como o jogador 0 (#x0, #y0).
    """

    gr_positions = X[["#x0", "#y0"]].copy()

    mean_x = gr_positions["#x0"].mean()
    mean_y = gr_positions["#y0"].mean()

    return {
        "positions": gr_positions,
        "mean_position": (mean_x, mean_y)
    }


import numpy as np


def pi2_distance_travelled(X: pd.DataFrame):

    """
    PI 2 — Distância Percorrida pelo Guarda-Redes
    Calculada como soma das distâncias euclidianas entre posições consecutivas.
    Assume sequência temporal implícita na ordem das linhas.
    """

    x = X["#x0"].values
    y = X["#y0"].values

    dx = np.diff(x)
    dy = np.diff(y)

    distances = np.sqrt(dx**2 + dy**2)

    total_distance = distances.sum()

    return {
        "total_distance": total_distance,
        "instant_distances": distances
    }


import numpy as np


def pi3_threat_frequency_by_zone(
    X: pd.DataFrame,
    bins_x: int = 10,
    bins_y: int = 10
):
    """
    PI 3 — Frequência de Ameaças por Zona
    Calcula a densidade de posições da bola (#ball_x, #ball_y)
    num grid espacial.
    """

    ball_x = X["#ball_x"].values
    ball_y = X["#ball_y"].values

    heatmap, x_edges, y_edges = np.histogram2d(
        ball_x,
        ball_y,
        bins=[bins_x, bins_y]
    )

    return {
        "heatmap": heatmap,
        "x_edges": x_edges,
        "y_edges": y_edges
    }


def pi4_reaction_intensity(X: pd.DataFrame):
    """
    PI 4 — Intensidade de Reação do Guarda-Redes
    Calcula a intensidade da reação com base na magnitude da velocidade.
    """

    vx = X["#vx0"].values
    vy = X["#vy0"].values

    speed_series = np.sqrt(vx**2 + vy**2)

    mean_speed = speed_series.mean()
    max_speed = speed_series.max()

    return {
        "speed_series": speed_series,
        "mean_speed": mean_speed,
        "max_speed": max_speed
    }




def pi5_threat_origin_zones(
    X: pd.DataFrame,
    bins_x: int = 10,
    bins_y: int = 10
):
    """
    PI 5 — Zona de Origem das Ameaças
    Analisa a distribuição espacial das posições iniciais da bola
    como proxy da origem das ações ofensivas.
    """

    ball_x = X["#ball_x"].values
    ball_y = X["#ball_y"].values

    heatmap, x_edges, y_edges = np.histogram2d(
        ball_x,
        ball_y,
        bins=[bins_x, bins_y]
    )

    return {
        "heatmap": heatmap,
        "x_edges": x_edges,
        "y_edges": y_edges
    }


import pandas as pd


# ==================================================
# PI 5 — CANAL DE PROGRESSÃO DAS AMEAÇAS OFENSIVAS
# ==================================================
def pi5_threat_progression_channels(X: pd.DataFrame):
    """
    PI 5 — Canal de Progressão das Ameaças Ofensivas

    Objetivo:
    Identificar por que corredor do campo (esquerdo, central, direito)
    o adversário construiu mais ameaças ofensivas.

    Persona: Treinador Principal
    Contexto: Pós-Jogo
    """

    if "#x0" not in X.columns:
        raise ValueError("Coluna '#x0' não encontrada para cálculo do PI 5.")

    # Classificação dos canais com base na coordenada X normalizada
    def classify_channel(x):
        if x < 0.33:
            return "Esquerdo"
        elif x < 0.66:
            return "Central"
        else:
            return "Direito"

    channels = X["#x0"].apply(classify_channel)

    # Contagem absoluta
    counts = channels.value_counts().reindex(
        ["Esquerdo", "Central", "Direito"],
        fill_value=0
    )

    total = counts.sum()

    # Percentagens
    percentages = (counts / total * 100) if total > 0 else counts

    return {
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
        "total_threats": int(total)
    }
