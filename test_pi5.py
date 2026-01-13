from src.data_loading import load_datasets
from src.preprocessing import infer_game_context
from src.kpis import pi5_threat_progression_channels
from src.visualizations import plot_pi5_threat_progression_channels

# 1. Carregar dados
data = load_datasets()
X = data["X_train"]

# 2. Aplicar contexto (igual ao Streamlit)
X_contextual = infer_game_context(X)

# 3. Simular persona do Treinador Principal em Pós-Jogo
# Pós-Jogo usa dados do contexto "Jogo"
X_persona = X_contextual[X_contextual["contexto"] == "Jogo"]

# 4. Calcular PI5
pi5 = pi5_threat_progression_channels(X_persona)

# 5. Visualizar
fig = plot_pi5_threat_progression_channels(pi5)
fig.show()
