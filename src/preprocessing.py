import pandas as pd


# ==================================================
# VALIDAÇÃO DE DADOS
# ==================================================
def basic_validation(
    X: pd.DataFrame,
    y: pd.DataFrame | None = None,
    name: str = ""
) -> dict:
    """
    Validação básica de consistência dos dados.

    - Verifica dimensões
    - Alinhamento entre X e y
    - Valores nulos
    - Tipos de dados

    Retorna um relatório estruturado utilizável
    em dashboards, logs ou relatórios.
    """

    report = {
        "dataset": name,
        "X_shape": X.shape,
        "y_shape": y.shape if y is not None else None,
        "same_length": len(X) == len(y) if y is not None else None,
        "X_nulls": int(X.isnull().sum().sum()),
        "y_nulls": int(y.isnull().sum().sum()) if y is not None else None,
        "dtypes": X.dtypes.astype(str).to_dict()
    }

    return report


# ==================================================
# CONTEXTO DE ANÁLISE — TREINO vs JOGO
# ==================================================
def infer_game_context(X: pd.DataFrame) -> pd.DataFrame:
    """
    Infere o contexto de análise segundo o relatório M2:
    - Treino
    - Jogo

    Estratégia:
    - Se já existir uma coluna 'contexto', respeita-a
    - Caso contrário, assume todo o dataset como 'Jogo'

    Esta abordagem é:
    - Coerente com o relatório
    - Não especulativa
    - Facilmente extensível no futuro
    """

    X = X.copy()

    # Se o dataset já tiver contexto explícito, usar
    if "contexto" in X.columns:
        return X

    # Comportamento seguro por defeito
    X["contexto"] = "Jogo"

    return X
