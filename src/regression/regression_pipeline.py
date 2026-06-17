import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# Importa o pré-processador modular definido no diretório do projeto
from src.preprocessing.income_preprocessing import get_preprocessing_pipeline


def train_and_save_pipeline():
    """
    Treina o pipeline completo de regressão para prever o salário mensal
    (MonthlyIncome) e exporta dois ficheiros .pkl.

    Fluxo:
        1. Carregamento dos dados brutos
        2. Remoção de colunas irrelevantes (variância zero + identificadores)
        3. Divisão treino/teste (80/20)
        4. Construção do pré-processador + GradientBoostingRegressor (otimizado)
        5. Treino do pipeline completo
        6. Avaliação: R2 Score + RMSE
        7. Exportação do pipeline serializado (.pkl)
    """
    # Caminhos para leitura do CSV e gravação dos ficheiros de modelo
    data_path = "data/raw/employee_data.csv"
    model_output_path = "models/pipeline_regression.pkl"
    group_output_path = "models/Grupo16_pipeline_regression.pkl"

    # -------------------------------------------------------------------------
    # 1. Carregamento dos Dados
    # -------------------------------------------------------------------------
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset não encontrado em: {data_path}")

    print("Carregando os dados brutos...")
    df = pd.read_csv(data_path)

    # -------------------------------------------------------------------------
    # 2. Engenharia e Seleção de Variáveis
    # -------------------------------------------------------------------------
    # Removemos a variável alvo e colunas sem valor preditivo identificadas na EDA:
    #   - 'EmployeeCount'  → constante (sempre 1)
    #   - 'StandardHours'  → constante (sempre 80)
    #   - 'Over18'         → constante (sempre 'Y')
    #   - 'EmployeeNumber' → identificador único sem relação com o salário
    colunas_a_remover = ["MonthlyIncome", "EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    X = df.drop(columns=colunas_a_remover)
    y = df["MonthlyIncome"]  # Variável alvo: salário mensal em USD

    # -------------------------------------------------------------------------
    # 3. Divisão do Dataset em Treino e Teste
    # -------------------------------------------------------------------------
    # 80% treino / 20% teste. random_state=42 garante reprodutibilidade total.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------------------------------------------------
    # 4. Construção do Pré-processador e Modelo Final
    # -------------------------------------------------------------------------
    print("Construindo o pré-processador...")
    preprocessor = get_preprocessing_pipeline(X)

    # Modelo final: Gradient Boosting Regressor com hiperparâmetros otimizados via GridSearchCV.
    #
    # Resultados da comparação de modelos (notebook 03):
    #   - Linear Regression    → R2 ≈ 0.9443
    #   - Random Forest        → R2 ≈ 0.9505
    #   - Gradient Boosting ✓  → R2 ≈ 0.9523, RMSE ≈ 1005.90 (MELHOR)
    #
    # Hiperparâmetros selecionados pelo GridSearchCV (cv=5, scoring='r2'):
    #   - learning_rate=0.05  → passo de atualização conservador, evita overfitting
    #   - max_depth=3         → árvores rasas reduzem variância
    #   - n_estimators=100    → 100 árvores sequenciais em boosting
    print("Construindo o modelo final (Gradient Boosting Regressor otimizado)...")
    regressor = GradientBoostingRegressor(
        learning_rate=0.05,  # Taxa de aprendizagem (step size)
        max_depth=3,          # Profundidade máxima de cada árvore fraca
        n_estimators=100,     # Número de iterações de boosting
        random_state=42       # Reprodutibilidade
    )

    # -------------------------------------------------------------------------
    # 5. Criação do Pipeline Integrado (Requisito Obrigatório do Projeto)
    # -------------------------------------------------------------------------
    # O Pipeline encapsula o pré-processamento inteiro + o modelo final num único objeto.
    # pipeline.predict(novos_dados) aplica automaticamente todas as transformações
    # (imputer, scaler, encoder) antes de calcular a predição.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # Imputer + Scaler + OneHotEncoder
            ("regressor", regressor)           # GradientBoostingRegressor otimizado
        ]
    )

    # -------------------------------------------------------------------------
    # 6. Treino do Pipeline Completo
    # -------------------------------------------------------------------------
    print("Treinando o pipeline completo de regressão...")
    pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 7. Avaliação das Métricas de Performance
    # -------------------------------------------------------------------------
    # Métricas de avaliação do Blind Test: R2 e RMSE.
    #   - R2 (Coeficiente de Determinação): quanto da variância de MonthlyIncome
    #     é explicada pelo modelo. Varia de 0 a 1; valores > 0.9 são excelentes.
    #   - RMSE (Root Mean Squared Error): erro médio em USD. Quanto menor, melhor.
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"\nDesempenho no conjunto de teste:")
    print(f"  R2 Score : {r2:.4f}  <- quanto da variância o modelo explica")
    print(f"  RMSE     : {rmse:.2f} USD  <- erro médio em USD")

    # -------------------------------------------------------------------------
    # 8. Exportação do Pipeline Serializado (.pkl)
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Exportação com nome genérico
    print(f"\nExportando o pipeline para {model_output_path}...")
    with open(model_output_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Exportação com nome do grupo (Grupo16) — ficheiro a entregar no Moodle
    print(f"Exportando o pipeline para {group_output_path}...")
    with open(group_output_path, "wb") as f:
        pickle.dump(pipeline, f)

    print("\nSucesso! Pipelines de regressão guardados com êxito.")
    return pipeline


if __name__ == "__main__":
    train_and_save_pipeline()