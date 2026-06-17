import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Importa o pré-processador modular definido no projeto
from src.preprocessing.attrition_preprocessing import get_preprocessing_pipeline


def train_and_save_pipeline():
    """
    Treina o pipeline completo de classificação para prever o abandono de
    colaboradores (Attrition) e exporta dois ficheiros .pkl.

    Fluxo:
        1. Carregamento dos dados brutos
        2. Remoção de colunas irrelevantes (variância zero)
        3. Divisão estratificada treino/teste (80/20)
        4. Construção do pré-processador + modelo (LogisticRegression balanced)
        5. Treino do pipeline completo
        6. Avaliação: Accuracy + F1-Score (classe 'Yes')
        7. Exportação do pipeline serializado (.pkl)
    """
    # Caminhos de leitura do CSV e gravação dos ficheiros de modelo
    data_path = "data/raw/employee_data.csv"
    model_output_path = "models/pipeline_classification.pkl"
    group_output_path = "models/Grupo16_pipeline_classification.pkl"

    # -------------------------------------------------------------------------
    # 1. Carregamento dos Dados
    # -------------------------------------------------------------------------
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset não encontrado em: {data_path}")

    print("Carregando os dados para classificação...")
    df = pd.read_csv(data_path)

    # -------------------------------------------------------------------------
    # 2. Definição de Variáveis (X e y)
    # -------------------------------------------------------------------------
    # Removemos a variável alvo 'Attrition' e as colunas de variância zero
    # identificadas na EDA:
    #   - 'EmployeeCount' (sempre 1)
    #   - 'StandardHours' (sempre 80)
    #   - 'Over18'        (sempre 'Y')
    # Nota: 'EmployeeNumber' é apenas um identificador único sem valor preditivo.
    colunas_a_remover = ["Attrition", "EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    X = df.drop(columns=colunas_a_remover)
    y = df["Attrition"]  # Alvo: 'Yes' (saiu) ou 'No' (ficou)

    # -------------------------------------------------------------------------
    # 3. Divisão Treino e Teste com Estratificação
    # -------------------------------------------------------------------------
    # stratify=y garante que a proporção de 'Yes'/'No' se mantém igual nos
    # dois conjuntos, essencial dado o desequilíbrio das classes (~84% No / ~16% Yes).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------------------------------------------
    # 4. Construção do Pré-processador e Modelo
    # -------------------------------------------------------------------------
    print("Construindo o pré-processador de classificação...")
    preprocessor = get_preprocessing_pipeline(X)

    # Modelo final: Regressão Logística com class_weight="balanced".
    #
    # Justificação:
    #   - O dataset é desequilibrado (~84% No, ~16% Yes). Sem balanceamento,
    #     o modelo ignora a classe minoritária 'Yes'.
    #   - class_weight="balanced" aumenta automaticamente o peso de 'Yes'
    #     proporcionalmente à sua sub-representação, melhorando o F1-Score
    #     e o Recall para a classe de interesse (colaboradores que saem).
    #   - A Regressão Logística foi preferida ao Random Forest porque
    #     demonstrou melhor F1-Score para 'Yes' neste problema específico.
    #   - max_iter=1000 garante convergência mesmo após o aumento do peso.
    print("Construindo o modelo final (Logistic Regression com class_weight='balanced')...")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Compensa o desequilíbrio de classes
        random_state=42
    )

    # -------------------------------------------------------------------------
    # 5. Criação do Pipeline Integrado (Requisito Obrigatório do Projeto)
    # -------------------------------------------------------------------------
    # O Pipeline encapsula pré-processamento + modelo num único objeto.
    # Ao chamar pipeline.predict(novos_dados), o scikit-learn aplica
    # automaticamente todas as transformações antes de gerar a predição.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # Imputer + Encoder + Scaler
            ("classifier", classifier)         # Modelo final de classificação
        ]
    )

    # -------------------------------------------------------------------------
    # 6. Treino do Pipeline Completo
    # -------------------------------------------------------------------------
    print("Treinando o pipeline completo de classificação...")
    pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 7. Avaliação das Métricas de Performance
    # -------------------------------------------------------------------------
    # A métrica principal do Blind Test é o F1-Score para a classe 'Yes'.
    # Reportamos também a Accuracy e o Classification Report completo.
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="Yes")

    print(f"\nDesempenho no conjunto de teste:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score (Yes): {f1:.4f}  <- métrica principal do Blind Test")
    print("\nRelatório de Classificação completo:")
    print(classification_report(y_test, y_pred))

    # -------------------------------------------------------------------------
    # 8. Exportação do Pipeline Serializado (.pkl)
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Exportação com nome genérico
    print(f"\nExportando o pipeline para {model_output_path}...")
    with open(model_output_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Exportação com nome do grupo (Grupo16) — ficheiro a entregar
    print(f"Exportando o pipeline para {group_output_path}...")
    with open(group_output_path, "wb") as f:
        pickle.dump(pipeline, f)

    print("\nSucesso! Pipelines de classificação guardados com êxito.")
    return pipeline


if __name__ == "__main__":
    train_and_save_pipeline()