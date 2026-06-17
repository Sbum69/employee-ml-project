import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Retorna o pipeline de pré-processamento (ColumnTransformer) para os dados de renda.
    Este pipeline realiza a limpeza, imputação de nulos, normalização e codificação
    das colunas categóricas e numéricas de forma totalmente automatizada.
    
    Parâmetros:
    -----------
    X : pd.DataFrame
        O DataFrame contendo as variáveis preditoras (sem a variável alvo MonthlyIncome).
        
    Retorna:
    --------
    ColumnTransformer
        O pipeline configurado pronto para ser acoplado a um modelo de Regressão.
    """
    # -------------------------------------------------------------------------
    # 1. Separação Automática de Tipos de Colunas
    # -------------------------------------------------------------------------
    # Seleciona todas as colunas com tipos numéricos (int64 ou float64)
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    
    # Seleciona todas as colunas categóricas/texto (tipo object)
    categorical_features = X.select_dtypes(include=["object"]).columns
    
    # -------------------------------------------------------------------------
    # 2. Pipeline para Variáveis Numéricas
    # -------------------------------------------------------------------------
    # As variáveis numéricas passam por duas etapas sequenciais:
    #   a) Imputer: Preenche eventuais valores em falta usando a mediana da coluna.
    #   b) Scaler: Normaliza os valores (média=0, desvio padrão=1) para ajudar algoritmos
    #      lineares e baseados em distância a convergir mais rápido.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # -------------------------------------------------------------------------
    # 3. Pipeline para Variáveis Categóricas
    # -------------------------------------------------------------------------
    # As variáveis categóricas passam por duas etapas sequenciais:
    #   a) Imputer: Preenche valores em falta usando a categoria mais frequente (moda).
    #   b) Encoder: Converte as categorias de texto em colunas binárias (One-Hot Encoding).
    #      Usamos handle_unknown="ignore" para que o pipeline não falhe (não dê erro)
    #      caso apareça alguma nova categoria inesperada durante o Blind Test.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # -------------------------------------------------------------------------
    # 4. Agrupamento dos Processadores (ColumnTransformer)
    # -------------------------------------------------------------------------
    # Aplica as transformações numéricas e categóricas apenas nas colunas corretas.
    # Qualquer outra coluna não listada aqui (remainder='drop') é ignorada,
    # agindo como um seletor e garantindo a robustez do pipeline de Blind Test.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor
