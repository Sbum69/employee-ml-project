import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Retorna o pipeline de pré-processamento (ColumnTransformer) para a classificação de Attrition.
    Este pipeline realiza a limpeza, imputação de nulos, normalização e codificação
    das colunas de forma totalmente automatizada.
    
    Parâmetros:
    -----------
    X : pd.DataFrame
        O DataFrame contendo as variáveis preditoras (sem a variável alvo Attrition).
        
    Retorna:
    --------
    ColumnTransformer
        O pipeline configurado pronto para ser acoplado a um classificador.
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
    # As variáveis numéricas passam pelas etapas de:
    #   a) Imputer: Substituição de valores nulos pela mediana da respectiva coluna.
    #   b) Scaler: Normalização padrão (Média=0, Variância=1).
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # -------------------------------------------------------------------------
    # 3. Pipeline para Variáveis Categóricas
    # -------------------------------------------------------------------------
    # As variáveis categóricas de texto passam por:
    #   a) Imputer: Preenchimento de nulos com o valor mais comum (moda).
    #   b) Encoder: Codificação do texto em colunas binárias via One-Hot Encoding.
    #      A configuração handle_unknown="ignore" evita que o pipeline quebre no Blind Test
    #      se surgirem categorias de texto novas não observadas no treino.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # -------------------------------------------------------------------------
    # 4. Agrupamento em ColumnTransformer
    # -------------------------------------------------------------------------
    # Mapeia as transformações correspondentes para suas respectivas colunas.
    # O restante das colunas não identificadas é descartado por padrão (remainder='drop').
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor
