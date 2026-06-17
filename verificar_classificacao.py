# -*- coding: utf-8 -*-
"""
Este script serve para testar e validar o seu Pipeline de Classificação (ficheiro .pkl).
Ele simula exatamente o "Blind Test" que o professor fará no dia da apresentação do projeto.
O código foi escrito de forma simples e está 100% comentado em português para facilitar a sua compreensão.
"""

# Importar as bibliotecas necessárias
import pandas as pd  # A biblioteca Pandas serve para manipular e ler dados em tabelas (DataFrames)
import pickle       # A biblioteca Pickle serve para carregar o modelo de IA que guardamos no ficheiro
import os           # A biblioteca OS serve para lidar com caminhos de ficheiros no sistema operacional

# 1. Definir os caminhos para os ficheiros
caminho_modelo = os.path.join("models", "pipeline_classification.pkl")
caminho_dados = os.path.join("data", "raw", "employee_data.csv")

print("=========================================================")
print("INICIANDO A VALIDAÇÃO DA PIPELINE DE CLASSIFICAÇÃO")
print("=========================================================\n")

# 2. Carregar o modelo treinado (Ficheiro .pkl)
# O modelo está guardado no formato binário (wb/rb). Usamos "rb" para ler (read binary).
print(f"-> A carregar o pipeline a partir de: {caminho_modelo}")
try:
    with open(caminho_modelo, "rb") as file:
        pipeline = pickle.load(file)
    print("[OK] Pipeline carregado com sucesso!\n")
except FileNotFoundError:
    print(f"[ERRO] O ficheiro '{caminho_modelo}' não foi encontrado. Certifique-se de que correu o treino antes.")
    exit(1)

# 3. Carregar o conjunto de dados bruto (raw CSV)
# Aqui lemos o arquivo original para pegar algumas linhas de exemplo para simular o teste.
print(f"-> A carregar os dados originais a partir de: {caminho_dados}")
df_original = pd.read_csv(caminho_dados)
print(f"[OK] Dados carregados! O dataset tem {df_original.shape[0]} funcionários e {df_original.shape[1]} colunas.\n")

# 4. Simular o "Blind Test"
# No Blind Test, o professor vai passar dados novos que não contêm a coluna "Attrition".
# Vamos retirar as primeiras 5 linhas e apagar a coluna "Attrition" para fingir que são dados novos do teste.
amostra_teste = df_original.head(5).copy()

# Guardar as respostas reais para podermos comparar no final
respostas_reais = amostra_teste["Attrition"].values

# Apagar a coluna "Attrition" da nossa amostra. O pipeline tem de conseguir trabalhar sem ela!
amostra_sem_target = amostra_teste.drop("Attrition", axis=1)

print("-> A enviar dados brutos para o Pipeline (Caixa Negra)...")
print("Nota: Os dados enviados contêm valores de texto e números sem qualquer tratamento manual!")

# 5. Gerar as previsões usando o Pipeline
# Como o pipeline é autossuficiente (contém o pré-processador e o modelo),
# nós podemos simplesmente passar os dados brutos usando a função .predict()
previsoes = pipeline.predict(amostra_sem_target)

# Também podemos obter as probabilidades de cada resposta.
# predict_proba() devolve uma lista com duas probabilidades: [Probabilidade de Ficar (No), Probabilidade de Sair (Yes)]
probabilidades = pipeline.predict_proba(amostra_sem_target)

# 6. Mostrar os Resultados de forma legível
print("\n=================== RESULTADOS DO TESTE ===================")
for i in range(len(previsoes)):
    prob_sair = probabilidades[i][1] * 100  # A probabilidade de sair está no índice 1 (classe 'Yes')
    print(f"\nFuncionário #{i+1}:")
    print(f"  - Previsão da IA: Abandona a empresa? {'SIM (Yes)' if previsoes[i] == 'Yes' else 'NÃO (No)'}")
    print(f"  - Probabilidade de abandono calculada pela IA: {prob_sair:.2f}%")
    print(f"  - Resposta Real (Gabarito): {'SIM' if respostas_reais[i] == 'Yes' else 'NÃO'}")

print("\n=========================================================")
print("[OK] Validação concluída! O seu pipeline está a funcionar como 'black box'.")
print("=========================================================")
