import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Importar os dados do arquivo CSV
df = pd.read_csv('medical_examination.csv')

# 2: Adicionar a coluna 'overweight' (acima do peso)
# Calcula o IMC (Índice de Massa Corporal) dividindo o peso (kg) pela altura (m) ao quadrado
# Se o IMC for maior que 25, define como 1 (acima do peso), caso contrário, 0 (peso normal)
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3: Normalizar os dados de colesterol e glicose
# Se colesterol ou glicose for 1 (bom), define como 0
# Se for maior que 1 (ruim), define como 1
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4: Função para desenhar o gráfico categórico
def draw_cat_plot():
    # 5: Criar o DataFrame para o gráfico categórico usando pd.melt
    # Transforma os dados de colesterol, glicose, fumo, álcool, atividade física e sobrepeso em formato longo (long format)
    df_cat = pd.melt(df, 
                     id_vars=['cardio'],  # Variável que será usada para separar os gráficos
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])  # Variáveis que queremos visualizar

    # 6: Agrupar e reformular os dados
    # Agrupa os dados por cardio, variável e valor e conta a quantidade de cada combinação
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().reset_index(name='total')

    # 7: Desenhar o gráfico categórico
    # Cria o gráfico de barras para cada valor das variáveis categóricas, separado por cardio
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 8: Salvar o gráfico como 'catplot.png'
    fig.savefig('catplot.png')
    return fig

# 10: Função para desenhar o mapa de calor
def draw_heat_map():
    # 11: Limpeza dos dados
    # Filtrar dados onde a pressão diastólica (ap_lo) não pode ser maior que a sistólica (ap_hi)
    # Remover alturas e pesos que estão fora do intervalo entre o 2.5º e o 97.5º percentil
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12: Calcular a matriz de correlação
    # Calcula a correlação entre todas as variáveis numéricas do DataFrame
    corr = df_heat.corr()

    # 13: Gerar uma máscara para a parte superior do triângulo
    # Isso oculta a parte superior da matriz de correlação para evitar duplicidade
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Configurar o gráfico matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15: Desenhar o mapa de calor (heatmap)
    # Usamos o sns.heatmap para visualizar a matriz de correlação
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, cmap='coolwarm', cbar_kws={'shrink': .5})

    # 16: Salvar o mapa de calor como 'heatmap.png'
    fig.savefig('heatmap.png')
    return fig
