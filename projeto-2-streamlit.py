import streamlit as st
import matplotlib.pyplot as plt

# Título do App
st.title("Projeto 2 - Conversão para Streamlit")

# Seu código começa aqui:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree




renda = pd.read_csv('./projeto 2/input/previsao_de_renda.csv')

renda.info()
renda

renda.nunique()

renda.drop(columns=['Unnamed: 0', 'id_cliente'], inplace=True)

st.write('Quantidade total de linhas:', len(renda), '\n')

st.write('Quantidade de linhas duplicadas:', renda.duplicated().sum(), '\n')

st.write('Quantidade após remoção das linhas duplicadas:', 
      len(renda.drop_duplicates()), '\n')

renda.drop_duplicates(inplace=True, ignore_index=True)
renda.info()

prof = ProfileReport(df=renda, 
                     minimal=False, 
                     explorative=True)
os.makedirs(name='./output', exist_ok=True)
prof.to_file('./output/renda_analysis.html')

prof


renda.describe().transpose()

renda_numerico = renda.iloc[:, 3:].select_dtypes(include=np.number)

# Calcula a correlação
correlacao = renda_numerico.corr()

# Seleciona a última linha
ultima_linha_correlacao = correlacao.tail(n=1)

st.write(ultima_linha_correlacao)

# Personalização do pairplot
sns.pairplot(data=renda, 
             hue='tipo_renda', 
             vars=['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia', 'renda'], 
             diag_kind='kde', # ou 'hist'
             palette='viridis', # Escolha uma paleta de cores
             plot_kws={'alpha': 0.7}) # Ajuste a transparência dos pontos

# Adicionando título ao gráfico
plt.suptitle('Pairplot das Variáveis Quantitativas por Tipo de Renda', fontsize=16)

# Ajustando o tamanho da figura (opcional)
plt.gcf().set_size_inches(12, 12)

# Exibindo o gráfico
#plt.show() # Não necessário em Jupyter Notebook

#Identificando as colunas não numéricas
colunas_nao_numericas = renda.select_dtypes(exclude=np.number).columns

#Criando um novo dataframe apenas com colunas numéricas.
renda_numerico = renda.drop(columns=colunas_nao_numericas)

#Calculando a correlação
correlacao = renda_numerico.corr()

#Criando o cmap
cmap = sns.diverging_palette(h_neg=100,
                                h_pos=359,
                                as_cmap=True,
                                sep=1,
                                center='light')

#Criando o clustermap
ax = sns.clustermap(data=correlacao,
                    figsize=(10, 10),
                    center=0,
                    cmap=cmap)

#Rotacionando os labels
plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

#Exibindo o gráfico
plt.show()

# Configuração do tamanho da figura
plt.figure(figsize=(16, 9))

# Scatterplot com cores diferenciadas por tipo de renda e tamanho por idade
sns.scatterplot(x='tempo_emprego',
                y='renda',
                hue='tipo_renda',
                size='idade',
                data=renda,
                alpha=0.6, # Aumentando a transparência para melhor visualização
                sizes=(20, 200), # Ajustando a escala do tamanho dos pontos
                palette='viridis') # Escolhendo uma paleta de cores adequada

# Linha de tendência
sns.regplot(x='tempo_emprego',
            y='renda',
            data=renda,
            scatter=False,
            color='red', # Mudando a cor da linha de tendência
            line_kws={'linewidth': 2}) # Ajustando a espessura da linha de tendência

# Adicionando título e rótulos dos eixos
plt.title('Renda vs. Tempo de Emprego por Tipo de Renda e Idade', fontsize=16)
plt.xlabel('Tempo de Emprego (anos)', fontsize=12)
plt.ylabel('Renda', fontsize=12)

# Melhorando a legenda
plt.legend(title='Tipo de Renda e Idade')

# Exibindo o gráfico
#plt.show() # Não necessário em Jupyter Notebook

# Configuração do tamanho da figura
plt.rc('figure', figsize=(12, 4))

# Criação dos subplots
fig, axes = plt.subplots(nrows=1, ncols=2)

# Pointplot para posse de imóvel
sns.pointplot(x='posse_de_imovel',
              y='renda',
              data=renda,
              dodge=True,
              ax=axes[0],
              color='skyblue',  # Escolhendo uma cor
              markers='o',  # Escolhendo o marcador
              linestyles='-',  # Escolhendo o estilo da linha
              errorbar=('ci', 95))  # Atualizando o parâmetro ci
axes[0].set_title('Renda por Posse de Imóvel')  # Adicionando título ao subplot
axes[0].set_xlabel('Posse de Imóvel')  # Adicionando label ao eixo x
axes[0].set_ylabel('Renda')  # Adicionando label ao eixo y
axes[0].set_xticks([0, 1])  # Definindo os ticks
axes[0].set_xticklabels(['Não', 'Sim'])  # Alterando os labels do eixo x

# Pointplot para posse de veículo
sns.pointplot(x='posse_de_veiculo',
              y='renda',
              data=renda,
              dodge=True,
              ax=axes[1],
              color='salmon',  # Escolhendo uma cor
              markers='o',  # Escolhendo o marcador
              linestyles='-',  # Escolhendo o estilo da linha
              errorbar=('ci', 95))  # Atualizando o parâmetro ci
axes[1].set_title('Renda por Posse de Veículo')  # Adicionando título ao subplot
axes[1].set_xlabel('Posse de Veículo')  # Adicionando label ao eixo x
axes[1].set_ylabel('Renda')  # Adicionando label ao eixo y
axes[1].set_xticks([0, 1])  # Definindo os ticks
axes[1].set_xticklabels(['Não', 'Sim'])  # Alterando os labels do eixo x

# Adicionando um título geral ao gráfico
fig.suptitle('Renda Média por Posse de Imóvel e Veículo', fontsize=16)

# Ajustando o layout para evitar sobreposição de elementos
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Exibindo o gráfico
plt.show()  # Não necessário em Jupyter Notebook

def plot_qualitative_variables(renda, col):
    """
    Plota gráficos de barras empilhadas e perfis médios no tempo para uma variável qualitativa.

    Args:
        renda (pd.DataFrame): DataFrame contendo os dados.
        col (str): Nome da coluna qualitativa.
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    fig.subplots_adjust(wspace=0.6)

    # Formatação das datas para exibição
    tick_labels = renda['data_ref'].dt.strftime('%b/%Y').unique()

    # Barras empilhadas
    renda_crosstab = pd.crosstab(
        index=renda['data_ref'], columns=renda[col], normalize='index'
    )
    ax0 = renda_crosstab.plot.bar(stacked=True, ax=axes[0], colormap='viridis') #colormap adicionado
    ax0.set_xticks(range(len(tick_labels)))  # Defina os ticks aqui
    ax0.set_xticklabels(labels=tick_labels, rotation=45, ha='right') #ha='right' adicionado para melhor visualização
    axes[0].legend(bbox_to_anchor=(1, 0.5), loc='center left', title=f"'{col}'")
    axes[0].set_title(f'Distribuição de {col} ao Longo do Tempo')
    axes[0].set_xlabel('Data de Referência')
    axes[0].set_ylabel('Proporção')

    # Perfis médios no tempo
    ax1 = sns.pointplot(
        x='data_ref', y='renda', hue=col, data=renda, dodge=True, errorbar=('ci', 95), ax=axes[1], palette='Set2' #palette adicionado
    )
    ax1.set_xticks(range(len(tick_labels)))  # Defina os ticks aqui
    ax1.set_xticklabels(labels=tick_labels, rotation=45, ha='right') #ha='right' adicionado para melhor visualização
    axes[1].legend(bbox_to_anchor=(1, 0.5), loc='center left', title=f"'{col}'")
    axes[1].set_title(f'Renda Média por {col} ao Longo do Tempo')
    axes[1].set_xlabel('Data de Referência')
    axes[1].set_ylabel('Renda Média')

    plt.tight_layout() #melhora o layout para evitar sobreposição
    #plt.show() # Não necessário em Jupyter Notebook

# Conversão da coluna 'data_ref' para datetime
renda['data_ref'] = pd.to_datetime(renda['data_ref'])

# Seleção das colunas qualitativas
qualitativas = renda.select_dtypes(include=['object', 'boolean']).columns

# Plotagem dos gráficos para cada variável qualitativa
for col in qualitativas:
    plot_qualitative_variables(renda, col)

renda.drop(columns='data_ref', inplace=True)
renda.dropna(inplace=True)

pd.DataFrame(index=renda.nunique().index, 
             data={'tipos_dados': renda.dtypes, 
                   'qtd_valores': renda.notna().sum(), 
                   'qtd_categorias': renda.nunique().values})

renda_dummies = pd.get_dummies(data=renda)
renda_dummies.info()

# Calcula a correlação das variáveis com a coluna 'renda'
correlacoes = renda_dummies.corr()['renda'].sort_values(ascending=False)

# Converte a série de correlações em um DataFrame
correlacoes_df = correlacoes.to_frame().reset_index()

# Renomeia as colunas para 'var' e 'corr'
correlacoes_df = correlacoes_df.rename(columns={'index': 'var', 'renda': 'corr'})

# Aplica o estilo de barra ao DataFrame
correlacoes_estilizadas = correlacoes_df.style.bar(color=['darkred', 'darkgreen'], align=0)

# Exibe o DataFrame estilizado
correlacoes_estilizadas

# Separação das variáveis independentes (X) e dependente (y)
X = renda_dummies.drop(columns='renda')
y = renda_dummies['renda']

# Exibição das dimensões de X e y
st.write(f'Quantidade de linhas e colunas de X: {X.shape}')
st.write(f'Quantidade de linhas de y: {len(y)}')

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Exibição das dimensões dos conjuntos de treino e teste
st.write(f'X_train: {X_train.shape}')
st.write(f'X_test: {X_test.shape}')
st.write(f'y_train: {y_train.shape}')
st.write(f'y_test: {y_test.shape}')

score = pd.DataFrame(columns=['max_depth', 'min_samples_leaf', 'score'])

for x in range(1, 21):
    for y in range(1, 31):
        reg_tree = DecisionTreeRegressor(random_state=42, 
                                         max_depth=x, 
                                         min_samples_leaf=y)
        reg_tree.fit(X_train, y_train)
        
        score = pd.concat(objs=[score, 
                                pd.DataFrame({'max_depth': [x], 
                                              'min_samples_leaf': [y], 
                                              'score': [reg_tree.score(X=X_test, 
                                                                       y=y_test)]})], 
                          axis=0, 
                          ignore_index=True)
        
score.sort_values(by='score', ascending=False)

reg_tree = DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_leaf=4)
reg_tree.fit(X_train, y_train)

# Configuração do tamanho da figura
plt.rc('figure', figsize=(18, 9))

# Plotagem da árvore de decisão
tp = tree.plot_tree(decision_tree=reg_tree, 
                    feature_names=X.columns, 
                    filled=True)

# Exibição do gráfico
plt.show()

# Exportação da árvore de decisão em formato de texto
text_tree_print = tree.export_text(decision_tree=reg_tree)

# Impressão da árvore de decisão
st.write(text_tree_print)


# Cálculo do coeficiente de determinação (R²) para os conjuntos de treino e teste
r2_train = reg_tree.score(X_train, y_train)
r2_test = reg_tree.score(X_test, y_test)

# Template para exibição dos resultados
template = 'O coeficiente de determinação (R²) da árvore com profundidade = {0} para a base de {1} é: {2:.2f}'

# Impressão dos resultados com substituição de ponto por vírgula
st.write(template.format(reg_tree.get_depth(), 'treino', r2_train).replace(".", ","))
st.write(template.format(reg_tree.get_depth(), 'teste', r2_test).replace(".", ","), '\n')

# Adiciona a coluna 'renda_predict' ao DataFrame 'renda' com as previsões arredondadas
renda['renda_predict'] = np.round(reg_tree.predict(X), 2)

# Seleciona e exibe as colunas 'renda' e 'renda_predict'
resultado = renda[['renda', 'renda_predict']]

# Exibe o DataFrame resultante
st.write(resultado)

# Dados de entrada
entrada = pd.DataFrame([{
    'sexo': 'M',
    'posse_de_veiculo': False,
    'posse_de_imovel': True,
    'qtd_filhos': 1,
    'tipo_renda': 'Assalariado',
    'educacao': 'Superior completo',
    'estado_civil': 'Solteiro',
    'tipo_residencia': 'Casa',
    'idade': 34,
    'tempo_emprego': None,
    'qt_pessoas_residencia': 1
}])

# Processamento dos dados
entrada = pd.concat([X, pd.get_dummies(entrada)]).fillna(0).tail(1)

# Previsão
renda_estimada = np.round(reg_tree.predict(entrada)[0], 2)

# Apresentação do resultado
st.write(f"Renda estimada: R$ {renda_estimada:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))


