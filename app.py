import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Configuração do Streamlit
st.set_page_config(page_title="Previsão de Rendas", page_icon="📊", layout="wide")
# Centraliza o título
st.markdown("<h1 style='text-align: center;'>Previsão de Renda</h1>", unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('./input/previsao_de_renda.csv').drop(columns=['Unnamed: 0'])
    return df
renda = load_data()

# Exibir informações básicas
st.subheader("📊 Informações sobre os dados")
st.write(renda.info())
st.write(renda.head())

# Matriz de correlação
st.subheader("📈 Matriz de Correlação")
renda_numerico = renda.select_dtypes(include=np.number)
correlacao = renda_numerico.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("- O clustermap reforça a observação de baixa correlação entre a maioria das variáveis e a `renda`. Destaca-se a variável `tempo_emprego`, que apresenta uma correlação positiva moderada com a renda, sugerindo que indivíduos com mais tempo de emprego tendem a ter rendas mais altas. As variáveis booleanas `posse_de_imovel` e `posse_de_veiculo` mostram correlações fracas com a renda, indicando que a posse desses bens tem pouca influência na renda dos indivíduos na amostra.")

st.text("")

# Matriz de dispersão
st.subheader("📊 Matriz de Dispersão")
fig = sns.pairplot(renda_numerico).fig  # Cria a figura a partir do PairGrid
st.pyplot(fig)  # Exibe a figura no Streamlit
st.subheader("- A análise do pairplot, que exibe a matriz de dispersão entre todas as variáveis quantitativas, revela a presença de outliers na variável renda. Esses outliers, embora raros, podem distorcer a análise de tendência, exigindo consideração cuidadosa em modelagens futuras. Além disso, a baixa densidade de pontos fora da diagonal principal do pairplot sugere uma fraca correlação linear entre as variáveis quantitativas. Essa observação corrobora os resultados da matriz de correlação, que também indicou baixos coeficientes de correlação entre as mesmas variáveis.")

st.text("")

# Gráfico de dispersão
st.subheader("📉 Renda vs. Tempo de Emprego por Tipo de Renda e Idade")
fig, ax = plt.subplots(figsize=(16, 9))
sns.scatterplot(x='tempo_emprego', y='renda', hue='tipo_renda', size='idade', data=renda,
                alpha=0.6, sizes=(20, 200), palette='viridis', ax=ax)
sns.regplot(x='tempo_emprego', y='renda', data=renda, scatter=False, color='red',
            line_kws={'linewidth': 2}, ax=ax)
st.pyplot(fig)

st.markdown(
    "<div style='text-align: center;'>"
    "<h2>Renda vs. Tempo de Emprego por Tipo de Renda e Idade</h2>"
    "</div>",
    unsafe_allow_html=True
)
st.subheader("""
             
Este gráfico explora a relação entre renda e tempo de emprego, diferenciando por tipo de renda e idade.

**Observações Principais:**

* **Tendência Positiva:** A renda tende a aumentar com o tempo de emprego, conforme indicado pela linha de tendência.
* **Dispersão:** A alta dispersão dos pontos sugere que outros fatores, além do tempo de emprego, influenciam a renda.
* **Diferenciação:** A diferenciação por tipo de renda e idade revela padrões distintos na relação entre essas variáveis e a renda.
* **Outliers:** Pontos com renda excepcionalmente alta destacam casos de sucesso notáveis.

**Pontos-chave:**

* A linha de tendência mostra uma correlação positiva entre tempo de emprego e renda.
* A distribuição dos pontos indica que a renda é influenciada por múltiplos fatores.
* A legenda detalha os tipos de renda e as faixas de idade representadas no gráfico.
* Outliers sugerem a presença de indivíduos com rendas significativamente acima da média." "
""")

# Gerar Pandas Profiling Report
st.header("📑 Pandas Profiling Report")
if "profiling_report" not in st.session_state:
    prof = ProfileReport(df=renda, minimal=False, explorative=True)
    output_path = './output/renda_analysis.html'
    prof.to_file(output_path)
    st.session_state.profiling_report_path = output_path

# Exibir o relatório
with open(st.session_state.profiling_report_path, 'r', encoding='utf-8') as f:
    html_content = f.read()
st.components.v1.html(html_content, height=1000, scrolling=True)

# Modelagem de dados
st.header("🤖 Modelagem de dados")
X = renda_numerico.drop('renda', axis=1)
y = renda['renda']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de Regressão
st.subheader("🌳 Modelos de Regressão")

@st.cache_resource
def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse
st.write(f"Erro Quadrático Médio (MSE) - Árvore de Decisão: {train_decision_tree(X_train, y_train, X_test, y_test)}")

@st.cache_resource
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse
st.write(f"Erro Quadrático Médio (MSE) - Floresta Aleatória: {train_random_forest(X_train, y_train, X_test, y_test)}")

# Otimização de Hiperparâmetros
st.subheader("⚙️ Otimização de Hiperparâmetros")
if st.button("Otimizar Modelo"):
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    st.session_state.best_params = grid_search.best_params_
    st.session_state.best_score = grid_search.best_score_
    st.rerun()

# Exibir os melhores parâmetros
if "best_params" in st.session_state:
    st.write(f"Melhores parâmetros: {st.session_state.best_params}")
    st.write(f"Melhor pontuação: {st.session_state.best_score}")