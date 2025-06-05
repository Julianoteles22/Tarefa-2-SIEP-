
# Dashboard Interativo de Regressão e ANOVA - Ames Housing (com base embutida)
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
import plotly.express as px

st.set_page_config(page_title="Dashboard - Ames Housing", layout="wide")

st.title("📊 Dashboard Interativo - Análise de Preços de Imóveis (Ames Housing)")

# Leitura direta do arquivo já embutido no repositório
df = pd.read_csv("AmesHousing.csv")

# Ajustes nos nomes de colunas
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'Overall Qual': 'OverallQual',
    'Fireplace Qu': 'FireplaceQu',
    'Gr Liv Area': 'GrLivArea',
    'Garage Area': 'GarageArea',
    'Kitchen Qual': 'KitchenQual'
})

st.subheader("🔎 Visualização dos Dados")
st.dataframe(df.head())

st.subheader("📈 Gráficos Interativos")
col1, col2 = st.columns(2)
with col1:
    x_var = st.selectbox("Selecione a variável X (numérica)", df.select_dtypes(include=np.number).columns)
with col2:
    y_var = st.selectbox("Selecione a variável Y", df.select_dtypes(include=np.number).columns, index=65)  # SalePrice default

fig = px.scatter(df, x=x_var, y=y_var, trendline="ols", title=f"{y_var} vs {x_var}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔬 ANOVA")
cat_var = st.selectbox("Selecione uma variável categórica para ANOVA", df.select_dtypes(include='object').columns)
model_anova = ols(f'{y_var} ~ C({cat_var})', data=df).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
st.write("Tabela ANOVA:")
st.dataframe(anova_table)

residuals = model_anova.resid
st.write("Teste de normalidade (Shapiro-Wilk):", stats.shapiro(residuals))
grouped = [df[y_var][df[cat_var] == cat] for cat in df[cat_var].dropna().unique()]
st.write("Teste de homocedasticidade (Levene):", stats.levene(*grouped))

st.subheader("📉 Regressão Linear Múltipla")
num_vars = st.multiselect("Selecione variáveis numéricas para regressão", df.select_dtypes(include=np.number).columns.drop(y_var), default=['GrLivArea', 'GarageArea', 'OverallQual'])
df_model = df[[y_var] + num_vars].dropna()
X = sm.add_constant(df_model[num_vars])
y = np.log(df_model[y_var])
model = sm.OLS(y, X).fit()
st.text(model.summary())

y_pred = model.predict(X)
st.write(f"R²: {model.rsquared:.4f}")
st.write(f"RMSE: {np.sqrt(np.mean((y - y_pred)**2)):.4f}")
st.write(f"MAE: {np.mean(np.abs(y - y_pred)):.4f}")
