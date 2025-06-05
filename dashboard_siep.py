
# Dashboard Interativo de Regressão e ANOVA - Ames Housing (com base embutida)
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
import plotly.express as px

st.set_page_config(page_title="Dashboard - Ames Housing", layout="wide")

# Logo da UnB + título e subtítulo
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("Image (1).png", width=100)
with col_title:
    st.markdown("## Dashboard Interativo de Regressão e ANOVA - Ames Housing (com base embutida)")
    st.markdown("### Alunos: Juliano Teles Abrahao (231013411) - João Pedro Lima de Carvalho (231013402)")


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
    numericas = df.select_dtypes(include=np.number).columns
default_y = "SalePrice" if "SalePrice" in numericas else numericas[0]
y_var = st.selectbox("Selecione a variável Y", numericas, index=list(numericas).index(default_y))

fig = px.scatter(df, x=x_var, y=y_var, trendline="ols", title=f"{y_var} vs {x_var}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔬 ANOVA")
cat_var = st.selectbox("Selecione uma variável categórica para ANOVA", df.select_dtypes(include='object').columns)
from patsy import dmatrices

# Garante nomes válidos e evita SyntaxError
safe_cat = cat_var.replace(" ", "_").replace("-", "_")
df = df.rename(columns={cat_var: safe_cat})
safe_cat = cat_var.replace(" ", "_").replace("-", "_")
df = df.rename(columns={cat_var: safe_cat})

model_anova = ols(f'{y_var} ~ C({safe_cat})', data=df).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
st.dataframe(anova_table)

residuals = model_anova.resid
st.write("Teste de normalidade (Shapiro-Wilk):", stats.shapiro(residuals))
grouped = [df[y_var][df[safe_cat] == cat] for cat in df[safe_cat].dropna().unique()]
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
