"""
Dashboard Streamlit — Exploração do dataset dog-and-cat-data
Para rodar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
from pathlib import Path

st.set_page_config(page_title="Dog & Cat Data — Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Imagem de capa (fundo da página)
# ---------------------------------------------------------------------------
import base64

CAPA_PATH = "assets/capa.png"

def set_background(image_path: str):
    path = Path(image_path)
    if not path.exists():
        st.warning(f"Imagem de capa não encontrada em: {image_path}")
        return
    encoded = base64.b64encode(path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Fundo semi-transparente atrás do conteúdo, pra manter o texto legível */
        .stApp > header {{
            background-color: transparent;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            padding: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background(CAPA_PATH)

# ---------------------------------------------------------------------------
# Carregamento dos dados (com cache para não baixar/ler de novo a cada clique)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    dataset_path = Path(kagglehub.dataset_download("newbie2016/dog-and-cat-data"))

    # Procura o primeiro arquivo tabular (csv ou excel) dentro da pasta baixada
    candidates = list(dataset_path.rglob("*.csv")) + \
                 list(dataset_path.rglob("*.xlsx")) + \
                 list(dataset_path.rglob("*.xls"))

    if not candidates:
        return None, dataset_path

    file_path = candidates[0]
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    return df, file_path


st.title("🐶🐱 Dashboard — Dog & Cat Data")

with st.spinner("Carregando dados..."):
    df, source_path = load_data()

if df is None:
    st.error(f"Nenhum arquivo .csv/.xlsx encontrado em: {source_path}")
    st.stop()

st.caption(f"Fonte: `{source_path.name}` — {df.shape[0]} linhas × {df.shape[1]} colunas")

# ---------------------------------------------------------------------------
# Sidebar — filtros dinâmicos baseados no tipo de cada coluna
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")

SIDEBAR_IMAGE_PATH = "assets/sidebar.png"

def set_sidebar_background(image_path: str):
    path = Path(image_path)
    if not path.exists():
        st.sidebar.warning(f"Imagem não encontrada em: {image_path}")
        return
    encoded = base64.b64encode(path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        /* Painel semi-transparente atrás dos filtros, pra manter o texto legível */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 1rem;
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_sidebar_background(SIDEBAR_IMAGE_PATH)


filtered_df = df.copy()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

MAX_FILTERS = 5  # limite total de filtros exibidos na sidebar

# Colunas categóricas elegíveis (mais de 1 e no máximo 50 valores únicos)
eligible_categorical = [
    col for col in categorical_cols
    if 1 < df[col].dropna().nunique() <= 50
]
# Colunas numéricas elegíveis (com variação de valores)
eligible_numeric = [
    col for col in numeric_cols
    if df[col].min() < df[col].max()
]

# Intercala categóricas e numéricas até atingir o limite de MAX_FILTERS
filter_cols = []
i = j = 0
while len(filter_cols) < MAX_FILTERS and (i < len(eligible_categorical) or j < len(eligible_numeric)):
    if i < len(eligible_categorical):
        filter_cols.append(("cat", eligible_categorical[i]))
        i += 1
    if len(filter_cols) < MAX_FILTERS and j < len(eligible_numeric):
        filter_cols.append(("num", eligible_numeric[j]))
        j += 1

for kind, col in filter_cols:
    if kind == "cat":
        unique_vals = df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"{col}", sorted(map(str, unique_vals)))
        if selected:
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]
    else:
        col_min, col_max = float(df[col].min()), float(df[col].max())
        val_range = st.sidebar.slider(
            f"{col}", min_value=col_min, max_value=col_max, value=(col_min, col_max)
        )
        filtered_df = filtered_df[
            (filtered_df[col] >= val_range[0]) & (filtered_df[col] <= val_range[1])
        ]

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(filtered_df)} de {len(df)} linhas após os filtros")

# ---------------------------------------------------------------------------
# KPIs rápidos
# ---------------------------------------------------------------------------
kpi_cols = st.columns(4)
kpi_cols[0].metric("Linhas (filtradas)", len(filtered_df))
kpi_cols[1].metric("Colunas", df.shape[1])
kpi_cols[2].metric("Colunas numéricas", len(numeric_cols))
kpi_cols[3].metric("Colunas categóricas", len(categorical_cols))

# ---------------------------------------------------------------------------
# Tabela de dados
# ---------------------------------------------------------------------------
st.subheader("Dados filtrados")
st.dataframe(filtered_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Gráficos (apenas 2, simples)
# ---------------------------------------------------------------------------
st.subheader("Visualizações")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Contagem por categoria**")
    if categorical_cols:
        cat_choice = st.selectbox("Escolha a coluna", categorical_cols, key="cat_chart")
        counts = filtered_df[cat_choice].value_counts()
        st.bar_chart(counts)
    else:
        st.info("Nenhuma coluna categórica encontrada.")

with col_b:
    st.markdown("**Distribuição numérica**")
    if numeric_cols:
        num_choice = st.selectbox("Escolha a coluna", numeric_cols, key="num_chart")
        st.bar_chart(filtered_df[num_choice].value_counts().sort_index())
    else:
        st.info("Nenhuma coluna numérica encontrada.")