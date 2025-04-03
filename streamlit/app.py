import streamlit as st
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Kobe Bryant - Modelo de Previsão", page_icon="🏀", layout="wide")

# Tabs para separação das páginas
aba_previsao, aba_monitoramento = st.tabs(["📊 Previsão", "📈 Monitoramento da Operação"])

# ----------------------------
# ABA 1: Previsão
# ----------------------------
with aba_previsao:
    st.title("🏀 Previsão de Acerto do Arremesso (Kobe Bryant)")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=34.0443)
            lon = st.number_input("Longitude", value=-118.2690)
            minutes_remaining = st.slider("Minutos restantes", 0, 12, 5)
        with col2:
            period = st.selectbox("Período", [1, 2, 3, 4], index=1)
            playoffs = st.selectbox("Playoffs?", [0, 1], format_func=lambda x: "Sim" if x else "Não")
            shot_distance = st.slider("Distância do arremesso (ft)", 0, 60, 15)

        submit = st.form_submit_button("🔎 Prever")

    if submit:
        input_data = pd.DataFrame([{
            "lat": lat,
            "lon": lon,
            "minutes_remaining": minutes_remaining,
            "period": period,
            "playoffs": playoffs,
            "shot_distance": shot_distance
        }])

        try:
            response = requests.post(
                url="http://localhost:5001/invocations",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"inputs": input_data.to_dict(orient="records")})
            )
            if response.status_code == 200:
                probs = response.json()
                st.success("✅ Previsão realizada com sucesso!")
                st.metric("📊 Probabilidade de ERRO", f"{probs['predictions'][0][0]*100:.2f}%")
                st.metric("🏀 Probabilidade de ACERTO", f"{probs['predictions'][0][1]*100:.2f}%")
            else:
                st.error(f"Erro na requisição: {response.status_code}")
                st.code(response.text)

        except Exception as e:
            st.error("Erro ao conectar com o servidor MLflow.")
            st.exception(e)

# ----------------------------
# ABA 2: Monitoramento
# ----------------------------
with aba_monitoramento:
    st.title("📈 Monitoramento da Operação do Modelo")

    dados_path = Path("data/07_model_output/resultados_aplicacao.parquet")
    if not dados_path.exists():
        st.warning("⚠️ Arquivo de monitoramento não encontrado.")
    else:
        df = pd.read_parquet(dados_path)

        st.header("Resumo Estatístico")
        col1, col2, col3 = st.columns(3)
        col1.metric("🔢 Total de registros", len(df))
        col2.metric("🎯 Taxa de acerto prevista", f"{(df['y_pred'] == df['y_true']).mean():.2%}")
        col3.metric("🏀 Média probabilidade de acerto", f"{df['prob_1'].mean():.2%}")

        st.subheader("Distribuição das Probabilidades de Acerto")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df["prob_1"], bins=20, kde=True, ax=ax)
        ax.set_xlabel("Probabilidade de acerto")
        ax.set_ylabel("Frequência")
        st.pyplot(fig)

        st.subheader("Análise de Erros")
        df['erro'] = df['y_true'] != df['y_pred']
        erros = df[df['erro']]

        col1, col2 = st.columns([1, 2])
        col1.metric("❌ Erros totais", len(erros))
        col1.metric("📉 Taxa de erro", f"{len(erros)/len(df):.2%}")

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.boxplot(data=df, x="erro", y="prob_1", ax=ax2)
        ax2.set_xticklabels(["Acerto", "Erro"])
        ax2.set_ylabel("Probabilidade de Acerto")
        st.pyplot(fig2)

        st.subheader("📂 Dados brutos")
        st.dataframe(df.head(50))
        st.download_button("⬇️ Baixar resultados", data=df.to_csv(index=False), file_name="resultados_monitoramento.csv")
