import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Kobe Shot Prediction", page_icon="🏀")

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

# Quando o botão for clicado
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

# Start Server: streamlit run streamlit/app.py