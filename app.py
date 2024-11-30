import streamlit as st
import pandas as pd
import joblib
import altair as alt

st.set_page_config(page_title="Prediksi Spesies Bunga Iris", layout="wide")

st.title("🌸 Prediksi Spesies Bunga Iris")
st.markdown(
    """
    Selamat datang di aplikasi prediksi spesies bunga Iris!  
    Gunakan slider di samping untuk memasukkan fitur bunga dan lihat hasil prediksi spesies.
    """
)

with st.sidebar:
    st.header("Masukkan Data")
    sepal_length = st.slider("Panjang Sepal (cm)", 4.0, 8.0, step=0.1)
    sepal_width = st.slider("Lebar Sepal (cm)", 2.0, 5.0, step=0.1)
    petal_length = st.slider("Panjang Petal (cm)", 1.0, 7.0, step=0.1)
    petal_width = st.slider("Lebar Petal (cm)", 0.1, 3.0, step=0.1)

col1, col2 = st.columns(2)

with col1:
    st.header("📊 Data Prediksi")
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    )
    st.dataframe(input_data)

with col2:
    st.header("🌼 Hasil Prediksi")
    if st.button("Prediksi"):
        model = joblib.load("iris_knn_model.pkl")
        prediction = model.predict(input_data)
        predicted_species = prediction[0]
        
        st.success(f"Spesies Bunga Iris yang Diprediksi: **{predicted_species}**")
        
        if predicted_species == "Iris-setosa":
            st.image("iris_setosa.jpg", use_column_width=True, caption="Iris Setosa")
        elif predicted_species == "Iris-versicolor":
            st.image("iris_versicolor.jpg", use_column_width=True, caption="Iris Versicolor")
        elif predicted_species == "Iris-virginica":
            st.image("iris_virginica.jpg", use_column_width=True, caption="Iris Virginica")


st.markdown("---")
st.markdown(
    "📌 **Project by Anugrah Gilang Ramadhan** | Source Code: [GitHub](https://github.com/your-repo)"
)
