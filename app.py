import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

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

    # Load Dataset dan Training Model
    df = pd.read_csv("Iris.csv")
    df.rename(columns={"Species": "species"}, inplace=True)  # Normalisasi kolom

    # Hapus kolom 'Id' yang tidak diperlukan
    df = df.drop(columns=["Id"], errors="ignore")

    X = df.drop("species", axis=1)
    y = df["species"]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    if st.button("Prediksi"):
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
    "📌 **Project by Anugrah Gilang Ramadhan** | Source Code: [GitHub](https://github.com/AnugrahGilang/Prediksi-Spesies-Tanaman-Iris)"
)
