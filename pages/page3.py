# testing.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pickle
from navigation import make_sidebar

make_sidebar()
# Fungsi untuk memuat model dari file menggunakan pickle
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Fungsi untuk melakukan prediksi dengan model yang telah dilatih
def predict(model, X):
    return model.predict(X)

# Fungsi untuk memuat dan memproses dataset
def load_data(uploaded_file):
    df_new = pd.read_csv(uploaded_file)
    X_new = df_new.drop(['Diagnosa', 'Nama'], axis=1)  # Menghapus kolom 'status' dan 'Nama'
    return df_new, X_new

# Membuat antarmuka pengguna dengan Streamlit
def main():
    st.title('Klasifikasi Kecemasan dengan KNN')

    uploaded_file = st.file_uploader("Upload dataset baru untuk klasifikasi:", type=['csv'])

    if uploaded_file is not None:
        df_new, X_new = load_data(uploaded_file)  # Load dataset yang diunggah oleh pengguna
        st.write("Data yang akan diklasifikasikan:")
        st.write(df_new)

        # Memuat model yang sudah dilatih
        model = load_model("model_klasifikasi.pkl")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_new_imputed = imputer.fit_transform(X_new)
        
        if st.button('Lakukan Klasifikasi'):
            predictions = predict(model, X_new_imputed)
            df_new['Klasifikasi Status'] = predictions
            st.write("Hasil Klasifikasi:")
            st.write(df_new)

            # Tampilkan hasil prediksi ke dalam diagram batang
            st.header("Visualisasi Hasil Klasifikasi")
            
            # Diagram Batangs
            st.subheader("Diagram Batang")
            pred_counts = df_new['Klasifikasi Status'].value_counts()
            plt.bar(pred_counts.index, pred_counts.values)
            plt.xlabel('Klasifikasi Status')
            plt.ylabel('Jumlah')
            st.pyplot()

            # Diagram Garis
            st.subheader("Diagram Garis")
            plt.plot(pred_counts.index, pred_counts.values, marker='o')
            plt.xlabel('Prediksi Status')
            plt.ylabel('Jumlah')
            st.pyplot()

            # Tampilkan hasil prediksi ke dalam diagram lingkaran
            st.subheader("Diagram Lingkaran")
            plt.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
            st.pyplot()
    else:
        st.write("Silakan unggah dataset baru dalam format CSV untuk dilakukan klasifikasi.")

if __name__ == "__main__":
    main()
