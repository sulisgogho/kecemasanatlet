import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
from navigation import make_sidebar
import streamlit as st

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['Diagnosa', 'Nama'], axis=1)
    y = df['Diagnosa']
    return X, y

def train_and_save_model(X, y, filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_imputed, y_train)
    
    X_test_imputed = imputer.transform(X_test)
    predictions = model.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Akurasi Model: {accuracy*100:.2f}%")
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    # Generate and display confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    st.pyplot(fig)

    # Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_imputed)
    
    # Encode labels as integers for visualization
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    predictions_encoded = le.transform(predictions)
    
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions_encoded, cmap='viridis', alpha=0.7, vmin=-0.5, vmax=len(le.classes_)-0.5)
    plt.colorbar(scatter, ticks=range(len(le.classes_)), ax=ax)
    plt.title('PCA of Test Data with KNN Predictions')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    st.pyplot(fig)

def main():
    make_sidebar()
    st.title("Model Klasifikasi Kecemasan Atlet")
    
    uploaded_file = st.file_uploader("Upload file CSV", type="csv")
    
    if uploaded_file is not None:
        X, y = load_data(uploaded_file)
        train_and_save_model(X, y, "model_klasifikasi.pkl")

if __name__ == "__main__":
    main()
