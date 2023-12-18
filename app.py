import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
import seaborn as sns


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Create Stemmer and StopWord Remover
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Download NLTK resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
dataset = pd.read_csv('assets/dataset.csv')

# Extract features and labels
X = dataset['komentar']
y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Text Cleaning Function
def clean_text(text):
    # Lowercasing
    text = text.lower()

    # Tokenisasi dengan Sastrawi
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [stopword_remover.remove(token) for token in tokens]

    # Gabungkan kembali tokens menjadi teks
    cleaned_text = ' '.join(tokens)

    return cleaned_text

st.title("Analisis Sentimen Ulasan Kedai Kopi")

menu = st.sidebar.selectbox("Menu", ["Home",  "Praproses", "Input", "Metrik Evaluasi"])

# Home Page
if menu == "Home":
    st.write("Selamat datang di halaman utama! Silakan pilih menu di sidebar.")

# Input Page
elif menu == "Input":
    st.subheader("Input Ulasan")
    # Input teks dari pengguna
    user_input = st.text_area("Masukkan ulasan Anda di sini:")
    btn_submit = st.button("Submit Ulasan")

    # Jika pengguna memasukkan teks dan menekan tombol submit
    if user_input and btn_submit:
        # Membersihkan teks menggunakan Text Cleaning Function
        cleaned_input = clean_text(user_input)

        # Transformasi teks menggunakan TF-IDF
        user_input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        # Prediksi sentimen
        prediction = rf_classifier.predict(user_input_tfidf)[0]

        # Tampilkan hasil
        st.subheader("Hasil Analisis Sentimen:")
        st.write("Ulasan setelah Text Cleaning:", cleaned_input)
        st.write("Sentimen:", prediction)

# Praproses Page
elif menu == "Praproses":
    st.subheader("Praproses")

    # Menampilkan dataset sebelum text cleaning
    st.write("Dataset Sebelum Text Cleaning:")
    st.dataframe(dataset)

    # Button untuk Text Cleaning pada semua kolom 'komentar'
    btn_text_cleaning = st.button("Praproses Data (Text Cleaning)")

    # Jika tombol text cleaning ditekan
    if btn_text_cleaning:
        # Melakukan text cleaning pada semua kolom 'komentar'
        dataset['komentar'] = dataset['komentar'].apply(clean_text)
        st.write("Dataset Setelah Text Cleaning:")
        st.dataframe(dataset)

    # Praproses lainnya bisa ditambahkan di sini

# Metrik Evaluasi Page
elif menu == "Metrik Evaluasi":
    st.subheader("Metrik Evaluasi Model")

    # Evaluasi model
    y_pred_test = rf_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    st.subheader("Classification Report:")
    st.text(classification_report(y_test, y_pred_test))

    st.subheader("Confusion Matrix:")
    st.text(confusion_matrix(y_test, y_pred_test))

    # Grafik jumlah komentar berdasarkan sentimen
    sentimen_count = dataset['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentimen_count.index, y=sentimen_count.values)
    plt.title("Jumlah Komentar Berdasarkan Sentimen")
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah")
    st.pyplot(plt)
