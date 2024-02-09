import streamlit as st
from streamlit_option_menu import option_menu
import re
import nltk
from nltk.tokenize import word_tokenize
import ssl
import os
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context
from streamlit_gsheets import GSheetsConnection
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#melakukan text cleaning ya berliana
import re

with st.sidebar:
    selected = option_menu(
        "Main Menu", 
        ["Home", 'Praproses', "Klasifikasi", "Visualisasi"], 
        icons=['house', 'gear', 'box-seam', 'archive'], 
        menu_icon="cast",
        default_index=0
    )
conn = st.connection("gsheets", type=GSheetsConnection)
data = conn.read(worksheet="development", usecols=[0,1])
data = data.dropna(how="all")

def clean_text(text):
    # menghilangkan karakter
    cleaned_text = text.str.replace('[^a-zA-Z0-9\s]', '')
    # merubah huruf menjadi kecil
    cleaned_text = cleaned_text.str.lower()
    return cleaned_text

def tokenize_text(text):
    tokens = text.apply(word_tokenize)
    return tokens

def remove_stopwords(words):
    nltk.download('stopwords')
    list_stopwords = stopwords.words('indonesian')
    list_stopwords = set(list_stopwords)
    return [word for word in words if word not in list_stopwords]

def stemming(words):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in words]

def analyze_sentiment(text):
    # Load model dan vektorisasi yang telah dilatih sebelumnya
    model = RandomForestClassifier()  # Ganti dengan model yang sebenarnya
    vect = TfidfVectorizer()  # Ganti dengan vektorisasi yang sebenarnya

    # Prediksi sentimen
    y_pred = model.predict(vect.transform([text]))
    y_pred_proba = model.predict_proba(vect.transform([text]))

    # Tampilkan hasil output
    if y_pred == 1:
        result = f"Kalimat di atas memiliki sentimen POSITIF dengan probabilitas {np.round(np.max(y_pred_proba, axis=1), 2)[0]*100}%"
        st.success(result)
    elif y_pred == 2:
        result = f"Kalimat di atas memiliki sentimen NEGATIF dengan probabilitas {np.round(np.max(y_pred_proba, axis=1), 2)[0]*100}%"
        st.error(result)
    else:
        result = f"Kalimat di atas memiliki sentimen NETRAL dengan probabilitas {np.round(np.max(y_pred_proba, axis=1), 2)[0]*100}%"
        st.warning(result)

tfidf_vectorizer = TfidfVectorizer()

if selected == "Home":
    st.title("Halaman Dashboard")
    kolom_sentimen = data["sentimen"]
    total_data = len(data)

    count_sentimen_positif = (kolom_sentimen == 1).sum()
    count_sentimen_netral = (kolom_sentimen == 0).sum()
    count_sentimen_negatif = (kolom_sentimen == 2).sum()

    st.write("Informasi Sentimen:")
    if count_sentimen_positif > 0:
        css_card_positif = """
            <style>
                .card {
                    background-color: #808080;
                    height: 150px;
                    color: white;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                }
                .card-positif {
                    background-color: #008000;
                }
                .text {
                    font-size: 60px;
                    font-weight: bold;
                }
                .card-negatif {
                    background-color: #FF0000;
                }
                .card-total {
                    background-color: #bf00ff;
                }
            </style>
        """
        st.markdown(css_card_positif, unsafe_allow_html=True)
        
        # Membagi layar menjadi tiga kolom
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        with col1:
            st.markdown(f'<div class="card card-total"><div> Total:  </div> <div class="text" >{total_data} </div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="card card-positif"><div> sentimen positif:  </div> <div class="text" >{count_sentimen_positif} </div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="card"> <div>sentimen netral: </div> <div class="text" > {count_sentimen_netral} </div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="card card-negatif"> <div> sentimen negatif: </div> <div class="text"> {count_sentimen_negatif} </div>  </div>', unsafe_allow_html=True)
    else:
        st.write("Tidak ada sentimen positif dalam data.")

if selected == "Praproses":
    st.dataframe(data)
    st.title("Halaman Praproses")
    if st.button("Praproses Data"):
        # Membersihkan teks
        cleaned_tweets = clean_text(data['Tweet'])
        # Memperbarui kolom 'Tweet' dengan teks yang telah dibersihkan
        data['Tweet'] = cleaned_tweets
        updated_data = data[['sentimen', 'Tweet']]
        # Menampilkan data frame setelah praproses
        st.write("Hasil Setelah Cleaning Text:")
        # Menyimpan data ke dalam file Excel
        updated_data.to_excel("assets/praproses.xlsx", index=False)
        st.dataframe(updated_data)

        # Melakukan tokenisasi pada data yang telah dibersihkan
        tokens = tokenize_text(updated_data['Tweet'])  
        tokens_df = pd.DataFrame({updated_data.columns[1]: tokens})  
        tokens_df_with_sentiment = pd.concat([updated_data[['sentimen']], tokens_df], axis=1)
        st.write("Hasil Setelah Stopword Tokenisasi:")
        st.dataframe(tokens_df_with_sentiment)

        # Melakukan stopword removal pada hasil tokenisasi
        tokens_without_stopwords = tokens.apply(remove_stopwords)
        tokens_without_stopwords_df = pd.DataFrame({updated_data.columns[1]: tokens_without_stopwords}) 
        tokens_without_stopwords_df_with_sentiment = pd.concat([updated_data[['sentimen']], tokens_without_stopwords_df], axis=1)
        st.write("Hasil Setelah Stopword Removal:")
        st.dataframe(tokens_without_stopwords_df_with_sentiment)

        # Melakukan stemming pada hasil penghapusan stopwords
        stemmed_tokens = tokens_without_stopwords_df[updated_data.columns[1]].apply(stemming)
        stemmed_tokens_df = pd.DataFrame({updated_data.columns[1]: stemmed_tokens}) 
        stemmed_tokens_df_with_sentiment = pd.concat([updated_data[['sentimen']], stemmed_tokens_df], axis=1)

        st.write("Hasil Setelah Stemming:")
        st.dataframe(stemmed_tokens_df_with_sentiment)

        st.write("Hasil Setelah Unlisting:")
        unlist = stemmed_tokens_df_with_sentiment.copy()  # Salin DataFrame sebelumnya
        unlist[updated_data.columns[1]] = unlist[updated_data.columns[1]].agg(lambda x: ','.join(map(str, x)))
        unlist.to_excel("assets/clean_text.xlsx", index=False)
        conn.update(worksheet="development", data=unlist)
        st.dataframe(unlist)

if selected == "Klasifikasi":
    st.title("Halaman Klasifikasi")

    # Cek apakah file clean_text.xlsx ada
    if not os.path.exists("assets/clean_text.xlsx"):
        st.error("Anda harus melakukan praproses terlebih dahulu.")
    else:
        col1, col2, col3 = st.columns(3, gap="medium")

        # Load data yang telah diproses
        clean_data = pd.read_excel("assets/clean_text.xlsx")

        # Hapus baris yang mengandung nilai NaN
        clean_data = clean_data.dropna()

        if clean_data.empty:
            st.error("Data setelah praproses tidak tersedia atau semuanya mengandung nilai NaN.")
        else:
            # Terapkan metode TF-IDF
  
            X = tfidf_vectorizer.fit_transform(clean_data['Tweet'])
            y = clean_data['sentimen']

            # Bagi data menjadi data pelatihan dan data pengujian
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Latih model klasifikasi menggunakan algoritma Random Forest
            classifier = RandomForestClassifier()
            classifier.fit(X_train, y_train)

            # Evaluasi kinerja model
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            df = pd.DataFrame(data)
            # Menghitung jumlah tweet untuk masing-masing sentimen
            count_sentimen_positif = (df['sentimen'] == 1).sum()
            count_sentimen_netral = (df['sentimen'] == 0).sum()
            count_sentimen_negatif = (df['sentimen'] == 2).sum()
            # Membuat dataframe untuk digunakan dalam grafik pie
            sentimen_data = {
                "Sentimen": ["Positif", "Netral", "Negatif"],
                "Jumlah": [count_sentimen_positif, count_sentimen_netral, count_sentimen_negatif]
            }
            sentimen_df = pd.DataFrame(sentimen_data)
            # Membuat grafik pie menggunakan Plotly Express
            fig = px.pie(sentimen_df, values="Jumlah", names="Sentimen", title="Persentase Data Sentimen")

            # Menampilkan grafik
            st.plotly_chart(fig)

            with col1:
                st.write("Akurasi:", accuracy)
                st.write("Laporan Klasifikasi:")
                st.text(report)

            with col2:
                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix:")
                st.write(cm)



X = tfidf_vectorizer.fit_transform(data['Tweet'])
classifier = RandomForestClassifier()
classifier.fit(X, data['sentimen'])

if selected == "Visualisasi":
    st.subheader("Visualisasi Prediksi Sentimen")
    text_input = st.text_area("Masukkan teks:")
    if st.button("Prediksi Sentimen"):
        if text_input:
            X_input = tfidf_vectorizer.transform([text_input])

            y_pred_input = classifier.predict(X_input)
            y_pred_proba_input = classifier.predict_proba(X_input)

            if y_pred_input == 1:
                result = f"Kalimat di atas memiliki sentimen POSITIF dengan probabilitas {str(np.round(np.max(y_pred_proba_input, axis=1), 2))[1:5]}%"
                st.success(result)
            elif y_pred_input == 2:
                result = f"Kalimat di atas memiliki sentimen NEGATIF dengan probabilitas {str(np.round(np.max(y_pred_proba_input, axis=1), 2))[1:5]}%"
                st.error(result)
            else:
                result = f"Kalimat di atas memiliki sentimen NETRAL dengan probabilitas {str(np.round(np.max(y_pred_proba_input, axis=1), 2))[1:5]}%"
                st.warning(result)
        else:
            st.warning("Masukkan teks terlebih dahulu.")

