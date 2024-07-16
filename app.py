import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import re

# Gerekli NLTK verilerini indirin
nltk.download('stopwords')
nltk.download('punkt')

# Kaydedilmiş modeli ve CountVectorizer'ı yükleyin
with open('sentiment_model.pkl', 'rb') as model_file:
    lr = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

# Metin ön işleme fonksiyonu
def harfdegistir(cumle):
    cumle = re.sub("[^a-zA-ZşŞçÇöÖüÜıİğĞ]", " ", cumle)
    cumle = cumle.lower()
    cumle = nltk.word_tokenize(cumle)
    cumle = [word for word in cumle if not word in set(stopwords.words("turkish"))]
    cumle = " ".join(cumle)
    return cumle

# Görüş sınıflandırma fonksiyonu
def giris_cumlesi_siniflandir(cumle):
    temiz_cumle = harfdegistir(cumle)
    vector = cv.transform([temiz_cumle]).toarray()
    tahmin = lr.predict(vector)
    return tahmin[0]

# Streamlit arayüzü
st.title('Metin Duygu Analizi')
st.write('Metin kutusuna bir cümle yazın ve sonucunu görün.')

# Kullanıcıdan giriş al
user_input = st.text_area('Cümle girin:')

if st.button('Analiz Et'):
    if user_input:
        result = giris_cumlesi_siniflandir(user_input)
        if result == 0:
            st.write('Görüş Olumlu')
        elif result == 1:
            st.write('Görüş Olumsuz')
        else:
            st.write('Tarafsız Görüş')
    else:
        st.write('Lütfen bir cümle girin.')
