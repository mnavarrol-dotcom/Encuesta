import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
spanish_stopwords = stopwords.words('spanish')

analyzer = SentimentIntensityAnalyzer()
spanish_lexicon = {
    'bueno': 2.0, 'excelente': 3.0, 'malo': -2.0, 'terrible': -3.0,
    'agradable': 1.5, 'horrible': -3.0, 'fácil': 1.0, 'difícil': -1.5,
    'útil': 2.0, 'pésimo': -3.0, 'mejor': 2.0, 'peor': -2.0,
    'rápido': 1.5, 'lento': -1.5,
}
analyzer.lexicon.update(spanish_lexicon)

def sentimiento_vader(texto):
    if not isinstance(texto, str) or not texto.strip():
        return None
    scores = analyzer.polarity_scores(texto)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positivo'
    elif compound <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'

def preprocesar_textos(textos):
    return [str(t).lower().replace("más ", "más_").replace(" mas ", " más_")
            for t in textos if isinstance(t, str) and len(t.strip()) >= 5]

def codificar_temas(textos, n_topics=3, n_palabras=5):
    vectorizer = CountVectorizer(stop_words=spanish_stopwords, max_features=1000)
    X = vectorizer.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
    lda.fit(X)
    palabras = vectorizer.get_feature_names_out()
    temas = []
    for idx, topic in enumerate(lda.components_):
        top_words = [palabras[i].replace('_', ' ') for i in topic.argsort()[-n_palabras:][::-1]]
        temas.append(f"Tema {idx+1}: " + ", ".join(top_words))
    return temas, lda, vectorizer

def graficar_distribucion_sentimientos(data, columna):
    plt.figure(figsize=(6,4))
    sns.countplot(x=data, palette=['#e74c3c', '#95a5a6', '#2ecc71'])
    plt.title(f"Distribución Sentimientos - {columna}")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    st.pyplot(plt)

def mostrar_nube(textos):
    texto_unido = " ".join(textos)
    wordcloud = WordCloud(stopwords=set(spanish_stopwords), background_color="white",
                          width=800, height=400).generate(texto_unido)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

st.title("Análsis de preguntas abiertas de encuesta académica")

uploaded_file = st.file_uploader("Carga tu archivo Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    columnas = ['PREG 22', 'PREG 23', 'PREG 24']
    columnas_existentes = [col for col in columnas if col in df.columns]
    if not columnas_existentes:
        st.error(f"No se encontraron las columnas {columnas}")
    else:
        columna_seleccionada = st.selectbox("Selecciona la columna a analizar:", columnas_existentes)
        df[columna_seleccionada + '_sentimiento'] = df[columna_seleccionada].apply(sentimiento_vader)
        
        st.subheader(f"Distribución de Sentimientos en {columna_seleccionada}")
        graficar_distribucion_sentimientos(df[columna_seleccionada + '_sentimiento'], columna_seleccionada)
        
        textos_procesados = preprocesar_textos(df[columna_seleccionada])
        if len(textos_procesados) >= 10:
            st.subheader("Temas identificados")
            temas, lda_model, vectorizer = codificar_temas(textos_procesados)
            for t in temas:
                st.write("- " + t)
            
            filtro_sentimiento = st.selectbox("Filtrar comentarios por sentimiento para nube de palabras:",
                                              ['Todos', 'Positivo', 'Neutro', 'Negativo'])
            if filtro_sentimiento == 'Todos':
                textos_filtrados = textos_procesados
            else:
                textos_filtrados = preprocesar_textos(
                    df[df[columna_seleccionada + '_sentimiento'] == filtro_sentimiento][columna_seleccionada]
                )
            if textos_filtrados:
                st.subheader(f"Nube de palabras - Comentarios {filtro_sentimiento}")
                mostrar_nube(textos_filtrados)
            else:
                st.info("No hay suficientes comentarios para mostrar la nube de palabras.")
        else:
            st.info("No hay suficientes textos para análisis de temas (mínimo 10).")
else:
    st.info("Carga un archivo para comenzar el análisis.")
