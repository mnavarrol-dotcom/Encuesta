import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import nltk
import re
import unicodedata
from collections import Counter

nltk.download('stopwords')
from nltk.corpus import stopwords

# Stopwords base y personalizadas
spanish_stopwords = stopwords.words('spanish')
stopwords_personalizadas = {'creo', 'hacer', 'siento', 'verdad', 'tan', 'tal', 'pdv', 'asi', 'sido','haria'}
stopwords_totales = set(spanish_stopwords) | stopwords_personalizadas

# Expresiones que se unen para no separarlas
EXPRESIONES_UNIDAS = {
    "ensayo presencial": "ensayo_presencial",
    "clases virtuales": "clases_virtuales",
    "trabajo final": "trabajo_final",
    "examen final": "examen_final",
    "clases presenciales": "clases_presenciales",
    "horario flexible": "horario_flexible",
    "material didáctico": "material_didactico",
    "tutor virtual": "tutor_virtual",
    "apoyo docente": "apoyo_docente",
    "evaluación continua": "evaluacion_continua"
}

# Lexicon base de VADER en español
base_lexicon = {
    'bueno': 2.0, 'excelente': 3.0, 'malo': -2.0, 'terrible': -3.0,
    'agradable': 1.5, 'horrible': -3.0, 'fácil': 1.0, 'difícil': -1.5,
    'útil': 2.0, 'pésimo': -3.0, 'mejor': 2.0, 'peor': -2.0,
    'rápido': 1.5, 'lento': -1.5,
}

# Lexicon extendido para la pregunta ¿Qué cambios implementarías?
lexicon_cambios = {
    'mejorar': 2.0, 'incrementar': 1.5, 'reducir': -1.0, 'cambiar': 1.0, 'adaptar': 1.5,
    'problema': -2.0, 'problemas': -2.0, 'error': -2.5, 'falla': -2.5,
    'sugerencia': 1.0, 'sugerencias': 1.0, 'mas_apoyo': 2.0, 'mas_claridad': 1.5,
    'flexible': 1.5, 'dificultad': -2.0, 'complicado': -2.0,
}

def quitar_tildes(texto):
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto

def agrupar_mas_con_palabra(texto):
    # Agrupa "mas palabra" -> "mas_palabra" (palabra de al menos 3 letras)
    return re.sub(r'\bmas (\w{3,})', r'mas_\1', texto)

def preprocesar_textos(textos):
    textos_procesados = []
    for t in textos:
        if not isinstance(t, str) or len(t.strip()) < 5:
            continue
        t = t.lower()
        t = quitar_tildes(t)
        t = re.sub(r'\s+', ' ', t)  # eliminar espacios dobles

        # Reemplazar expresiones unidas
        for expresion, reemplazo in EXPRESIONES_UNIDAS.items():
            t = t.replace(expresion, reemplazo)

        # Agrupar "mas + palabra"
        t = agrupar_mas_con_palabra(t)

        # Eliminar caracteres no alfabéticos excepto guion bajo y ñ
        t = re.sub(r'[^a-zA-Z_ñ\s]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()  # limpiar espacios dobles de nuevo

        palabras = t.split()
        palabras_filtradas = [p for p in palabras if p not in stopwords_totales and len(p) > 2]
        if palabras_filtradas:
            textos_procesados.append(" ".join(palabras_filtradas))
    return textos_procesados

def sentimiento_vader(texto, es_pregunta_cambios=False):
    if not isinstance(texto, str) or not texto.strip():
        return None
    analyzer = SentimentIntensityAnalyzer()
    # Actualizar lexicon según pregunta
    if es_pregunta_cambios:
        analyzer.lexicon.update({**base_lexicon, **lexicon_cambios})
    else:
        analyzer.lexicon.update(base_lexicon)
    scores = analyzer.polarity_scores(texto)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positivo'
    elif compound <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'

def codificar_temas(textos, n_topics=3, n_palabras=5):
    vectorizer = CountVectorizer(stop_words=stopwords_totales, max_features=1000)
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
    palette = {'Negativo': '#e74c3c', 'Neutro': '#95a5a6', 'Positivo': '#2ecc71'}
    sns.countplot(x=data, palette=palette)
    plt.title(f"Distribución Sentimientos - {columna}")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def mostrar_nube(textos):
    texto_unido = " ".join(textos)
    wordcloud = WordCloud(stopwords=stopwords_totales, background_color="white",
                          width=800, height=400).generate(texto_unido)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.clf()

def graficar_frecuencias_palabras(textos):
    palabras = " ".join(textos).split()
    contador = Counter(palabras)
    mas_comunes = contador.most_common(20)
    palabras_comunes = [p[0].replace('_', ' ') for p in mas_comunes]
    frecuencias = [p[1] for p in mas_comunes]

    plt.figure(figsize=(10,6))
    sns.barplot(x=frecuencias, y=palabras_comunes, palette="viridis")
    plt.title("Frecuencia de palabras más comunes")
    plt.xlabel("Frecuencia")
    plt.ylabel("Palabras")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

# --- STREAMLIT ---

st.title("Análisis de preguntas abiertas de encuesta académica")

uploaded_file = st.file_uploader("Carga tu archivo Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    columnas = ['PREG 22', 'PREG 23', 'PREG 24']
    columnas_existentes = [col for col in columnas if col in df.columns]

    if not columnas_existentes:
        st.error(f"No se encontraron las columnas {columnas}")
    else:
        columna_seleccionada = st.selectbox("Selecciona la columna a analizar:", columnas_existentes)
        
        # Detectar si la columna es la pregunta "¿Qué cambios implementarías?" para usar lexicon extendido
        es_pregunta_cambios = False
        texto_pregunta = quitar_tildes(columna_seleccionada.lower())
        if "cambios" in texto_pregunta and "implementarias" in texto_pregunta:
            es_pregunta_cambios = True

        df[columna_seleccionada + '_sentimiento'] = df[columna_seleccionada].apply(lambda x: sentimiento_vader(x, es_pregunta_cambios))

        st.subheader(f"Distribución de Sentimientos en {columna_seleccionada}")
        graficar_distribucion_sentimientos(df[columna_seleccionada + '_sentimiento'], columna_seleccionada)

        textos_procesados = preprocesar_textos(df[columna_seleccionada])
        if len(textos_procesados) >= 10:
            st.subheader("Temas identificados")
            temas, lda_model, vectorizer = codificar_temas(textos_procesados)
            for t in temas:
                st.write("- " + t)

            filtro_sentimiento = st.selectbox("Filtrar comentarios por sentimiento para análisis:",
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

                st.subheader(f"Frecuencia de palabras - Comentarios {filtro_sentimiento}")
                graficar_frecuencias_palabras(textos_filtrados)
            else:
                st.info("No hay suficientes comentarios para mostrar visualizaciones.")
        else:
            st.info("No hay suficientes textos para análisis de temas (mínimo 10).")
else:
    st.info("Carga un archivo para comenzar el análisis.")
