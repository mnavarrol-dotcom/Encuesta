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
stopwords_personalizadas = {'creo', 'hacer', 'siento', 'verdad', 'tan', 'tal', 'pdv', 'asi', 'sido','haria', 'hace'}
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
    "evaluación continua": "evaluacion_continua",
    "ensayos presenciales" : "ensayos_presenciales",
    "ensayos virtuales" : "ensayos_virtuales",
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

# -------------------------------
# FUNCIONES DE PREPROCESAMIENTO
# -------------------------------
def quitar_tildes(texto):
    texto = unicodedata.normalize('NFD', texto)
    return ''.join([c for c in texto if unicodedata.category(c) != 'Mn'])

def agrupar_mas_con_palabra(texto):
    return re.sub(r'\bmas (\w{3,})', r'mas_\1', texto)

def preprocesar_textos(textos):
    textos_procesados = []
    for t in textos:
        if not isinstance(t, str) or len(t.strip()) < 5:
            continue
        t = t.lower()
        t = quitar_tildes(t)
        t = re.sub(r'\s+', ' ', t)
        for expresion, reemplazo in EXPRESIONES_UNIDAS.items():
            t = t.replace(expresion, reemplazo)
        t = agrupar_mas_con_palabra(t)
        t = re.sub(r'[^a-zA-Z_ñ\s]', '', t)
        palabras = [p for p in t.split() if p not in stopwords_totales and len(p) > 2]
        if palabras:
            textos_procesados.append(" ".join(palabras))
    return textos_procesados

# -------------------------------
# SENTIMIENTO Y TEMAS
# -------------------------------
def sentimiento_vader(texto, es_pregunta_cambios=False):
    if not isinstance(texto, str) or not texto.strip():
        return None
    analyzer = SentimentIntensityAnalyzer()
    if es_pregunta_cambios:
        analyzer.lexicon.update({**base_lexicon, **lexicon_cambios})
    else:
        analyzer.lexicon.update(base_lexicon)
    compound = analyzer.polarity_scores(texto)['compound']
    return 'Positivo' if compound >= 0.05 else 'Negativo' if compound <= -0.05 else 'Neutro'

def codificar_temas(textos, n_topics=3, n_palabras=5):
    vectorizer = CountVectorizer(stop_words=list(stopwords_totales), max_features=1000)
    X = vectorizer.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    palabras = vectorizer.get_feature_names_out()
    temas = []
    for idx, topic in enumerate(lda.components_):
        top_words = [palabras[i].replace('_', ' ') for i in topic.argsort()[-n_palabras:][::-1]]
        temas.append(f"Tema {idx+1}: " + ", ".join(top_words))
    return temas

# -------------------------------
# GRAFICACIÓN
# -------------------------------
def graficar_distribucion_sentimientos(data, columna):
    plt.figure(figsize=(6,4))
    palette = {'Negativo': '#e74c3c', 'Neutro': '#95a5a6', 'Positivo': '#2ecc71'}
    sns.countplot(x=data, palette=palette)
    plt.title(f"Distribución Sentimientos - {columna}")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def mostrar_nube(textos):
    texto_unido = " ".join(textos)
    if not texto_unido.strip():
        st.info("No hay texto suficiente para generar la nube.")
        return
    wordcloud = WordCloud(stopwords=stopwords_totales, background_color="white", width=800, height=400).generate(texto_unido)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.clf()

def graficar_frecuencias_palabras(textos):
    palabras = " ".join(textos).split()
    mas_comunes = Counter(palabras).most_common(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=[p[1] for p in mas_comunes], y=[p[0].replace('_',' ') for p in mas_comunes], palette="viridis")
    plt.title("Frecuencia de palabras más comunes")
    st.pyplot(plt)
    plt.clf()

def calcular_coocurrencia(textos, top_n=30):
    palabras = " ".join(textos).split()
    top = [p for p, _ in Counter(palabras).most_common(top_n)]
    matrix = pd.DataFrame(0, index=top, columns=top)
    for t in textos:
        tokens = set(t.split())
        for w1 in tokens:
            for w2 in tokens:
                if w1 in top and w2 in top:
                    matrix.loc[w1, w2] += 1
    return matrix

def graficar_mapa_calor_coocurrencia(coocurrencia_df, titulo="Mapa de Calor de Coocurrencia"):
    if coocurrencia_df.empty:
        st.info("No hay datos suficientes para generar el mapa de calor.")
        return
    plt.figure(figsize=(10,8))
    sns.heatmap(coocurrencia_df, cmap="coolwarm", linewidths=0.5)
    plt.title(titulo)
    st.pyplot(plt)
    plt.clf()

# -------------------------------
# INTERFAZ STREAMLIT
# -------------------------------
st.title("📊 Análisis de Preguntas Abiertas Académicas")

uploaded_file = st.file_uploader("📂 Carga tu archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    nombre = uploaded_file.name.lower()

    # Detección automática del tipo de base
    if "acp" in nombre:
        tipo_base = "ACP"
    elif "ge" in nombre:
        tipo_base = "GE"
    elif "dynamic" in nombre:
        tipo_base = "Dynamic"
    elif any(c in nombre for c in ["3", "cuarto", "tercero"]):
        tipo_base = "3° y 4°"
    elif any(c in nombre for c in ["1", "2", "primero", "segundo"]):
        tipo_base = "1° y 2°"
    else:
        # Detectar por columnas presentes
        cols = list(df.columns)
        if 'PREG 24' in cols and 'PREG 25' in cols:
            tipo_base = "3° y 4°"
        elif 'PREG 21' in cols and 'PREG 22' in cols:
            tipo_base = "1° y 2°"
        elif 'PREG 22' in cols and 'PREG 23' in cols:
            tipo_base = "ACP"
        elif 'PREG 25' in cols and 'PREG 26' in cols:
            tipo_base = "GE"
        else:
            tipo_base = "Desconocida"

    st.success(f"✅ Tipo de base detectado automáticamente: **{tipo_base}**")

    columnas_por_base = {
        "3° y 4°": ['PREG 24', 'PREG 25', 'PREG 26'],
        "1° y 2°": ['PREG 21', 'PREG 22', 'PREG 23'],
        "ACP": ['PREG 22', 'PREG 23', 'PREG 24'],
        "GE": ['PREG 25', 'PREG 26', 'PREG 27'],
        "Dynamic": ['PREG 25', 'PREG 26', 'PREG 27']
    }

    columnas_disponibles = [c for c in columnas_por_base.get(tipo_base, []) if c in df.columns]
    if not columnas_disponibles:
        st.warning("No se detectaron columnas válidas de preguntas abiertas en esta base.")
    else:
        columna_sel = st.selectbox("Selecciona la pregunta a analizar:", ["Todas"] + columnas_disponibles)

        # Filtros SEDE y NIVEL
        if "SEDE" in df.columns and "NIVEL" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                sede_sel = st.multiselect("Filtrar por SEDE:", sorted(df["SEDE"].dropna().unique()))
            with col2:
                nivel_sel = st.multiselect("Filtrar por NIVEL:", sorted(df["NIVEL"].dropna().unique()))
            if sede_sel:
                df = df[df["SEDE"].isin(sede_sel)]
            if nivel_sel:
                df = df[df["NIVEL"].isin(nivel_sel)]

        columnas_a_analizar = columnas_disponibles if columna_sel == "Todas" else [columna_sel]

        for col in columnas_a_analizar:
            st.markdown(f"### 🔍 Análisis de {col}")
            es_pregunta_cambios = "cambio" in col.lower()
            df[col + "_sentimiento"] = df[col].apply(lambda x: sentimiento_vader(x, es_pregunta_cambios))
            graficar_distribucion_sentimientos(df[col + "_sentimiento"], col)

            textos = preprocesar_textos(df[col])
            if len(textos) >= 10:
                temas = codificar_temas(textos)
                st.write("**Temas principales:**")
                for t in temas:
                    st.write("-", t)
                st.subheader("☁️ Nube de palabras")
                mostrar_nube(textos)
                st.subheader("📈 Frecuencia de palabras")
                graficar_frecuencias_palabras(textos)
                st.subheader("🔥 Mapa de calor de coocurrencia")
                cooc = calcular_coocurrencia(textos)
                graficar_mapa_calor_coocurrencia(cooc)
            else:
                st.info(f"No hay suficientes textos en {col} para análisis (mínimo 10).")
else:
    st.info("Por favor, carga un archivo Excel para comenzar el análisis.")
