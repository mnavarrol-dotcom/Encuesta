import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import itertools
import re
import unicodedata
from collections import Counter, defaultdict
import nltk

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
# FUNCIONES DE PROCESAMIENTO
# -------------------------------
def quitar_tildes(texto):
    texto = unicodedata.normalize('NFD', texto)
    return ''.join([c for c in texto if unicodedata.category(c) != 'Mn'])

def agrupar_mas_con_palabra(texto):
    return re.sub(r'\bmas (\w{3,})', r'mas_\1', texto)

@st.cache_data
def preprocesar_textos(textos):
    procesados = []
    for t in textos:
        if not isinstance(t, str) or len(t.strip()) < 5:
            continue
        t = t.lower()
        t = quitar_tildes(t)
        t = re.sub(r'\s+', ' ', t)
        for expr, repl in EXPRESIONES_UNIDAS.items():
            t = t.replace(expr, repl)
        t = agrupar_mas_con_palabra(t)
        t = re.sub(r'[^a-zA-Z_ñ\s]', '', t)
        palabras = [p for p in t.split() if p not in stopwords_totales and len(p) > 2]
        if palabras:
            procesados.append(" ".join(palabras))
    return procesados

# -------------------------------
# FUNCIONES DE ANÁLISIS
# -------------------------------
@st.cache_resource
def sentimiento_vader_lista(textos, es_pregunta_cambios=False):
    analyzer = SentimentIntensityAnalyzer()
    if es_pregunta_cambios:
        analyzer.lexicon.update({**base_lexicon, **lexicon_cambios})
    else:
        analyzer.lexicon.update(base_lexicon)
    resultados = []
    for t in textos:
        if not isinstance(t, str) or not t.strip():
            resultados.append(None)
        else:
            comp = analyzer.polarity_scores(t)['compound']
            resultados.append('Positivo' if comp >= 0.05 else 'Negativo' if comp <= -0.05 else 'Neutro')
    return resultados

@st.cache_resource
def codificar_temas(textos, n_topics=3, n_palabras=5):
    if len(textos) < 5:
        return ["Datos insuficientes"]
    vectorizer = CountVectorizer(stop_words=list(stopwords_totales), max_features=700, min_df=3, max_df=0.8)
    X = vectorizer.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    palabras = vectorizer.get_feature_names_out()
    temas = []
    for idx, topic in enumerate(lda.components_):
        top_words = [palabras[i].replace('_', ' ') for i in topic.argsort()[-n_palabras:][::-1]]
        temas.append(f"Tema {idx+1}: " + ", ".join(top_words))
    return temas

@st.cache_data
def calcular_coocurrencia_opt(textos, top_n=30):
    palabras = " ".join(textos).split()
    top = [p for p, _ in Counter(palabras).most_common(top_n)]
    top_set = set(top)
    cooc = defaultdict(Counter)
    for t in textos:
        tokens = [p for p in set(t.split()) if p in top_set]
        for w1, w2 in itertools.combinations(tokens, 2):
            cooc[w1][w2] += 1
            cooc[w2][w1] += 1
    df = pd.DataFrame(cooc).fillna(0).reindex(index=top, columns=top, fill_value=0)
    return df

# -------------------------------
# FUNCIONES DE GRAFICACIÓN
# -------------------------------
def graficar_distribucion_sentimientos(sentimientos, columna):
    plt.figure(figsize=(6,4))
    sns.countplot(x=sentimientos, palette={'Negativo':'#e74c3c','Neutro':'#95a5a6','Positivo':'#2ecc71'})
    plt.title(f"Distribución Sentimientos - {columna}")
    st.pyplot(plt)
    plt.clf()

def mostrar_nube(textos):
    texto_unido = " ".join(textos)
    if not texto_unido.strip():
        st.info("No hay texto suficiente para la nube.")
        return
    wc = WordCloud(stopwords=stopwords_totales, background_color="white", width=800, height=400).generate(texto_unido)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.clf()

def graficar_frecuencias(textos):
    palabras = " ".join(textos).split()
    comunes = Counter(palabras).most_common(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=[p[1] for p in comunes], y=[p[0].replace('_',' ') for p in comunes], palette="viridis")
    plt.title("Frecuencia de palabras más comunes")
    st.pyplot(plt)
    plt.clf()

def graficar_mapa_calor(df, titulo):
    if df.empty:
        st.info("No hay datos suficientes para el mapa de calor.")
        return
    plt.figure(figsize=(10,8))
    sns.heatmap(df, cmap="coolwarm", linewidths=0.5)
    plt.title(titulo)
    st.pyplot(plt)
    plt.clf()

# -------------------------------
# APLICACIÓN STREAMLIT
# -------------------------------
st.title("📊 Análisis optimizado de respuestas abiertas académicas")

uploaded_file = st.file_uploader("Carga tu archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    @st.cache_data
    def cargar_excel(f):
        return pd.read_excel(f)
    
    df = cargar_excel(uploaded_file)
    nombre = uploaded_file.name.lower()

    # --- Detectar tipo de base automáticamente ---
    if "acp" in nombre:
        tipo_detectado = "ACP"
    elif "ge" in nombre:
        tipo_detectado = "GE"
    elif "dynamic" in nombre:
        tipo_detectado = "Dynamic"
    elif "3" in nombre or "4" in nombre:
        tipo_detectado = "3° y 4°"
    elif "1" in nombre or "2" in nombre:
        tipo_detectado = "1° y 2°"
    else:
        tipo_detectado = "3° y 4°"

    st.subheader(f"📂 Tipo de base detectada automáticamente: **{tipo_detectado}**")

    columnas_por_base = {
        "3° y 4°": ['PREG 24', 'PREG 25', 'PREG 26'],
        "1° y 2°": ['PREG 21', 'PREG 22', 'PREG 23'],
        "ACP": ['PREG 22', 'PREG 23', 'PREG 24'],
        "GE": ['PREG 25', 'PREG 26', 'PREG 27'],
        "Dynamic": ['PREG 25', 'PREG 26', 'PREG 27']
    }
    columnas_disponibles = [c for c in columnas_por_base[tipo_detectado] if c in df.columns]

    if not columnas_disponibles:
        st.error("No se encontraron columnas PREG esperadas en la base cargada.")
        st.stop()

    # --- Filtros SEDE, NIVEL y PREG ---
    col1, col2, col3 = st.columns(3)
    with col1:
        sede_sel = st.multiselect("Filtrar por SEDE:", sorted(df["SEDE"].dropna().unique()) if "SEDE" in df else [])
    with col2:
        nivel_sel = st.multiselect("Filtrar por NIVEL:", sorted(df["NIVEL"].dropna().unique()) if "NIVEL" in df else [])
    with col3:
        preg_sel = st.selectbox("Selecciona la pregunta a analizar:", columnas_disponibles)

    if sede_sel:
        df = df[df["SEDE"].isin(sede_sel)]
    if nivel_sel:
        df = df[df["NIVEL"].isin(nivel_sel)]


    st.subheader(f"📊 Análisis de {preg_sel}")

    textos = preprocesar_textos(df[preg_sel].dropna().astype(str))
    if len(textos) == 0:
        st.warning("No hay texto suficiente para analizar.")
        st.stop()

    with st.spinner("Analizando sentimientos..."):
        sentimientos = sentimiento_vader_lista(textos)

    graficar_distribucion_sentimientos(sentimientos, preg_sel)

    if st.checkbox("Mostrar nube de palabras"):
        mostrar_nube(textos)

    if st.checkbox("Mostrar frecuencias de palabras"):
        graficar_frecuencias(textos)

    if st.checkbox("Mostrar temas (LDA)"):
        temas = codificar_temas(textos)
        st.write("**Temas detectados:**")
        for t in temas:
            st.markdown(f"- {t}")

    if st.checkbox("Mostrar mapa de calor de coocurrencia"):
        cooc = calcular_coocurrencia_opt(textos)
        graficar_mapa_calor(cooc, f"Coocurrencia - {preg_sel}")
