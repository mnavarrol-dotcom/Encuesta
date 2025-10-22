import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import unicodedata
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from nltk.corpus import stopwords

# --- OPTIMIZACIÓN DE CARGA ---
# Descargar stopwords solo si no existen
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Cachear el analizador de sentimiento
@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = get_analyzer()

# Cachear datos pesados
@st.cache_data(ttl=3600)
def cargar_excel(uploaded_file):
    # Cargar solo columnas relevantes
    df = pd.read_excel(
        uploaded_file,
        engine='openpyxl',
        usecols=lambda c: c.startswith('PREG') or c in ['SEDE', 'NIVEL']
    )
    return df

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Análisis de Encuestas", layout="wide")
st.title("📊 Análisis de Preguntas Abiertas de Encuestas Académicas")

# --- PARÁMETROS BASE ---
spanish_stopwords = stopwords.words('spanish')
stopwords_personalizadas = {'creo', 'hacer', 'siento', 'verdad', 'tan', 'tal', 'pdv', 'asi', 'sido', 'haria', 'hace'}
stopwords_totales = set(spanish_stopwords) | stopwords_personalizadas

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
    "ensayos presenciales": "ensayos_presenciales",
    "ensayos virtuales": "ensayos_virtuales",
}

base_lexicon = {
    'bueno': 2.0, 'excelente': 3.0, 'malo': -2.0, 'terrible': -3.0,
    'agradable': 1.5, 'horrible': -3.0, 'fácil': 1.0, 'difícil': -1.5,
    'útil': 2.0, 'pésimo': -3.0, 'mejor': 2.0, 'peor': -2.0,
    'rápido': 1.5, 'lento': -1.5,
}

lexicon_cambios = {
    'mejorar': 2.0, 'incrementar': 1.5, 'reducir': -1.0, 'cambiar': 1.0, 'adaptar': 1.5,
    'problema': -2.0, 'problemas': -2.0, 'error': -2.5, 'falla': -2.5,
    'sugerencia': 1.0, 'sugerencias': 1.0, 'mas_apoyo': 2.0, 'mas_claridad': 1.5,
    'flexible': 1.5, 'dificultad': -2.0, 'complicado': -2.0,
}

# --- FUNCIONES AUXILIARES ---
def quitar_tildes(texto):
    texto = unicodedata.normalize('NFD', texto)
    return ''.join([c for c in texto if unicodedata.category(c) != 'Mn'])

def agrupar_mas_con_palabra(texto):
    return re.sub(r'\bmas (\w{3,})', r'mas_\1', texto)

@st.cache_data(ttl=3600)
def preprocesar_textos(textos):
    textos_procesados = []
    for t in textos:
        if not isinstance(t, str) or len(t.strip()) < 5:
            continue
        t = t.lower()
        t = quitar_tildes(t)
        for expresion, reemplazo in EXPRESIONES_UNIDAS.items():
            t = t.replace(expresion, reemplazo)
        t = agrupar_mas_con_palabra(t)
        t = re.sub(r'[^a-zA-Z_ñ\s]', '', t)
        palabras = [p for p in t.split() if p not in stopwords_totales and len(p) > 2]
        if palabras:
            textos_procesados.append(" ".join(palabras))
    return textos_procesados

def sentimiento_vader(texto, es_pregunta_cambios=False):
    if not isinstance(texto, str) or not texto.strip():
        return None
    analyzer.lexicon.update(base_lexicon)
    if es_pregunta_cambios:
        analyzer.lexicon.update(lexicon_cambios)
    scores = analyzer.polarity_scores(texto)
    c = scores['compound']
    if c >= 0.05: return 'Positivo'
    elif c <= -0.05: return 'Negativo'
    else: return 'Neutro'

@st.cache_data(ttl=3600)
def codificar_temas(textos, n_topics=3, n_palabras=5):
    vectorizer = CountVectorizer(stop_words=list(stopwords_totales), max_features=1000)
    X = vectorizer.fit_transform(textos)
    lda = LatentDirrichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
    lda.fit(X)
    palabras = vectorizer.get_feature_names_out()
    temas = []
    for idx, topic in enumerate(lda.components_):
        top_words = [palabras[i].replace('_', ' ') for i in topic.argsort()[-n_palabras:][::-1]]
        temas.append(f"Tema {idx+1}: " + ", ".join(top_words))
    return temas

def graficar_distribucion_sentimientos(data, columna):
    plt.figure(figsize=(6,4))
    sns.countplot(x=data, palette={'Negativo':'#e74c3c','Neutro':'#95a5a6','Positivo':'#2ecc71'})
    plt.title(f"Distribución Sentimientos - {columna}")
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
    contador = Counter(palabras)
    comunes = contador.most_common(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=[c[1] for c in comunes], y=[c[0].replace('_',' ') for c in comunes], palette="viridis")
    plt.title("Palabras más frecuentes")
    st.pyplot(plt)
    plt.clf()

def filtrar_por_palabras(df, columna, palabras_busqueda):
    palabras_proc = [quitar_tildes(p.strip().lower()) for p in palabras_busqueda if p.strip()]
    if not palabras_proc:
        return pd.DataFrame(), {}, {}
    mask = df[columna].fillna("").apply(lambda txt: any(p in quitar_tildes(txt.lower()) for p in palabras_proc))
    df_filtrado = df[mask]
    total, textos = {}, {}
    for p in palabras_proc:
        ocurrencias = df_filtrado[columna].fillna("").apply(lambda txt: quitar_tildes(txt.lower()).count(p))
        total[p] = ocurrencias.sum()
        textos[p] = (ocurrencias > 0).sum()
    return df_filtrado, total, textos

# --- INTERFAZ PRINCIPAL ---
uploaded_file = st.file_uploader("📥 Carga tu archivo Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    with st.spinner("Cargando archivo..."):
        df = cargar_excel(uploaded_file)

    # --- DETECTAR BASE AUTOMÁTICAMENTE ---
    columnas = df.columns
    if {'PREG 24','PREG 25','PREG 26'}.issubset(columnas):
        tipo_base = "3° y 4°"
        columnas_objetivo = ['PREG 24','PREG 25','PREG 26']
    elif {'PREG 21','PREG 22','PREG 23'}.issubset(columnas):
        tipo_base = "1° y 2°"
        columnas_objetivo = ['PREG 21','PREG 22','PREG 23']
    elif {'PREG 22','PREG 23','PREG 24'}.issubset(columnas):
        tipo_base = "ACP"
        columnas_objetivo = ['PREG 22','PREG 23','PREG 24']
    elif {'PREG 25','PREG 26','PREG 27'}.issubset(columnas):
        tipo_base = "GE o Dynamic"
        columnas_objetivo = ['PREG 25','PREG 26','PREG 27']
    else:
        st.error("No se pudo identificar el tipo de base. Verifica las columnas.")
        st.stop()

    st.success(f"Base detectada automáticamente: **{tipo_base}**")

    # --- FILTROS SEDE / NIVEL ---
    if 'SEDE' in df.columns:
        sede_sel = st.selectbox("Filtrar por SEDE:", ["Todos"] + sorted(df['SEDE'].dropna().unique().tolist()))
        if sede_sel != "Todos":
            df = df[df['SEDE'] == sede_sel]
    if 'NIVEL' in df.columns:
        nivel_sel = st.selectbox("Filtrar por NIVEL:", ["Todos"] + sorted(df['NIVEL'].dropna().unique().tolist()))
        if nivel_sel != "Todos":
            df = df[df['NIVEL'] == nivel_sel]

    # --- SELECCIONAR COLUMNA ---
    columna_seleccionada = st.selectbox("Selecciona la pregunta a analizar:", columnas_objetivo)

    # --- ANÁLISIS PRINCIPAL ---
    with st.spinner("Procesando análisis..."):
        es_cambios = 'cambio' in columna_seleccionada.lower()
        df[columna_seleccionada + '_sentimiento'] = df[columna_seleccionada].apply(lambda x: sentimiento_vader(x, es_cambios))
        graficar_distribucion_sentimientos(df[columna_seleccionada + '_sentimiento'], columna_seleccionada)

        textos_proc = preprocesar_textos(df[columna_seleccionada])
        if len(textos_proc) >= 10:
            st.subheader("Temas identificados")
            temas = codificar_temas(textos_proc)
            for t in temas:
                st.write("• " + t)

            st.subheader("☁️ Nube de palabras")
            mostrar_nube(textos_proc)

            st.subheader("🔠 Frecuencia de palabras")
            graficar_frecuencias_palabras(textos_proc)
        else:
            st.info("No hay suficientes textos para el análisis de temas (mínimo 10).")

    # --- NUEVO FILTRO POR PALABRAS CLAVE ---
    st.subheader("🔍 Búsqueda de palabras clave en los comentarios")
    palabras_entrada = st.text_input("Ingresa palabras separadas por comas para buscar (ej: examen, presencial, rápido)")
    if st.button("Buscar palabras"):
        if palabras_entrada.strip():
            palabras_lista = [p.strip() for p in palabras_entrada.split(",")]
            df_filtrado_palabras, total_ocurrencias, textos_con_palabra = filtrar_por_palabras(df, columna_seleccionada, palabras_lista)
            if df_filtrado_palabras.empty:
                st.warning("No se encontraron comentarios que contengan esas palabras.")
            else:
                st.write(f"Se encontraron {len(df_filtrado_palabras)} comentarios que contienen al menos una de las palabras buscadas.")
                st.write("### Ocurrencias totales por palabra:")
                df_ocurrencias = pd.DataFrame({
                    "Palabra": [p.replace('_', ' ') for p in total_ocurrencias.keys()],
                    "Total de apariciones": total_ocurrencias.values(),
                    "Número de textos que la mencionan": [textos_con_palabra[p] for p in total_ocurrencias.keys()]
                })
                st.dataframe(df_ocurrencias)
                st.write("### Comentarios filtrados:")
                st.write(df_filtrado_palabras[[columna_seleccionada]])
        else:
            st.warning("Por favor, ingresa al menos una palabra para buscar.")
else:
    st.info("📂 Carga un archivo Excel para comenzar el análisis.")
