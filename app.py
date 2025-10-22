import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import nltk

# --- DESCARGAR STOPWORDS ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# --- CONFIGURACIÓN STREAMLIT ---
st.set_page_config(page_title="Análisis de Encuestas", layout="centered")
st.title("📊 Análisis de Preguntas Abiertas de Encuestas Académicas")

# --- PARÁMETROS ---
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
    "mas didacticas": "mas_didacticas",
    "mas actividades": "mas_actividades",
    "mas apoyo": "mas_apoyo",
    "ensayos presenciales": "ensayos_presenciales",
    "ensayos virtuales": "ensayos_virtuales",
    "cambio de profesor": "cambio_de_profesor",
    "cambio de profe": "cambio_de_profe",
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

analyzer = SentimentIntensityAnalyzer()

# --- FUNCIONES ---
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
    score = analyzer.polarity_scores(texto)['compound']
    if score >= 0.05: return 'Positivo'
    elif score <= -0.05: return 'Negativo'
    else: return 'Neutro'

def codificar_temas(textos, n_topics=3, n_palabras=5):
    vectorizer = CountVectorizer(stop_words=list(stopwords_totales), max_features=1000)
    X = vectorizer.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
    lda.fit(X)
    palabras = vectorizer.get_feature_names_out()
    temas = []
    for idx, topic in enumerate(lda.components_):
        top_words = [palabras[i].replace('_', ' ') for i in topic.argsort()[-n_palabras:][::-1]]
        temas.append(f"Tema {idx+1}: " + ", ".join(top_words))
    return temas

def graficar_distribucion_sentimientos(data, columna):
    plt.figure(figsize=(4,3))
    sns.countplot(x=data, palette={'Negativo':'#e74c3c','Neutro':'#95a5a6','Positivo':'#2ecc71'})
    plt.title(f"Sentimientos - {columna}")
    st.pyplot(plt)
    plt.clf()

def mostrar_nube(textos):
    texto_unido = " ".join(textos)
    if not texto_unido.strip():
        st.info("No hay texto suficiente para la nube.")
        return
    wc = WordCloud(stopwords=stopwords_totales, background_color="white", width=600, height=300).generate(texto_unido)
    plt.figure(figsize=(6,3))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    plt.clf()

def graficar_frecuencias_palabras(textos):
    palabras = " ".join(textos).split()
    comunes = Counter(palabras).most_common(15)
    plt.figure(figsize=(6,3))
    sns.barplot(x=[c[1] for c in comunes], y=[c[0].replace('_',' ') for c in comunes], palette="viridis")
    plt.title("Frecuencia de palabras")
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


# --- INTERFAZ ---
uploaded_file = st.file_uploader("Carga tu archivo Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    nombre = uploaded_file.name.lower()

    # Detección automática de base
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

    opciones = ["3° y 4°", "1° y 2°", "ACP", "GE", "Dynamic"]
    tipo_base = st.selectbox("Selecciona el tipo de base:", opciones, index=opciones.index(tipo_detectado))

    columnas_por_base = {
        "3° y 4°": ['PREG 24', 'PREG 25', 'PREG 26'],
        "1° y 2°": ['PREG 21', 'PREG 22', 'PREG 23'],
        "ACP": ['PREG 22', 'PREG 23', 'PREG 24'],
        "GE": ['PREG 25', 'PREG 26', 'PREG 27'],
        "Dynamic": ['PREG 25', 'PREG 26', 'PREG 27']
    }
    # Filtros SEDE / NIVEL
    if 'SEDE' in df.columns:
        sede_sel = st.selectbox("Filtrar por SEDE:", ["Todos"] + sorted(df['SEDE'].dropna().unique().tolist()))
        if sede_sel != "Todos":
            df = df[df['SEDE'] == sede_sel]
    if 'NIVEL' in df.columns:
        nivel_sel = st.selectbox("Filtrar por NIVEL:", ["Todos"] + sorted(df['NIVEL'].dropna().unique().tolist()))
        if nivel_sel != "Todos":
            df = df[df['NIVEL'] == nivel_sel]

    st.divider()
    st.subheader(f"🔍 Análisis de {columna_seleccionada}")

    es_cambios = 'cambio' in columna_seleccionada.lower()
    df[columna_seleccionada + '_sentimiento'] = df[columna_seleccionada].apply(lambda x: sentimiento_vader(x, es_cambios))

    graficar_distribucion_sentimientos(df[columna_seleccionada + '_sentimiento'], columna_seleccionada)
    textos_proc = preprocesar_textos(df[columna_seleccionada])

    if len(textos_proc) >= 10:
        st.subheader("Temas identificados")
        temas = codificar_temas(textos_proc)
        for t in temas:
            st.write("• " + t)

        mostrar_nube(textos_proc)
        graficar_frecuencias_palabras(textos_proc)
    else:
        st.info("No hay suficientes textos para el análisis de temas (mínimo 10).")

    # --- FILTRO POR PALABRAS ---
    st.subheader("🪶 Búsqueda de palabras clave")
    palabras_entrada = st.text_input("Palabras separadas por comas (ej: examen, presencial, rápido)")
    if st.button("Buscar palabras"):
        if palabras_entrada.strip():
            palabras_lista = [p.strip() for p in palabras_entrada.split(",")]
            df_filtrado_palabras, total_ocurrencias, textos_con_palabra = filtrar_por_palabras(df, columna_seleccionada, palabras_lista)
            if df_filtrado_palabras.empty:
                st.warning("No se encontraron comentarios con esas palabras.")
            else:
                st.write(f"**{len(df_filtrado_palabras)} comentarios** contienen al menos una de las palabras buscadas.")
                df_ocurrencias = pd.DataFrame({
                    "Palabra": [p.replace('_',' ') for p in total_ocurrencias.keys()],
                    "Total de apariciones": total_ocurrencias.values(),
                    "Número de textos que la mencionan": [textos_con_palabra[p] for p in total_ocurrencias.keys()]
                })
                st.dataframe(df_ocurrencias)
                st.write("### Comentarios filtrados:")
                st.write(df_filtrado_palabras[[columna_seleccionada]])
        else:
            st.warning("Ingresa al menos una palabra para buscar.")
else:
    st.info("📂 Carga un archivo Excel para comenzar el análisis.")

