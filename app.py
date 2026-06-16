import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL DE LA APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="App Analizadora de Datasets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# FUNCIONES UTILITARIAS
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    return df

def detectar_tipos(df):
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    binarias = [c for c in numericas if df[c].nunique() == 2]
    return numericas, categoricas, binarias

def detectar_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lim_inf) | (df[col] > lim_sup)]
    return outliers, lim_inf, lim_sup

# ─────────────────────────────────────────────
# SIDEBAR — MENÚ DE NAVEGACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=80)
    st.title("📊 Menú Principal")
    seccion = st.radio(
        "Selecciona una sección:",
        ["🏠 Home",
         "📂 Carga y Perfil del Dataset",
         "⚙️ Procesamiento de Datos",
         "📈 Análisis Visual"]
    )
    st.markdown("---")
    st.caption("Diploma Business Analyst")
    st.caption("DMC Institute · 2025")

    if "df" in st.session_state:
        st.success(f"✅ Dataset cargado\n{st.session_state.df.shape[0]} filas · {st.session_state.df.shape[1]} columnas")

# ═══════════════════════════════════════════════════════
# SECCIÓN 1 — HOME
# ═══════════════════════════════════════════════════════
if seccion == "🏠 Home":
    st.title("📊 App Analizadora de Datasets con Streamlit")
    st.subheader("Exploración y Visualización de Datos con Python")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Objetivo del Proyecto
        Esta aplicación interactiva permite cargar, validar, procesar y visualizar datos de manera 
        dinámica. Está diseñada para realizar **análisis exploratorio de datos (EDA)** sobre el dataset 
        de salud mental en adolescentes, integrando gráficos interactivos y estadísticas descriptivas.
        
        ### 👤 Autor
        **Miguel Angel** · Diploma Business Analyst  
        **Curso:** Exploración y Visualización de Datos con Python  
        **Institución:** DMC Institute · 2025
        
        ### ⚠️ Nota de Uso Responsable
        Los resultados presentados son **exploratorios** y no reemplazan validación técnica o 
        profesional. El análisis de salud mental adolescente no debe interpretarse como diagnóstico clínico.
        """)

    with col2:
        st.info("🛠️ **Tecnologías Utilizadas**\n\n"
                "- 🐍 Python 3.x\n"
                "- 🐼 Pandas\n"
                "- 🌊 Streamlit\n"
                "- 📊 Plotly\n"
                "- 🎨 Matplotlib\n"
                "- 🌊 Seaborn\n"
                "- 🐙 GitHub")

    st.markdown("---")
    st.subheader("📁 Dataset Disponible")

    with st.expander("🧠 Teen Mental Health Dataset — Haz clic para ver detalles"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Descripción:** Dataset sobre hábitos digitales, sueño, actividad física, 
            interacción social y variables de bienestar en adolescentes.
            
            **Tamaño:** 1,200 filas · 13 columnas
            """)
        with col_b:
            st.markdown("""
            **Variables principales:**
            - `age`, `gender` — Datos demográficos
            - `daily_social_media_hours` — Uso diario de redes
            - `platform_usage` — Plataforma (Instagram/TikTok/Both)
            - `sleep_hours` — Horas de sueño
            - `stress_level`, `anxiety_level` — Niveles de estrés y ansiedad
            - `addiction_level` — Nivel de adicción
            - `depression_label` — Etiqueta binaria de depresión
            """)

# ═══════════════════════════════════════════════════════
# SECCIÓN 2 — CARGA Y PERFIL DEL DATASET
# ═══════════════════════════════════════════════════════
elif seccion == "📂 Carga y Perfil del Dataset":
    st.title("📂 Carga y Perfil del Dataset")
    st.markdown("---")

    archivo = st.file_uploader(
        "Sube tu archivo CSV aquí",
        type=["csv"],
        help="Sube el archivo Teen_Mental_Health_Dataset.csv"
    )

    if archivo is not None:
        df = cargar_datos(archivo)
        st.session_state.df = df
        st.success(f"✅ Archivo '{archivo.name}' cargado correctamente.")
    elif "df" in st.session_state:
        df = st.session_state.df
        st.info("📌 Usando el dataset cargado previamente.")
    else:
        st.warning("⚠️ Por favor carga un archivo CSV para continuar.")
        st.stop()

    # — MÉTRICAS RÁPIDAS
    st.subheader("📊 Métricas Rápidas")
    numericas, categoricas, binarias = detectar_tipos(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📋 Filas", df.shape[0])
    c2.metric("📐 Columnas", df.shape[1])
    c3.metric("🔢 Variables Numéricas", len(numericas))
    c4.metric("🔤 Variables Categóricas", len(categoricas))
    c5.metric("❓ Valores Nulos", int(df.isnull().sum().sum()))
    c6.metric("🔁 Duplicados", int(df.duplicated().sum()))

    st.markdown("---")

    # — VISTA PREVIA
    st.subheader("👁️ Vista Previa del Dataset")
    n_filas = st.slider("Número de filas a mostrar:", 5, 50, 10)
    st.dataframe(df.head(n_filas), use_container_width=True)

    st.markdown("---")

    # — TIPOS DE DATOS Y COLUMNAS
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📐 Dimensiones")
        st.write(f"**Forma del DataFrame:** {df.shape}")
        st.write(f"**Columnas:** {', '.join(df.columns.tolist())}")

    with col2:
        st.subheader("🗂️ Tipos de Datos")
        tipos_df = pd.DataFrame({
            "Columna": df.dtypes.index,
            "Tipo": df.dtypes.values.astype(str)
        })
        st.dataframe(tipos_df, use_container_width=True)

    st.markdown("---")

    # — SELECCIÓN DE COLUMNAS
    st.subheader("🔍 Seleccionar Columnas de Interés")
    cols_seleccionadas = st.multiselect(
        "Elige las columnas que quieres analizar:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )
    if cols_seleccionadas:
        if st.checkbox("Mostrar datos de columnas seleccionadas"):
            st.dataframe(df[cols_seleccionadas].head(20), use_container_width=True)

# ═══════════════════════════════════════════════════════
# SECCIÓN 3 — PROCESAMIENTO DE DATOS
# ═══════════════════════════════════════════════════════
elif seccion == "⚙️ Procesamiento de Datos":
    st.title("⚙️ Procesamiento de Datos")
    st.markdown("---")

    if "df" not in st.session_state:
        st.error("❌ No hay datos cargados. Ve a 'Carga y Perfil del Dataset' primero.")
        st.stop()

    df = st.session_state.df
    numericas, categoricas, binarias = detectar_tipos(df)

    # — CLASIFICACIÓN DE VARIABLES
    st.subheader("🏷️ Clasificación Automática de Variables")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔢 Variables Numéricas:**")
        for v in numericas:
            st.write(f"- `{v}`")

    with col2:
        st.markdown("**🔤 Variables Categóricas:**")
        if categoricas:
            for v in categoricas:
                st.write(f"- `{v}`")
        else:
            st.info("No se encontraron variables categóricas.")

    with col3:
        st.markdown("**⚡ Variables Binarias:**")
        if binarias:
            for v in binarias:
                st.write(f"- `{v}`")
        else:
            st.info("No se encontraron variables binarias.")

    st.markdown("---")

    # — VALORES NULOS
    st.subheader("❓ Análisis de Valores Nulos")
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df) * 100).round(2)
    df_nulos = pd.DataFrame({
        "Columna": nulos.index,
        "Valores Nulos": nulos.values,
        "Porcentaje (%)": porcentaje_nulos.values
    })
    df_nulos = df_nulos[df_nulos["Valores Nulos"] > 0]

    if df_nulos.empty:
        st.success("✅ ¡No se encontraron valores nulos en el dataset!")
    else:
        st.warning(f"⚠️ Se encontraron {df_nulos['Valores Nulos'].sum()} valores nulos.")
        st.dataframe(df_nulos, use_container_width=True)
        fig_nulos = px.bar(df_nulos, x="Columna", y="Porcentaje (%)",
                           title="Porcentaje de Valores Nulos por Columna",
                           color="Porcentaje (%)", color_continuous_scale="reds")
        st.plotly_chart(fig_nulos, use_container_width=True)

    st.markdown("---")

    # — DUPLICADOS
    st.subheader("🔁 Detección de Duplicados")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        st.success("✅ No se encontraron registros duplicados.")
    else:
        st.warning(f"⚠️ Se encontraron {n_dup} registros duplicados.")

    st.markdown("---")

    # — ESTADÍSTICA DESCRIPTIVA
    st.subheader("📊 Estadística Descriptiva")
    if st.checkbox("Mostrar estadísticas descriptivas de variables numéricas"):
        st.dataframe(df[numericas].describe().round(2), use_container_width=True)

    st.markdown("---")

    # — OUTLIERS
    st.subheader("🎯 Detección de Outliers (Regla IQR)")
    col_outlier = st.selectbox("Selecciona una variable numérica:", numericas)

    if col_outlier:
        outliers, lim_inf, lim_sup = detectar_outliers_iqr(df, col_outlier)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Límite Inferior", f"{lim_inf:.2f}")
            st.metric("Límite Superior", f"{lim_sup:.2f}")
        with col_b:
            st.metric("Total Outliers", len(outliers))
            st.metric("% del dataset", f"{(len(outliers)/len(df)*100):.1f}%")

        fig_box, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df[col_outlier], color="#4C9BE8", ax=ax)
        ax.set_title(f"Boxplot — {col_outlier}", fontsize=13)
        ax.set_xlabel(col_outlier)
        st.pyplot(fig_box)

    st.markdown("---")

    # — FILTROS DINÁMICOS
    st.subheader("🔧 Filtros Dinámicos")
    col_filtro = st.selectbox("Filtrar por variable categórica:", categoricas)
    if col_filtro:
        opciones = df[col_filtro].unique().tolist()
        seleccion = st.multiselect(f"Valores de {col_filtro}:", opciones, default=opciones)
        df_filtrado = df[df[col_filtro].isin(seleccion)]
        st.write(f"Registros filtrados: **{len(df_filtrado)}** de {len(df)}")
        if st.checkbox("Ver datos filtrados"):
            st.dataframe(df_filtrado.head(20), use_container_width=True)

# ═══════════════════════════════════════════════════════
# SECCIÓN 4 — ANÁLISIS VISUAL
# ═══════════════════════════════════════════════════════
elif seccion == "📈 Análisis Visual":
    st.title("📈 Análisis Visual")
    st.markdown("---")

    if "df" not in st.session_state:
        st.error("❌ No hay datos cargados. Ve a 'Carga y Perfil del Dataset' primero.")
        st.stop()

    df = st.session_state.df
    numericas, categoricas, binarias = detectar_tipos(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Resumen",
        "📊 Univariado",
        "🔗 Bivariado",
        "🧩 Multivariado",
        "⏱️ Temporal",
        "💡 Insights"
    ])

    # ───── TAB 1: RESUMEN ─────
    with tab1:
        st.subheader("📋 Resumen General del Dataset")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Filas", df.shape[0])
        col2.metric("Total Columnas", df.shape[1])
        col3.metric("Variables Num.", len(numericas))
        col4.metric("Variables Cat.", len(categoricas))

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Tipos de datos:**")
            tipos_df = pd.DataFrame({"Tipo": df.dtypes.astype(str)})
            st.dataframe(tipos_df, use_container_width=True)
        with col_b:
            st.markdown("**Nulos y duplicados:**")
            resumen = pd.DataFrame({
                "Métrica": ["Valores Nulos", "Duplicados"],
                "Cantidad": [df.isnull().sum().sum(), df.duplicated().sum()]
            })
            st.dataframe(resumen, use_container_width=True)

        st.markdown("---")
        if st.checkbox("Mostrar tabla resumen estadístico completo"):
            st.dataframe(df[numericas].describe().round(2), use_container_width=True)

    # ───── TAB 2: UNIVARIADO ─────
    with tab2:
        st.subheader("📊 Análisis Univariado")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribución de Variables Numéricas")
            var_num = st.selectbox("Variable numérica:", numericas, key="uni_num")
            bins_val = st.slider("Número de bins:", 5, 50, 20, key="bins_uni")

            fig_hist = px.histogram(df, x=var_num, nbins=bins_val,
                                    title=f"Histograma — {var_num}",
                                    color_discrete_sequence=["#4C9BE8"],
                                    marginal="box")
            fig_hist.update_layout(bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(f"📌 La distribución de **{var_num}** muestra cómo se concentran los valores en los adolescentes del dataset.")

        with col2:
            st.markdown("#### Variables Categóricas")
            var_cat = st.selectbox("Variable categórica:", categoricas, key="uni_cat")
            conteo = df[var_cat].value_counts().reset_index()
            conteo.columns = [var_cat, "Conteo"]

            fig_bar = px.bar(conteo, x=var_cat, y="Conteo",
                             title=f"Conteo por {var_cat}",
                             color=var_cat,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(f"📌 Se observa la distribución de categorías en **{var_cat}**.")

        st.markdown("---")

        # Boxplot con Seaborn
        st.markdown("#### Boxplot por Género (Seaborn)")
        var_box = st.selectbox("Variable para boxplot:", numericas, key="box_gen")
        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        sns.set_style("whitegrid")
        sns.boxplot(data=df, x="gender", y=var_box,
                    palette=["#4C9BE8", "#FF6B9D"], ax=ax_s)
        ax_s.set_title(f"{var_box} por Género", fontsize=13, fontweight='bold')
        ax_s.set_xlabel("Género")
        ax_s.set_ylabel(var_box)
        st.pyplot(fig_s)
        st.caption(f"📌 Comparación de **{var_box}** entre géneros mediante boxplots.")

        st.markdown("---")

        # Gráfico circular
        st.markdown("#### Proporción por Plataforma (Matplotlib)")
        platform_counts = df["platform_usage"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        colores = ["#4C9BE8", "#FF6B9D", "#FFD93D"]
        ax_pie.pie(platform_counts.values, labels=platform_counts.index,
                   autopct='%1.1f%%', colors=colores, startangle=90,
                   wedgeprops=dict(edgecolor='white', linewidth=2))
        ax_pie.set_title("Distribución por Plataforma de Redes Sociales", fontsize=13)
        st.pyplot(fig_pie)
        st.caption("📌 Las tres plataformas (Instagram, TikTok y ambas) tienen proporciones similares entre los adolescentes.")

    # ───── TAB 3: BIVARIADO ─────
    with tab3:
        st.subheader("🔗 Análisis Bivariado")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Scatter Plot — Variables Numéricas")
            var_x = st.selectbox("Variable X:", numericas, key="biv_x", index=0)
            var_y = st.selectbox("Variable Y:", numericas, key="biv_y", index=2)
            color_by = st.selectbox("Color por:", categoricas, key="biv_color")

            fig_scatter = px.scatter(df, x=var_x, y=var_y, color=color_by,
                                     title=f"{var_x} vs {var_y}",
                                     color_discrete_sequence=px.colors.qualitative.Set2,
                                     hover_data=df.columns.tolist(),
                                     opacity=0.7)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"📌 Relación entre **{var_x}** y **{var_y}** segmentada por **{color_by}**.")

        with col2:
            st.markdown("#### Boxplot por Categoría")
            cat_biv = st.selectbox("Variable categórica:", categoricas, key="biv_cat")
            num_biv = st.selectbox("Variable numérica:", numericas, key="biv_num")

            fig_box2 = px.box(df, x=cat_biv, y=num_biv,
                              color=cat_biv,
                              title=f"{num_biv} por {cat_biv}",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_box2, use_container_width=True)
            st.caption(f"📌 Comparación de **{num_biv}** entre las categorías de **{cat_biv}**.")

        st.markdown("---")

        # Barras agrupadas
        st.markdown("#### Barras Agrupadas — Nivel de Depresión por Plataforma")
        dep_platform = df.groupby(["platform_usage", "depression_label"]).size().reset_index(name="conteo")
        dep_platform["depression_label"] = dep_platform["depression_label"].map({0: "Sin depresión", 1: "Con depresión"})
        fig_grouped = px.bar(dep_platform, x="platform_usage", y="conteo",
                             color="depression_label", barmode="group",
                             title="Casos de Depresión por Plataforma",
                             color_discrete_sequence=["#4C9BE8", "#FF6B9D"])
        st.plotly_chart(fig_grouped, use_container_width=True)
        st.caption("📌 Permite explorar si existe asociación entre la plataforma usada y la etiqueta de depresión.")

    # ───── TAB 4: MULTIVARIADO ─────
    with tab4:
        st.subheader("🧩 Análisis Multivariado")

        # Heatmap de correlación con Seaborn
        st.markdown("#### Mapa de Calor de Correlación (Seaborn)")
        vars_corr = st.multiselect("Selecciona variables para la correlación:",
                                    numericas, default=numericas)
        if len(vars_corr) >= 2:
            corr = df[vars_corr].corr().round(2)
            fig_heat, ax_heat = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
                        fmt=".2f", linewidths=0.5, ax=ax_heat,
                        annot_kws={"size": 10})
            ax_heat.set_title("Matriz de Correlación — Variables Numéricas",
                               fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_heat)
            st.caption("📌 Los valores cercanos a 1 o -1 indican correlaciones fuertes. La diagonal siempre es 1.")
        else:
            st.warning("⚠️ Selecciona al menos 2 variables.")

        st.markdown("---")

        # Barras apiladas
        st.markdown("#### Barras Apiladas — Nivel de Estrés por Plataforma e Interacción Social")
        pivot_data = df.groupby(["platform_usage", "social_interaction_level"])["stress_level"].mean().reset_index()
        fig_stack = px.bar(pivot_data, x="platform_usage", y="stress_level",
                           color="social_interaction_level", barmode="stack",
                           title="Nivel de Estrés promedio por Plataforma e Interacción Social",
                           color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_stack, use_container_width=True)
        st.caption("📌 Visualización multivariada que integra plataforma, nivel de interacción y estrés promedio.")

        st.markdown("---")

        # Scatter 3 variables
        st.markdown("#### Análisis de 3 Variables con Scatter")
        v1 = st.selectbox("Variable X:", numericas, key="multi_x", index=0)
        v2 = st.selectbox("Variable Y:", numericas, key="multi_y", index=2)
        v3 = st.selectbox("Tamaño por:", numericas, key="multi_size", index=4)
        cat_color = st.selectbox("Color por:", categoricas, key="multi_color")

        fig_multi = px.scatter(df, x=v1, y=v2, size=v3,
                               color=cat_color, title=f"{v1} vs {v2} (tamaño: {v3})",
                               color_discrete_sequence=px.colors.qualitative.Bold,
                               size_max=15, opacity=0.7)
        st.plotly_chart(fig_multi, use_container_width=True)

    # ───── TAB 5: TEMPORAL ─────
    with tab5:
        st.subheader("⏱️ Análisis por Edad (Componente Temporal)")
        st.info("ℹ️ Este dataset no tiene fechas. Se utiliza la variable **edad** como eje de análisis evolutivo.")

        # Evolución por edad
        st.markdown("#### Evolución de Métricas por Edad")
        var_temporal = st.selectbox("Variable a analizar:", numericas, key="temp_var",
                                    index=numericas.index("stress_level") if "stress_level" in numericas else 0)

        df_age = df.groupby("age")[var_temporal].mean().reset_index()
        fig_line = px.line(df_age, x="age", y=var_temporal,
                           title=f"Promedio de {var_temporal} por Edad",
                           markers=True,
                           color_discrete_sequence=["#4C9BE8"])
        fig_line.update_traces(line_width=2.5, marker_size=8)
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption(f"📌 Muestra cómo evoluciona el promedio de **{var_temporal}** a medida que aumenta la edad.")

        st.markdown("---")

        # Múltiples variables por edad con Matplotlib
        st.markdown("#### Comparación de Variables por Edad (Matplotlib)")
        vars_age = st.multiselect("Selecciona variables:", numericas,
                                   default=["stress_level", "anxiety_level", "sleep_hours"])

        if vars_age:
            df_multi_age = df.groupby("age")[vars_age].mean()
            fig_ma, ax_ma = plt.subplots(figsize=(10, 5))
            for col in vars_age:
                ax_ma.plot(df_multi_age.index, df_multi_age[col],
                           marker='o', label=col, linewidth=2)
            ax_ma.set_title("Evolución de Variables por Edad", fontsize=13, fontweight='bold')
            ax_ma.set_xlabel("Edad")
            ax_ma.set_ylabel("Valor Promedio")
            ax_ma.legend()
            ax_ma.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_ma)
            st.caption("📌 Comparación de múltiples variables por edad usando Matplotlib.")

    # ───── TAB 6: INSIGHTS ─────
    with tab6:
        st.subheader("💡 Hallazgos Clave e Insights")

        # Métricas de resumen
        col1, col2, col3 = st.columns(3)
        col1.metric("Promedio Horas Redes Sociales", f"{df['daily_social_media_hours'].mean():.1f} hrs/día")
        col2.metric("Promedio Horas de Sueño", f"{df['sleep_hours'].mean():.1f} hrs")
        col3.metric("Casos con Depresión", f"{df['depression_label'].sum()} ({df['depression_label'].mean()*100:.1f}%)")

        st.markdown("---")

        # Insight 1
        st.markdown("#### 📌 Insight 1 — Uso de Redes y Estrés")
        fig_i1 = px.scatter(df, x="daily_social_media_hours", y="stress_level",
                             color="gender", trendline="ols",
                             title="Horas en Redes Sociales vs Nivel de Estrés",
                             color_discrete_sequence=["#4C9BE8", "#FF6B9D"],
                             opacity=0.6)
        st.plotly_chart(fig_i1, use_container_width=True)
        st.markdown("> 📊 Se observa una tendencia positiva entre el uso de redes sociales y el nivel de estrés en los adolescentes. Mayor tiempo en redes se asocia con mayor estrés.")

        st.markdown("---")

        # Insight 2
        st.markdown("#### 📌 Insight 2 — Sueño e Interacción Social")
        fig_i2 = px.box(df, x="social_interaction_level", y="sleep_hours",
                         color="social_interaction_level",
                         title="Horas de Sueño por Nivel de Interacción Social",
                         category_orders={"social_interaction_level": ["low", "medium", "high"]},
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_i2, use_container_width=True)
        st.markdown("> 📊 Los adolescentes con mayor interacción social tienden a tener patrones de sueño diferentes. El análisis es exploratorio y no implica causalidad.")

        st.markdown("---")

        # Insight 3
        st.markdown("#### 📌 Insight 3 — Depresión por Plataforma y Género")
        dep_gen = df.groupby(["platform_usage", "gender"])["depression_label"].mean().reset_index()
        dep_gen["% Depresión"] = (dep_gen["depression_label"] * 100).round(1)
        fig_i3 = px.bar(dep_gen, x="platform_usage", y="% Depresión",
                         color="gender", barmode="group",
                         title="Tasa de Depresión (%) por Plataforma y Género",
                         color_discrete_sequence=["#4C9BE8", "#FF6B9D"])
        st.plotly_chart(fig_i3, use_container_width=True)
        st.markdown("> 📊 La tasa de depresión varía según la plataforma y el género. Estos patrones son exploratorios y no constituyen un diagnóstico clínico.")

        st.markdown("---")
        st.success("✅ Análisis exploratorio completado. Los hallazgos presentados son de carácter descriptivo e ilustrativo.")
