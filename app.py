import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Configuracion general de la aplicacion
st.set_page_config(
    page_title="App Analizadora de Datasets",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# FUNCIONES
# -------------------------------------------------------

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

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:
    st.title("Menu Principal")
    st.markdown("---")
    seccion = st.radio(
        "Selecciona una seccion:",
        ["Home",
         "Carga y Perfil del Dataset",
         "Procesamiento de Datos",
         "Analisis Visual"]
    )
    st.markdown("---")
    st.caption("Diploma Business Analyst")
    st.caption("DMC Institute - 2025")

    if "df" in st.session_state:
        st.success(
            f"Dataset cargado: {st.session_state.df.shape[0]} filas "
            f"y {st.session_state.df.shape[1]} columnas"
        )

# -------------------------------------------------------
# SECCION 1: HOME
# -------------------------------------------------------

if seccion == "Home":
    st.title("App Analizadora de Datasets con Streamlit")
    st.subheader("Exploracion y Visualizacion de Datos con Python")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Objetivo del Proyecto
        Esta aplicacion permite cargar, validar, procesar y visualizar datos de manera
        dinamica. Esta disenada para realizar analisis exploratorio de datos (EDA) sobre
        el dataset de salud mental en adolescentes, integrando graficos interactivos y
        estadisticas descriptivas.

        ### Autor
        **Miguel Angel** - Diploma Business Analyst  
        Curso: Exploracion y Visualizacion de Datos con Python  
        Institucion: DMC Institute - 2025

        ### Nota de Uso Responsable
        Los resultados presentados son exploratorios y no reemplazan validacion tecnica
        o profesional. El analisis de salud mental adolescente no debe interpretarse como
        diagnostico clinico.
        """)

    with col2:
        st.info(
            "Tecnologias utilizadas:\n\n"
            "- Python 3.x\n"
            "- Pandas\n"
            "- Streamlit\n"
            "- Plotly\n"
            "- Matplotlib\n"
            "- Seaborn\n"
            "- GitHub"
        )

    st.markdown("---")
    st.subheader("Dataset disponible")

    with st.expander("Teen Mental Health Dataset - Haz clic para ver detalles"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Descripcion:** Dataset sobre habitos digitales, sueno, actividad fisica,
            interaccion social y variables de bienestar en adolescentes.

            **Tamano:** 1,200 filas - 13 columnas
            """)
        with col_b:
            st.markdown("""
            **Variables principales:**
            - age, gender: datos demograficos
            - daily_social_media_hours: uso diario de redes
            - platform_usage: plataforma usada
            - sleep_hours: horas de sueno
            - stress_level, anxiety_level: niveles de estres y ansiedad
            - addiction_level: nivel de adiccion
            - depression_label: variable binaria de depresion
            """)

# -------------------------------------------------------
# SECCION 2: CARGA Y PERFIL DEL DATASET
# -------------------------------------------------------

elif seccion == "Carga y Perfil del Dataset":
    st.title("Carga y Perfil del Dataset")
    st.markdown("---")

    archivo = st.file_uploader(
        "Sube tu archivo CSV aqui",
        type=["csv"],
        help="Sube el archivo Teen_Mental_Health_Dataset.csv"
    )

    if archivo is not None:
        df = cargar_datos(archivo)
        st.session_state.df = df
        st.success(f"Archivo '{archivo.name}' cargado correctamente.")
    elif "df" in st.session_state:
        df = st.session_state.df
        st.info("Usando el dataset cargado previamente.")
    else:
        st.warning("Por favor carga un archivo CSV para continuar.")
        st.stop()

    # Metricas rapidas
    st.subheader("Metricas rapidas")
    numericas, categoricas, binarias = detectar_tipos(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Var. Numericas", len(numericas))
    c4.metric("Var. Categoricas", len(categoricas))
    c5.metric("Valores Nulos", int(df.isnull().sum().sum()))
    c6.metric("Duplicados", int(df.duplicated().sum()))

    st.markdown("---")

    # Vista previa
    st.subheader("Vista previa del dataset")
    n_filas = st.slider("Numero de filas a mostrar:", 5, 50, 10)
    st.dataframe(df.head(n_filas), use_container_width=True)

    st.markdown("---")

    # Tipos de datos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dimensiones")
        st.write(f"**Forma del DataFrame:** {df.shape}")
        st.write(f"**Columnas:** {', '.join(df.columns.tolist())}")

    with col2:
        st.subheader("Tipos de datos")
        tipos_df = pd.DataFrame({
            "Columna": df.dtypes.index,
            "Tipo": df.dtypes.values.astype(str)
        })
        st.dataframe(tipos_df, use_container_width=True)

    st.markdown("---")

    # Seleccion de columnas
    st.subheader("Seleccion de columnas de interes")
    cols_seleccionadas = st.multiselect(
        "Elige las columnas que quieres analizar:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )
    if cols_seleccionadas:
        if st.checkbox("Mostrar datos de columnas seleccionadas"):
            st.dataframe(df[cols_seleccionadas].head(20), use_container_width=True)

# -------------------------------------------------------
# SECCION 3: PROCESAMIENTO DE DATOS
# -------------------------------------------------------

elif seccion == "Procesamiento de Datos":
    st.title("Procesamiento de Datos")
    st.markdown("---")

    if "df" not in st.session_state:
        st.error("No hay datos cargados. Ve a 'Carga y Perfil del Dataset' primero.")
        st.stop()

    df = st.session_state.df
    numericas, categoricas, binarias = detectar_tipos(df)

    # Clasificacion de variables
    st.subheader("Clasificacion automatica de variables")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Variables Numericas:**")
        for v in numericas:
            st.write(f"- {v}")

    with col2:
        st.markdown("**Variables Categoricas:**")
        if categoricas:
            for v in categoricas:
                st.write(f"- {v}")
        else:
            st.info("No se encontraron variables categoricas.")

    with col3:
        st.markdown("**Variables Binarias:**")
        if binarias:
            for v in binarias:
                st.write(f"- {v}")
        else:
            st.info("No se encontraron variables binarias.")

    st.markdown("---")

    # Valores nulos
    st.subheader("Analisis de valores nulos")
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df) * 100).round(2)
    df_nulos = pd.DataFrame({
        "Columna": nulos.index,
        "Valores Nulos": nulos.values,
        "Porcentaje (%)": porcentaje_nulos.values
    })
    df_nulos = df_nulos[df_nulos["Valores Nulos"] > 0]

    if df_nulos.empty:
        st.success("No se encontraron valores nulos en el dataset.")
    else:
        st.warning(f"Se encontraron {df_nulos['Valores Nulos'].sum()} valores nulos.")
        st.dataframe(df_nulos, use_container_width=True)
        fig_nulos = px.bar(
            df_nulos, x="Columna", y="Porcentaje (%)",
            title="Porcentaje de valores nulos por columna",
            color="Porcentaje (%)", color_continuous_scale="reds"
        )
        st.plotly_chart(fig_nulos, use_container_width=True)

    st.markdown("---")

    # Duplicados
    st.subheader("Deteccion de duplicados")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        st.success("No se encontraron registros duplicados.")
    else:
        st.warning(f"Se encontraron {n_dup} registros duplicados.")

    st.markdown("---")

    # Estadistica descriptiva
    st.subheader("Estadistica descriptiva")
    if st.checkbox("Mostrar estadisticas descriptivas de variables numericas"):
        st.dataframe(df[numericas].describe().round(2), use_container_width=True)

    st.markdown("---")

    # Outliers
    st.subheader("Deteccion de outliers - Regla IQR")
    col_outlier = st.selectbox("Selecciona una variable numerica:", numericas)

    if col_outlier:
        outliers, lim_inf, lim_sup = detectar_outliers_iqr(df, col_outlier)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Limite Inferior", f"{lim_inf:.2f}")
            st.metric("Limite Superior", f"{lim_sup:.2f}")
        with col_b:
            st.metric("Total Outliers", len(outliers))
            st.metric("Porcentaje del dataset", f"{(len(outliers)/len(df)*100):.1f}%")

        fig_box, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df[col_outlier], color="#4C9BE8", ax=ax)
        ax.set_title(f"Boxplot - {col_outlier}", fontsize=13)
        ax.set_xlabel(col_outlier)
        st.pyplot(fig_box)

    st.markdown("---")

    # Filtros dinamicos
    st.subheader("Filtros dinamicos")
    col_filtro = st.selectbox("Filtrar por variable categorica:", categoricas)
    if col_filtro:
        opciones = df[col_filtro].unique().tolist()
        seleccion = st.multiselect(f"Valores de {col_filtro}:", opciones, default=opciones)
        df_filtrado = df[df[col_filtro].isin(seleccion)]
        st.write(f"Registros filtrados: **{len(df_filtrado)}** de {len(df)}")
        if st.checkbox("Ver datos filtrados"):
            st.dataframe(df_filtrado.head(20), use_container_width=True)

# -------------------------------------------------------
# SECCION 4: ANALISIS VISUAL
# -------------------------------------------------------

elif seccion == "Analisis Visual":
    st.title("Analisis Visual")
    st.markdown("---")

    if "df" not in st.session_state:
        st.error("No hay datos cargados. Ve a 'Carga y Perfil del Dataset' primero.")
        st.stop()

    df = st.session_state.df
    numericas, categoricas, binarias = detectar_tipos(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Resumen",
        "Univariado",
        "Bivariado",
        "Multivariado",
        "Temporal",
        "Insights"
    ])

    # --- TAB 1: RESUMEN ---
    with tab1:
        st.subheader("Resumen general del dataset")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total filas", df.shape[0])
        col2.metric("Total columnas", df.shape[1])
        col3.metric("Variables numericas", len(numericas))
        col4.metric("Variables categoricas", len(categoricas))

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Tipos de datos:**")
            tipos_df = pd.DataFrame({"Tipo": df.dtypes.astype(str)})
            st.dataframe(tipos_df, use_container_width=True)
        with col_b:
            st.markdown("**Nulos y duplicados:**")
            resumen = pd.DataFrame({
                "Metrica": ["Valores Nulos", "Duplicados"],
                "Cantidad": [df.isnull().sum().sum(), df.duplicated().sum()]
            })
            st.dataframe(resumen, use_container_width=True)

        st.markdown("---")
        if st.checkbox("Mostrar estadistico completo"):
            st.dataframe(df[numericas].describe().round(2), use_container_width=True)

    # --- TAB 2: UNIVARIADO ---
    with tab2:
        st.subheader("Analisis Univariado")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribucion de variables numericas")
            var_num = st.selectbox("Variable numerica:", numericas, key="uni_num")
            bins_val = st.slider("Numero de bins:", 5, 50, 20, key="bins_uni")

            fig_hist = px.histogram(
                df, x=var_num, nbins=bins_val,
                title=f"Histograma - {var_num}",
                color_discrete_sequence=["#4C9BE8"],
                marginal="box"
            )
            fig_hist.update_layout(bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(f"La distribucion de {var_num} muestra como se concentran los valores en los adolescentes del dataset.")

        with col2:
            st.markdown("#### Variables categoricas")
            var_cat = st.selectbox("Variable categorica:", categoricas, key="uni_cat")
            conteo = df[var_cat].value_counts().reset_index()
            conteo.columns = [var_cat, "Conteo"]

            fig_bar = px.bar(
                conteo, x=var_cat, y="Conteo",
                title=f"Conteo por {var_cat}",
                color=var_cat,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(f"Distribucion de categorias en la variable {var_cat}.")

        st.markdown("---")

        # Boxplot con Seaborn
        st.markdown("#### Boxplot por genero (Seaborn)")
        var_box = st.selectbox("Variable para boxplot:", numericas, key="box_gen")
        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        sns.set_style("whitegrid")
        sns.boxplot(
            data=df, x="gender", y=var_box,
            palette=["#4C9BE8", "#FF6B9D"], ax=ax_s
        )
        ax_s.set_title(f"{var_box} por genero", fontsize=13)
        ax_s.set_xlabel("Genero")
        ax_s.set_ylabel(var_box)
        st.pyplot(fig_s)
        st.caption(f"Comparacion de {var_box} entre generos.")

        st.markdown("---")

        # Grafico circular
        st.markdown("#### Proporcion por plataforma (Matplotlib)")
        platform_counts = df["platform_usage"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))
        colores = ["#4C9BE8", "#FF6B9D", "#FFD93D"]
        ax_pie.pie(
            platform_counts.values,
            labels=platform_counts.index,
            autopct='%1.1f%%',
            colors=colores,
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        ax_pie.set_title("Distribucion por plataforma de redes sociales", fontsize=13)
        st.pyplot(fig_pie)
        st.caption("Las tres plataformas tienen proporciones similares entre los adolescentes del dataset.")

    # --- TAB 3: BIVARIADO ---
    with tab3:
        st.subheader("Analisis Bivariado")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Scatter plot - variables numericas")
            var_x = st.selectbox("Variable X:", numericas, key="biv_x", index=0)
            var_y = st.selectbox("Variable Y:", numericas, key="biv_y", index=2)
            color_by = st.selectbox("Color por:", categoricas, key="biv_color")

            fig_scatter = px.scatter(
                df, x=var_x, y=var_y, color=color_by,
                title=f"{var_x} vs {var_y}",
                color_discrete_sequence=px.colors.qualitative.Set2,
                opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(f"Relacion entre {var_x} y {var_y} segmentada por {color_by}.")

        with col2:
            st.markdown("#### Boxplot por categoria")
            cat_biv = st.selectbox("Variable categorica:", categoricas, key="biv_cat")
            num_biv = st.selectbox("Variable numerica:", numericas, key="biv_num")

            fig_box2 = px.box(
                df, x=cat_biv, y=num_biv,
                color=cat_biv,
                title=f"{num_biv} por {cat_biv}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_box2, use_container_width=True)
            st.caption(f"Comparacion de {num_biv} entre las categorias de {cat_biv}.")

        st.markdown("---")

        st.markdown("#### Barras agrupadas - depresion por plataforma")
        dep_platform = df.groupby(["platform_usage", "depression_label"]).size().reset_index(name="conteo")
        dep_platform["depression_label"] = dep_platform["depression_label"].map(
            {0: "Sin depresion", 1: "Con depresion"}
        )
        fig_grouped = px.bar(
            dep_platform, x="platform_usage", y="conteo",
            color="depression_label", barmode="group",
            title="Casos de depresion por plataforma",
            color_discrete_sequence=["#4C9BE8", "#FF6B9D"]
        )
        st.plotly_chart(fig_grouped, use_container_width=True)
        st.caption("Explora si existe asociacion entre la plataforma usada y la etiqueta de depresion.")

    # --- TAB 4: MULTIVARIADO ---
    with tab4:
        st.subheader("Analisis Multivariado")

        # Heatmap de correlacion
        st.markdown("#### Mapa de calor de correlacion (Seaborn)")
        vars_corr = st.multiselect(
            "Selecciona variables para la correlacion:",
            numericas, default=numericas
        )

        if len(vars_corr) >= 2:
            corr = df[vars_corr].corr().round(2)
            fig_heat, ax_heat = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                corr, annot=True, cmap="coolwarm", center=0,
                fmt=".2f", linewidths=0.5, ax=ax_heat,
                annot_kws={"size": 10}
            )
            ax_heat.set_title("Matriz de correlacion - Variables numericas", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig_heat)
            st.caption("Los valores cercanos a 1 o -1 indican correlaciones fuertes. La diagonal siempre es 1.")
        else:
            st.warning("Selecciona al menos 2 variables.")

        st.markdown("---")

        # Barras apiladas
        st.markdown("#### Barras apiladas - estres por plataforma e interaccion social")
        pivot_data = df.groupby(
            ["platform_usage", "social_interaction_level"]
        )["stress_level"].mean().reset_index()

        fig_stack = px.bar(
            pivot_data, x="platform_usage", y="stress_level",
            color="social_interaction_level", barmode="stack",
            title="Nivel de estres promedio por plataforma e interaccion social",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_stack, use_container_width=True)
        st.caption("Visualizacion que integra plataforma, nivel de interaccion y estres promedio.")

        st.markdown("---")

        # Scatter con 3 variables
        st.markdown("#### Scatter con tres variables")
        v1 = st.selectbox("Variable X:", numericas, key="multi_x", index=0)
        v2 = st.selectbox("Variable Y:", numericas, key="multi_y", index=2)
        v3 = st.selectbox("Tamano por:", numericas, key="multi_size", index=4)
        cat_color = st.selectbox("Color por:", categoricas, key="multi_color")

        fig_multi = px.scatter(
            df, x=v1, y=v2, size=v3,
            color=cat_color,
            title=f"{v1} vs {v2} (tamano: {v3})",
            color_discrete_sequence=px.colors.qualitative.Bold,
            size_max=15, opacity=0.7
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    # --- TAB 5: TEMPORAL ---
    with tab5:
        st.subheader("Analisis por edad")
        st.info("Este dataset no tiene fechas. Se usa la variable edad como eje de analisis evolutivo.")

        st.markdown("#### Evolucion de metricas por edad")
        var_temporal = st.selectbox(
            "Variable a analizar:", numericas, key="temp_var",
            index=numericas.index("stress_level") if "stress_level" in numericas else 0
        )

        df_age = df.groupby("age")[var_temporal].mean().reset_index()
        fig_line = px.line(
            df_age, x="age", y=var_temporal,
            title=f"Promedio de {var_temporal} por edad",
            markers=True,
            color_discrete_sequence=["#4C9BE8"]
        )
        fig_line.update_traces(line_width=2.5, marker_size=8)
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption(f"Muestra como evoluciona el promedio de {var_temporal} a medida que aumenta la edad.")

        st.markdown("---")

        st.markdown("#### Comparacion de variables por edad (Matplotlib)")
        vars_age = st.multiselect(
            "Selecciona variables:",
            numericas,
            default=["stress_level", "anxiety_level", "sleep_hours"]
        )

        if vars_age:
            df_multi_age = df.groupby("age")[vars_age].mean()
            fig_ma, ax_ma = plt.subplots(figsize=(10, 5))
            for col in vars_age:
                ax_ma.plot(df_multi_age.index, df_multi_age[col], marker='o', label=col, linewidth=2)
            ax_ma.set_title("Evolucion de variables por edad", fontsize=13)
            ax_ma.set_xlabel("Edad")
            ax_ma.set_ylabel("Valor promedio")
            ax_ma.legend()
            ax_ma.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_ma)
            st.caption("Comparacion de multiples variables por edad usando Matplotlib.")

    # --- TAB 6: INSIGHTS ---
    with tab6:
        st.subheader("Hallazgos clave")

        col1, col2, col3 = st.columns(3)
        col1.metric("Promedio horas en redes", f"{df['daily_social_media_hours'].mean():.1f} hrs/dia")
        col2.metric("Promedio horas de sueno", f"{df['sleep_hours'].mean():.1f} hrs")
        col3.metric(
            "Casos con depresion",
            f"{df['depression_label'].sum()} ({df['depression_label'].mean()*100:.1f}%)"
        )

        st.markdown("---")

        st.markdown("#### Insight 1 - Uso de redes y estres")
        fig_i1 = px.scatter(
            df, x="daily_social_media_hours", y="stress_level",
            color="gender", trendline="ols",
            title="Horas en redes sociales vs nivel de estres",
            color_discrete_sequence=["#4C9BE8", "#FF6B9D"],
            opacity=0.6
        )
        st.plotly_chart(fig_i1, use_container_width=True)
        st.markdown(
            "Se observa una tendencia entre el uso de redes sociales y el nivel de estres "
            "en los adolescentes. Mayor tiempo en redes se asocia con mayor estres."
        )

        st.markdown("---")

        st.markdown("#### Insight 2 - Sueno e interaccion social")
        fig_i2 = px.box(
            df, x="social_interaction_level", y="sleep_hours",
            color="social_interaction_level",
            title="Horas de sueno por nivel de interaccion social",
            category_orders={"social_interaction_level": ["low", "medium", "high"]},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_i2, use_container_width=True)
        st.markdown(
            "Los adolescentes con mayor interaccion social tienden a tener patrones de sueno "
            "diferentes. El analisis es exploratorio y no implica causalidad."
        )

        st.markdown("---")

        st.markdown("#### Insight 3 - Depresion por plataforma y genero")
        dep_gen = df.groupby(["platform_usage", "gender"])["depression_label"].mean().reset_index()
        dep_gen["Tasa depresion (%)"] = (dep_gen["depression_label"] * 100).round(1)
        fig_i3 = px.bar(
            dep_gen, x="platform_usage", y="Tasa depresion (%)",
            color="gender", barmode="group",
            title="Tasa de depresion por plataforma y genero",
            color_discrete_sequence=["#4C9BE8", "#FF6B9D"]
        )
        st.plotly_chart(fig_i3, use_container_width=True)
        st.markdown(
            "La tasa de depresion varia segun la plataforma y el genero. Estos patrones son "
            "exploratorios y no constituyen un diagnostico clinico."
        )

        st.markdown("---")
        st.success("Analisis exploratorio completado. Los hallazgos son de caracter descriptivo.")
