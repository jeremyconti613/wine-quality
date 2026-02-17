import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

from src.preprocessing import (
    load_data,
    data_quality_report,
    missing_summary,
    detect_outliers_iqr,
    detect_outliers_zscore,
    prepare_features,
)


st.set_page_config(
    page_title="Analyse de la Qualité du Vin",
    page_icon="🍷",
    layout="wide",
)

# Palette fixe pour que le vin rouge soit en rouge
WINE_PALETTE = {
    "red": "#d62728",    # rouge
    "white": "#ffdd8e",  # blanc légèrement doré
}


@st.cache_data
def load_combined_data() -> pd.DataFrame:
    """Charge et combine les données vin rouge / vin blanc."""
    root = Path(__file__).resolve().parent
    red_path = root / "data" / "winequality-red.csv"
    white_path = root / "data" / "winequality-white.csv"
    df = load_data(red_path, white_path)
    return df


@st.cache_data
def get_quality_reports(df: pd.DataFrame):
    """Prépare les rapports de qualité de données et outliers."""
    report = data_quality_report(df)
    missing_df = missing_summary(df)
    iqr_outliers = detect_outliers_iqr(df)
    z_outliers = detect_outliers_zscore(df)
    return report, missing_df, iqr_outliers, z_outliers


@st.cache_data
def get_prepared_features(df: pd.DataFrame, threshold: int = 7):
    X, y, feature_cols = prepare_features(df, quality_threshold=threshold)
    return X, y, feature_cols


def build_regression_features(df: pd.DataFrame):
    """Prépare les features pour la régression de la note exacte.

    On ne conserve que les variables :
    - density
    - residual sugar
    - alcohol
    - volatile acidity
    """
    df_reg = df.copy()
    # Optionnellement, on pourrait encoder le type de vin, mais ici on ne garde
    # que les caractéristiques physicochimiques indiquées.
    candidate_features = [
        "density",
        "residual sugar",
        "alcohol",
        "volatile acidity",
    ]
    feature_cols = [c for c in candidate_features if c in df_reg.columns]
    X = df_reg[feature_cols]
    y = df_reg["quality"]
    return X, y, feature_cols


def main():
    df = load_combined_data()
    report, missing_df, iqr_outliers, z_outliers = get_quality_reports(df)

    st.title("Analyse de la Qualité du Vin")
    st.markdown(
        """
        Application Streamlit inspirée du notebook d'analyse exploratoire des vins **Vinho Verde**
        (rouge et blanc).  
        L'objectif est de **comprendre les données**, **préparer les futures analyses**
        et **mettre en évidence les relations entre caractéristiques physicochimiques et qualité**.
        """
    )

    # Navigation principale
    section = st.sidebar.radio(
        "Navigation",
        (
            "1. Contexte & Données",
            "2. Préparation des données",
            "3. Qualité des données",
            "4. Visualisations exploratoires",
            "5. Relations & hypothèses",
            "6. Régression (note exacte)",
            "7. Classification (régression logistique)",
            "8. Interprétation & limites",
        ),
    )

    st.sidebar.markdown("### Paramètres")
    wine_filter = st.sidebar.multiselect(
        "Type de vin",
        options=sorted(df["wine_type"].unique()),
        default=list(sorted(df["wine_type"].unique())),
    )
    df_filtered = df[df["wine_type"].isin(wine_filter)].copy()

    if section == "1. Contexte & Données":
        show_context_and_data(df_filtered, df)
    elif section == "2. Préparation des données":
        show_preparation(df_filtered)
    elif section == "3. Qualité des données":
        show_data_quality(df_filtered, report, missing_df, iqr_outliers, z_outliers)
    elif section == "4. Visualisations exploratoires":
        show_visualisations(df_filtered)
    elif section == "5. Relations & hypothèses":
        show_relations_and_hypotheses(df_filtered)
    elif section == "6. Régression (note exacte)":
        show_regression(df_filtered)
    elif section == "7. Classification (régression logistique)":
        show_logistic_regression(df_filtered)
    elif section == "8. Interprétation & limites":
        show_conclusion(df_filtered)


def show_context_and_data(df: pd.DataFrame, df_full: pd.DataFrame):
    st.header("1. Compréhension des données")

    st.subheader("1.1 Contexte métier & problématique")
    st.markdown(
        """
        - **Domaine** : analyse sensorielle et œnologie sur les vins portugais *Vinho Verde*.  
        - **Problématique** : prédire la **qualité perçue** d'un vin à partir de ses
          **mesures physicochimiques** (acidité, teneur en sucre, alcool, etc.).  
        - **Variable cible** : `quality`, note de 0 à 10 issue de dégustations d'experts.
        """
    )

    st.subheader("1.2 Structure des jeux de données")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nb. observations (total)", f"{len(df_full):,}".replace(",", " "))
    with col2:
        st.metric("Nb. variables", df_full.shape[1])
    with col3:
        st.metric("Types", "Toutes numériques + type de vin")

    st.markdown("#### Aperçu des données filtrées")
    st.dataframe(df.head())

    st.markdown("#### Description des variables")
    desc = {
        "fixed acidity": "Acidité fixe (acides non volatils)",
        "volatile acidity": "Acidité volatile (acides responsables du goût de vinaigre)",
        "citric acid": "Acide citrique (fraîcheur, acidité vive)",
        "residual sugar": "Sucre résiduel après fermentation",
        "chlorides": "Chlorures (teneur en sel)",
        "free sulfur dioxide": "SO₂ libre (agent conservateur)",
        "total sulfur dioxide": "SO₂ total",
        "density": "Densité du vin",
        "pH": "Acidité globale (échelle 0-14)",
        "sulphates": "Sulfates (protection antimicrobienne)",
        "alcohol": "Teneur en alcool (%)",
        "quality": "Qualité sensorielle (0 = mauvais, 10 = excellent)",
        "wine_type": "Type de vin (rouge / blanc)",
    }
    desc_df = pd.DataFrame(
        [{"variable": k, "description": v} for k, v in desc.items()]
    )
    st.table(desc_df)


def show_preparation(df: pd.DataFrame):
    st.header("2. Préparer les données pour les analyses")

    st.markdown(
        """
        Dans le notebook, la préparation vise principalement à :  
        - **Créer une cible binaire** : distinguer les vins *bons* (qualité ≥ seuil) des autres.  
        - **Encoder le type de vin** (`wine_type`) en variables numériques.  
        - **Standardiser** les variables numériques.
        """
    )

    threshold = st.slider(
        "Seuil de qualité pour considérer un vin comme « bon » (quality ≥ seuil)",
        min_value=int(df["quality"].min()),
        max_value=int(df["quality"].max()),
        value=6,
        step=1,
    )

    X, y, feature_cols = get_prepared_features(df, threshold=threshold)

    st.subheader("2.1 Nouvelle variable cible")
    st.markdown(
        f"""
        - Une nouvelle variable `quality_label` est définie :  
          - 1 → vin **bon** (quality ≥ {threshold})  
          - 0 → vin **standard ou médiocre** (quality < {threshold})  
        - Cette transformation permet d'aborder le problème en **classification binaire**.
        """
    )

    st.write("Répartition de `quality` et de `quality_label` :")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df["quality"].value_counts().sort_index())
    with col2:
        label_counts = y.value_counts().rename(index={0: "0 (non-bon)", 1: "1 (bon)"})
        st.bar_chart(label_counts)

    st.subheader("2.2 Matrice de caractéristiques après préparation")
    st.markdown(
        """
        - Les colonnes incluent les mesures physicochimiques et un encodage du type de vin.  
        - Cette matrice est prête à être **scalée** puis utilisée dans des modèles
          (régression, arbres, SVM, etc.).
        """
    )
    st.write("Aperçu des features (X) :")
    st.dataframe(X.head())

    st.markdown(
        """
        **Justification des choix de préparation** :  
        - La cible binaire facilite l'interprétation métier (*bons vs autres vins*).  
        - L'encodage `wine_type` permet de capturer les différences structurelles rouge/blanc.  
        - Le scaling (dans le pipeline complet) est adapté aux modèles sensibles à l'échelle
          des variables (SVM, régression logistique, etc.).
        """
    )


def show_data_quality(
    df: pd.DataFrame,
    report: dict,
    missing_df: pd.DataFrame,
    iqr_outliers: pd.DataFrame,
    z_outliers: pd.DataFrame,
):
    st.header("3. Vérification de la qualité des données")

    st.subheader("3.1 Valeurs manquantes")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Nb. lignes", report["rows"])
    with c2:
        st.metric("Nb. colonnes", report["cols"])
    with c3:
        st.metric("Valeurs manquantes (total)", report["missing_total"])

    st.markdown(
        """
        - Le dataset d'origine ne contient **aucune valeur manquante** (comme vérifié dans le notebook).  
        - Les éventuels traitements de valeurs manquantes ne sont donc **pas nécessaires** ici.
        """
    )

    st.subheader("3.2 Types de données")
    st.write(pd.Series(report["dtypes"], name="dtype").to_frame())

    st.subheader("3.3 Détection des outliers")
    st.markdown(
        """
        Nous utilisons deux approches complémentaires :  
        - **Règle de l'IQR (Interquartile Range)** : points situés en dehors \[Q1 − 1.5×IQR ; Q3 + 1.5×IQR\].  
        - **Z-score** : points dont la distance à la moyenne dépasse un certain seuil (ici 3 écarts-types).
        """
    )

    tabs = st.tabs(["Résumé IQR", "Résumé Z-score", "Boxplots"])
    with tabs[0]:
        st.write("Outliers par variable (IQR) :")
        st.dataframe(iqr_outliers)
    with tabs[1]:
        st.write("Outliers par variable (Z-score) :")
        st.dataframe(z_outliers)
    with tabs[2]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Variables à afficher en boxplot",
            options=numeric_cols,
            default=["alcohol", "residual sugar", "chlorides"],
        )
        if selected_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df[selected_cols], ax=ax)
            ax.set_title("Boxplots des variables sélectionnées")
            st.pyplot(fig)

    st.markdown(
        """
        **Gestion des outliers (stratégie adoptée)** :  
        - Les outliers reflètent souvent des **cas réels extrêmes** (vins très sucrés, très acides, etc.).  
        - Plutôt que de les supprimer systématiquement, la stratégie recommandée est de :  
          - Les **analyser** (impact sur les modèles, stabilité des coefficients).  
          - Éventuellement les **caper** (clipper) si l'on observe une sensibilité excessive de certains modèles.
        """
    )


def show_visualisations(df: pd.DataFrame):
    st.header("4. Visualisations exploratoires")

    st.subheader("4.1 Distribution de la qualité")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x="quality", data=df, hue="wine_type", ax=ax, palette=WINE_PALETTE)
        ax.set_title("Distribution de la qualité par type de vin")
        st.pyplot(fig)
    with col2:
        st.markdown(
            """
            - La majorité des vins se situe entre **5 et 7**.  
            - Très peu de vins sont notés comme **exceptionnels** (8–9) ou **très mauvais** (≤4).  
            - Cette **asymétrie** justifie de traiter la qualité comme une variable **ordinale/déséquilibrée**.
            """
        )

    st.subheader("4.2 Distributions univariées")
    feature = st.selectbox(
        "Choisir une variable numérique à explorer",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        index=10,
    )
    fig, ax = plt.subplots()
    sns.histplot(
        df,
        x=feature,
        hue="wine_type",
        kde=True,
        ax=ax,
        element="step",
        palette=WINE_PALETTE,
    )
    ax.set_title(f"Distribution de {feature} par type de vin")
    st.pyplot(fig)

    st.subheader("4.3 Corrélation globale")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Matrice de corrélation (variables numériques)")
    st.pyplot(fig)

    st.markdown(
        """
        **Exemples de tendances visibles (issues du notebook)** :  
        - L'**alcool** est généralement **positivement corrélé** à la qualité.  
        - Une **acidité volatile élevée** (goût vinaigré) tend à être **négativement corrélée** à la qualité.  
        - Certaines variables sont fortement corrélées entre elles (ex. SO₂ libre / total), ce qui
          invite à une **sélection de variables** ou à des méthodes robustes à la colinéarité.
        """
    )


def show_relations_and_hypotheses(df: pd.DataFrame):
    st.header("5. Relations entre variables & hypothèses")

    st.subheader("5.1 Relations qualité vs caractéristiques clés")
    x_var = st.selectbox(
        "Variable explicative",
        options=[
            "alcohol",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "sulphates",
            "pH",
            "density",
        ],
        index=0,
    )

    fig, ax = plt.subplots()
    sns.boxplot(
        x="quality",
        y=x_var,
        data=df,
        hue="wine_type",
        ax=ax,
        palette=WINE_PALETTE,
    )
    ax.set_title(f"{x_var} en fonction de la qualité")
    st.pyplot(fig)

    st.markdown(
        """
        **Exemple d'interprétation** (à adapter selon la variable choisie) :  
        - `alcohol` : les vins les mieux notés ont en moyenne une **teneur en alcool plus élevée**.  
        - `volatile acidity` : les vins de mauvaise qualité présentent souvent une **acidité volatile plus forte** pour les **vins Rouges**.
        - `residual sugar` : le sucre résiduel peut différencier certains styles de vins blancs.
        """
    )

    st.subheader("5.2 Hypothèses de travail")
    st.markdown(
        """
        À partir des observations exploratoires, on peut formuler plusieurs hypothèses :  
        - **H1** : plus l'alcool est élevé (dans des limites raisonnables), plus la qualité perçue augmente.  
        - **H2** : une acidité volatile trop forte dégrade la perception de qualité des vins Rouges.  
        - **H3** : le type de vin (rouge vs blanc) module l'effet de certaines variables sur la qualité.  

        **Choix de modèles possibles** (comme discuté dans le notebook) :  
        - **Régression** (linéaire, régularisée) pour prédire la note exacte.  
        - **Classification** (logistique, arbres, Random Forest, SVM) pour prédire *bon* vs *non bon*.  
        - Les modèles à marge large comme les **SVM** sont bien adaptés à ce type de donnée
          numériquement homogène et ont montré de bonnes performances dans la littérature.
        """
    )


def show_regression(df: pd.DataFrame):
    st.header("6. Régression linéaire pour prédire la note exacte")

    st.markdown(
        """
        Nous entraînons ici un **modèle de régression linéaire simple** pour prédire
        directement la note de qualité `quality` (0–10) à partir d’un sous-ensemble
        de caractéristiques **choisies par l'utilisateur** parmi les variables
        physicochimiques disponibles.
        """
    )

    # Choix interactif des caractéristiques d'entrée
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "quality"
    ]

    selected_features = st.multiselect(
        "Caractéristiques utilisées comme entrées du modèle de régression",
        options=numeric_cols,
        default=numeric_cols,
    )

    if not selected_features:
        st.warning("Veuillez sélectionner au moins une caractéristique pour entraîner le modèle.")
        return

    # Préparation des données pour la régression en fonction des caractéristiques choisies
    X = df[selected_features]
    y = df["quality"]
    feature_cols = selected_features

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle de régression linéaire (sans régularisation explicite)
    model = LinearRegression()

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    st.subheader("6.1 Performances du modèle sur le jeu de test")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("MAE", f"{mae:.3f}")
    with c2:
        st.metric("RMSE", f"{rmse:.3f}")
    with c3:
        st.metric("R²", f"{r2:.3f}")

    st.markdown(
        """
        - **MAE** : erreur absolue moyenne en points de qualité.  
        - **RMSE** : pénalise davantage les grandes erreurs.  
        - **R²** : proportion de la variance de `quality` expliquée par le modèle.
        """
    )

    st.subheader("6.2 Analyse des résidus")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title("Distribution des résidus (y_réel - y_prédit)")
        ax.set_xlabel("Résidu")
        st.pyplot(fig)
    with col_r2:
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Qualité prédite")
        ax.set_ylabel("Résidu")
        ax.set_title("Résidus en fonction des prédictions")
        st.pyplot(fig)

    st.markdown(
        f"""
        - Résidus **moyens** : {residuals.mean():.3f} (devrait être proche de 0 si le modèle est bien calibré).  
        - Résidus **écart-type** : {residuals.std():.3f}.  
        - On recherche visuellement une **dispersion aléatoire** autour de 0 (pas de structure évidente),
          signe que les hypothèses du modèle linéaire sont raisonnables.
        """
    )

    st.subheader("6.3 Prédiction vs valeur réelle")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    min_q = min(y_test.min(), y_pred.min())
    max_q = max(y_test.max(), y_pred.max())
    ax.plot([min_q, max_q], [min_q, max_q], "k--", label="y = x")
    ax.set_xlabel("Qualité réelle")
    ax.set_ylabel("Qualité prédite")
    ax.set_title("Qualité réelle vs prédite (jeu de test)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("6.4 Poids des variables (coefficients)")
    if hasattr(model, "coef_"):
        coef_series = pd.Series(model.coef_, index=feature_cols)
        coef_sorted = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
        st.write("Top variables (en valeur absolue des coefficients) :")
        st.dataframe(coef_sorted.to_frame("coefficient"))

        fig, ax = plt.subplots(figsize=(8, 5))
        coef_sorted.head(15).plot(kind="barh", ax=ax)
        ax.set_title("Principales variables explicatives selon le modèle")
        ax.invert_yaxis()
        st.pyplot(fig)

    st.markdown(
        """
        **Interprétation** :  
        - Les coefficients indiquent comment la note de qualité varie en moyenne lorsqu'une
          variable augmente d'une unité (toutes choses égales par ailleurs).  
        """
    )


def show_logistic_regression(df: pd.DataFrame):
    st.header("7. Classification avec régression logistique")

    st.markdown(
        """
        Nous entraînons ici un modèle de **régression logistique** pour prédire si un vin
        est **bon** (1) ou **non bon** (0) à partir de ses caractéristiques.
        La cible binaire `quality_label` est définie à partir de la note `quality`.
        """
    )

    # Choix du seuil pour définir un "bon" vin
    threshold = st.slider(
        "Seuil de qualité pour considérer un vin comme « bon » (quality ≥ seuil)",
        min_value=int(df["quality"].min()),
        max_value=int(df["quality"].max()),
        value=6,
        step=1,
    )

    # Préparation des features binaires via la fonction existante
    X_full, y, feature_cols_full = get_prepared_features(df, threshold=threshold)

    st.markdown(
        f"Classe positive : vins avec `quality` ≥ **{threshold}** (label = 1)."
    )

    # Choix interactif des caractéristiques d'entrée
    selected_features = st.multiselect(
        "Caractéristiques utilisées comme entrées du modèle logistique",
        options=feature_cols_full,
        default=feature_cols_full,
    )

    if not selected_features:
        st.warning("Veuillez sélectionner au moins une caractéristique pour entraîner le modèle.")
        return

    X = X_full[selected_features]

    # Split train / test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle de régression logistique
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train_scaled, y_train)

    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    st.subheader("7.1 Performances du modèle sur le jeu de test")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{acc:.3f}")
    with c2:
        st.metric("ROC AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

    st.subheader("7.2 Matrice de confusion")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Réel 0", "Réel 1"],
        columns=["Prédit 0", "Prédit 1"],
    )
    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Matrice de confusion (0 = non bon, 1 = bon)")
    st.pyplot(fig)

    st.subheader("7.3 Importance des variables (coefficients)")
    coef_series = pd.Series(clf.coef_[0], index=selected_features)
    coef_sorted = coef_series.reindex(
        coef_series.abs().sort_values(ascending=False).index
    )
    st.write("Coefficients triés par importance absolue :")
    st.dataframe(coef_sorted.to_frame("coefficient"))

    fig, ax = plt.subplots(figsize=(8, 5))
    coef_sorted.head(15).plot(kind="barh", ax=ax)
    ax.set_title("Principales variables explicatives (régression logistique)")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.markdown(
        """
        **Interprétation** :  
        - Un coefficient positif indique qu'une augmentation de la variable **augmente**
          la probabilité qu'un vin soit classé comme **bon** (label = 1).  
        - Un coefficient négatif indique l'effet inverse.  
        - La régression logistique fournit donc un compromis intéressant entre
          **performance prédictive** et **interprétabilité** des effets des variables.
        """
    )


def show_conclusion(df: pd.DataFrame):
    st.header("8. Interprétation globale & limites")

    st.subheader("8.1 Synthèse des principaux résultats exploratoires")
    st.markdown(
        """
        - Les jeux de données (rouge et blanc) sont **propres**, sans valeurs manquantes,
          et bien documentés.  
        - La qualité des vins est **modérément corrélée** avec certaines variables clés
          (alcool, acidité volatile, sulfates, etc.).  
        - La distribution de `quality` est **déséquilibrée**, avec peu d'extrêmes.
        """
    )

    st.subheader("8.2 Interprétation statistique & significativité (niveau exploratoire)")
    st.markdown(
        """
        - Les corrélations observées servent de **pistes** mais ne suffisent pas à établir
          une **causalité**.  
        - Des tests plus formels (tests de corrélation, modèles paramétriques) peuvent être
          intégrés dans un second temps pour quantifier la **significativité**.  
        - La granularité de la note (0–10) et la subjectivité du jugement humain imposent
          une certaine **incertitude** sur la cible.
        """
    )

    st.subheader("8.3 Limitations")
    st.markdown(
        """
        - Absence d'informations sur le **prix**, la **marque**, le **millésime** ou la **région précise**.  
        - Les données proviennent d'une seule appellation (*Vinho Verde*), ce qui limite
          la **généralisation** à d'autres types de vins.  
        - La qualité est une mesure **subjective**, même si elle repose sur plusieurs experts.
        """
    )

    # st.subheader("8.4 Perspectives")
    # st.markdown(
    #     """
    #     - Intégrer des **modèles prédictifs** (SVM, Random Forest, Gradient Boosting) dans
    #       cette application pour comparer leurs performances.  
    #     - Explorer des approches de **sélection de variables** pour réduire la dimension
    #       et améliorer l'interprétabilité.  
    #     - Étendre l'analyse à d'autres datasets de vins afin de tester la **robustesse**
    #       des conclusions actuelles.
    #     """
    # )

    # st.info(
    #     "Cette application Streamlit résume le notebook en une présentation structurée : "
    #     "compréhension des données, qualité, préparation, exploration visuelle, "
    #     "formulation d'hypothèses et conclusions."
    # )


if __name__ == "__main__":
    main()
