"""
Optimización de hiperparámetros del Random Forest sobre un conjunto de
train/test reducido con PCA (95% de la varianza retenida).

- Misma división train/test que random_forest_wisc_bc.py (75/25,
  estratificada, semilla 42).
- Las variables se estandarizan antes del PCA (tienen escalas muy
  distintas, p. ej. area_mean ~ 650 vs smoothness_mean ~ 0.1); el
  escalador y el PCA se ajustan SOLO con el train y luego se aplican al test.
- Se guardan los conjuntos reducidos en resultados/train_pca.csv y
  resultados/test_pca.csv.
- La búsqueda de hiperparámetros usa un Pipeline (escalado + PCA + RF)
  para que en cada fold de la validación cruzada el PCA se ajuste solo
  con los datos de entrenamiento de ese fold.

Uso:
    pip install pandas scikit-learn matplotlib scipy
    python optimizacion_hiperparametros.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV,
                                     RepeatedStratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

URL = ("https://raw.githubusercontent.com/stedy/"
       "Machine-Learning-with-R-datasets/master/wisc_bc_data.csv")
RUTA_LOCAL = "wisc_bc_data.csv"
SEMILLA = 42
VARIANZA = 0.95
N_ITER = 60
SALIDA = "resultados"

# ------------------------------------------------------------------
# 1. Datos: misma división que el modelo anterior
# ------------------------------------------------------------------
df = pd.read_csv(RUTA_LOCAL if os.path.exists(RUTA_LOCAL) else URL)
os.makedirs(SALIDA, exist_ok=True)
X = df.drop(columns=["id", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEMILLA)

# ------------------------------------------------------------------
# 2. Construcción de train/test con PCA (95% de varianza)
# ------------------------------------------------------------------
escalador = StandardScaler().fit(X_train)
pca = PCA(n_components=VARIANZA, svd_solver="full").fit(
    escalador.transform(X_train))
n_comp = pca.n_components_
nombres = [f"PC{i + 1}" for i in range(n_comp)]

train_pca = pd.DataFrame(pca.transform(escalador.transform(X_train)),
                         columns=nombres, index=X_train.index)
test_pca = pd.DataFrame(pca.transform(escalador.transform(X_test)),
                        columns=nombres, index=X_test.index)
train_pca.assign(diagnosis=y_train).to_csv(f"{SALIDA}/train_pca.csv")
test_pca.assign(diagnosis=y_test).to_csv(f"{SALIDA}/test_pca.csv")

var = pca.explained_variance_ratio_
print("=== PCA (ajustado solo con el train) ===")
print(f"Variables originales: {X.shape[1]}  ->  componentes: {n_comp}")
print(f"Varianza retenida: {var.sum():.4f}")
print(pd.DataFrame({"varianza": var.round(4),
                    "acumulada": var.cumsum().round(4)},
                   index=nombres).to_string())
print(f"Train PCA: {train_pca.shape}  Test PCA: {test_pca.shape}")
print(f"Guardados en ./{SALIDA}/train_pca.csv y ./{SALIDA}/test_pca.csv")

# Variables con más peso en las dos primeras componentes
cargas = pd.DataFrame(pca.components_[:2].T, index=X.columns,
                      columns=["PC1", "PC2"])
for pc in ["PC1", "PC2"]:
    top = cargas[pc].abs().sort_values(ascending=False).head(5).index
    print(f"{pc} dominada por: {', '.join(top)}")

# ------------------------------------------------------------------
# 3. Modelo anterior (mejores parámetros de la grilla original)
#    aplicado sobre los datos PCA, como línea base
# ------------------------------------------------------------------
PARAMS_ANTERIORES = {"n_estimators": 500, "max_features": "sqrt",
                     "max_depth": None, "min_samples_leaf": 1}


def pipeline(**params_rf):
    return Pipeline([
        ("escalado", StandardScaler()),
        ("pca", PCA(n_components=VARIANZA, svd_solver="full")),
        ("rf", RandomForestClassifier(random_state=SEMILLA, **params_rf)),
    ])


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEMILLA)
metricas_cv = {"roc_auc": "roc_auc", "accuracy": "accuracy",
               "recall": "recall"}
cv_ant = cross_validate(pipeline(**PARAMS_ANTERIORES), X_train, y_train,
                        cv=cv, scoring=metricas_cv, n_jobs=-1)

# ------------------------------------------------------------------
# 4. Optimización de hiperparámetros (búsqueda aleatoria, CV 5x2)
# ------------------------------------------------------------------
espacio = {
    "rf__n_estimators": randint(100, 701),
    "rf__max_features": ["sqrt", "log2", None, 0.3, 0.5],
    "rf__max_depth": [None, 3, 5, 8, 12],
    "rf__min_samples_split": randint(2, 11),
    "rf__min_samples_leaf": randint(1, 6),
    "rf__bootstrap": [True, False],
    "rf__criterion": ["gini", "entropy"],
    "rf__class_weight": [None, "balanced", "balanced_subsample"],
}
busqueda = RandomizedSearchCV(
    pipeline(), espacio, n_iter=N_ITER, cv=cv, scoring=metricas_cv,
    refit="roc_auc", n_jobs=-1, random_state=SEMILLA)
busqueda.fit(X_train, y_train)
optimo = busqueda.best_estimator_
res = pd.DataFrame(busqueda.cv_results_)
i = busqueda.best_index_

print(f"\n=== RandomizedSearchCV ({N_ITER} configuraciones, CV 5x2) ===")
print("Mejores hiperparámetros:")
for k, v in busqueda.best_params_.items():
    print(f"  {k.replace('rf__', '')}: {v}")
print("\nComparación en validación cruzada (train PCA):")
print(pd.DataFrame({
    "Anterior + PCA": [cv_ant[f"test_{m}"].mean() for m in metricas_cv],
    "Optimizado + PCA": [res.loc[i, f"mean_test_{m}"] for m in metricas_cv],
}, index=["ROC AUC", "Accuracy", "Recall maligno"]).round(4).to_string())

print("\nTop 5 configuraciones (ROC AUC CV):")
cols = ["param_rf__n_estimators", "param_rf__max_features",
        "param_rf__max_depth", "param_rf__min_samples_leaf",
        "param_rf__bootstrap", "param_rf__criterion",
        "param_rf__class_weight", "mean_test_roc_auc", "mean_test_recall"]
tabla_top = res.sort_values("rank_test_roc_auc")[cols].head(5)
tabla_top.columns = [c.replace("param_rf__", "") for c in cols]
print(tabla_top.round(4).to_string(index=False))

# ------------------------------------------------------------------
# 5. Evaluación en test
# ------------------------------------------------------------------
def metricas(modelo, Xs):
    p = modelo.predict_proba(Xs)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"accuracy": accuracy_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, p),
            "recall_M": recall_score(y_test, pred), "FN": fn, "FP": fp}


sin_pca = RandomForestClassifier(**PARAMS_ANTERIORES, random_state=SEMILLA,
                                 n_jobs=-1).fit(X_train, y_train)
anterior_pca = pipeline(**PARAMS_ANTERIORES).fit(X_train, y_train)
tabla = pd.DataFrame({
    "Anterior - 30 variables originales": metricas(sin_pca, X_test),
    f"Anterior - PCA ({n_comp} comp.)": metricas(anterior_pca, X_test),
    f"Optimizado - PCA ({n_comp} comp.)": metricas(optimo, X_test),
}).T
print("\n=== TEST ===")
print(tabla.round(4).to_string())
print("\nReporte de clasificación - modelo optimizado con PCA:")
print(classification_report(y_test, optimo.predict(X_test),
                            target_names=["Benigno", "Maligno"], digits=4))

# ------------------------------------------------------------------
# 6. Gráficos: varianza explicada y proyección PC1-PC2
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(1, n_comp + 1)
ax1.bar(x, var, color="#9ec5e8", label="Individual")
ax1.plot(x, var.cumsum(), "o-", color="#2a7ab9", label="Acumulada")
ax1.axhline(VARIANZA, ls="--", color="gray")
ax1.text(n_comp, VARIANZA - 0.05, f"{VARIANZA:.0%}", ha="right")
ax1.set_xticks(x)
ax1.set_xlabel("Componente principal")
ax1.set_ylabel("Proporción de varianza explicada")
ax1.set_title(f"PCA: {n_comp} componentes retienen {var.sum():.1%}")
ax1.legend()
for clase, color, nombre in [(0, "#2a7ab9", "Benigno"),
                             (1, "#d1495b", "Maligno")]:
    m = y_train == clase
    ax2.scatter(train_pca.loc[m, "PC1"], train_pca.loc[m, "PC2"], s=14,
                alpha=0.7, color=color, label=nombre)
ax2.set_xlabel(f"PC1 ({var[0]:.1%})")
ax2.set_ylabel(f"PC2 ({var[1]:.1%})")
ax2.set_title("Train proyectado en PC1-PC2")
ax2.legend()
fig.tight_layout()
fig.savefig(f"{SALIDA}/pca_varianza.png", dpi=120)
print(f"Gráfico guardado en ./{SALIDA}/pca_varianza.png")
