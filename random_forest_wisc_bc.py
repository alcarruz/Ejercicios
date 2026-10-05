"""
Diagnosticando cáncer con Random Forest
Dataset: Breast Cancer Wisconsin (Diagnostic) - wisc_bc_data.csv
  Descripción: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
  Fuente CSV:  https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/wisc_bc_data.csv

Uso:
    pip install pandas scikit-learn matplotlib
    python random_forest_wisc_bc.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)

URL = ("https://raw.githubusercontent.com/stedy/"
       "Machine-Learning-with-R-datasets/master/wisc_bc_data.csv")
RUTA_LOCAL = "wisc_bc_data.csv"
SEMILLA = 42
SALIDA = "resultados"

# ------------------------------------------------------------------
# 1. Carga
# ------------------------------------------------------------------
df = pd.read_csv(RUTA_LOCAL if os.path.exists(RUTA_LOCAL) else URL)
os.makedirs(SALIDA, exist_ok=True)

# ------------------------------------------------------------------
# 2. Revisión del dataset
# ------------------------------------------------------------------
print("=== Revisión del dataset ===")
print(f"Filas: {df.shape[0]}  Columnas: {df.shape[1]}")
print(f"Valores nulos totales: {df.isna().sum().sum()}")
print(f"IDs duplicados: {df['id'].duplicated().sum()}")
print("\nDistribución de la variable objetivo (diagnosis):")
print(df["diagnosis"].value_counts().rename({"B": "Benigno", "M": "Maligno"}))
print(df["diagnosis"].value_counts(normalize=True).round(3))
print("\nResumen de algunas variables:")
print(df[["radius_mean", "texture_mean", "area_mean",
          "concave points_mean"]].describe().round(2))

# ------------------------------------------------------------------
# 3. Preparación
#    - 'id' no aporta información predictiva -> se elimina
#    - diagnosis: M (maligno) = 1, B (benigno) = 0
#    - Random Forest no requiere escalar las variables
# ------------------------------------------------------------------
X = df.drop(columns=["id", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
print(f"\nEntrenamiento: {len(X_train)}  Prueba: {len(X_test)}")

# ------------------------------------------------------------------
# 4. Modelo base
# ------------------------------------------------------------------
rf_base = RandomForestClassifier(n_estimators=500, oob_score=True,
                                 random_state=SEMILLA, n_jobs=-1)
rf_base.fit(X_train, y_train)
print("\n=== Random Forest base (500 árboles) ===")
print(f"OOB accuracy: {rf_base.oob_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, rf_base.predict(X_test)):.4f}")

# ------------------------------------------------------------------
# 5. Ajuste de hiperparámetros con validación cruzada (5 folds)
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
grilla = {
    "n_estimators": [200, 500],
    "max_features": ["sqrt", "log2", 0.3],
    "max_depth": [None, 8],
    "min_samples_leaf": [1, 3],
}
busqueda = GridSearchCV(
    RandomForestClassifier(random_state=SEMILLA, n_jobs=-1),
    grilla, cv=cv, scoring="roc_auc", n_jobs=-1)
busqueda.fit(X_train, y_train)
modelo = busqueda.best_estimator_
print("\n=== Mejor modelo (GridSearchCV, métrica ROC AUC) ===")
print(f"Mejores parámetros: {busqueda.best_params_}")
print(f"ROC AUC CV: {busqueda.best_score_:.4f}")
acc_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy")
print(f"Accuracy CV: {acc_cv.mean():.4f} ± {acc_cv.std():.4f}")

# ------------------------------------------------------------------
# 6. Evaluación en el conjunto de prueba
# ------------------------------------------------------------------
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]
print("\n=== Evaluación en conjunto de prueba ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print("\nMatriz de confusión (filas = real, columnas = predicho):")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=["Real Benigno", "Real Maligno"],
                   columns=["Pred Benigno", "Pred Maligno"]))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred,
                            target_names=["Benigno", "Maligno"], digits=4))

ConfusionMatrixDisplay(cm, display_labels=["Benigno", "Maligno"]).plot(
    cmap="Blues", colorbar=False)
plt.title("Random Forest - Matriz de confusión (test)")
plt.tight_layout()
plt.savefig(f"{SALIDA}/matriz_confusion.png", dpi=120)
plt.close()

# ------------------------------------------------------------------
# 7. Importancia de variables
# ------------------------------------------------------------------
importancias = (pd.Series(modelo.feature_importances_, index=X.columns)
                .sort_values(ascending=False))
print("=== Top 10 variables más importantes ===")
print(importancias.head(10).round(4))

top = importancias.head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top.index, top.values, color="#2a7ab9")
plt.xlabel("Importancia (reducción media de impureza Gini)")
plt.title("Random Forest - Top 15 variables")
plt.tight_layout()
plt.savefig(f"{SALIDA}/importancia_variables.png", dpi=120)
plt.close()
print(f"\nGráficos guardados en ./{SALIDA}/")
