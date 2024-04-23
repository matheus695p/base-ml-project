## FeatureSelector

### Descripción
`FeatureSelector` es un transformador personalizado para la selección de características utilizando diferentes métodos de selección de características. Este transformador te permite aplicar varios métodos de selección de características para seleccionar características importantes del conjunto de datos de entrada.

### Instalación
Para instalar `FeatureSelector`, puedes utilizar pip. Ejecuta el siguiente comando en tu terminal:

```
pip install -e .
```

### Uso

Para utilizar `FeatureSelector` en tu proyecto de Machine Learning, primero importa la clase desde el paquete. Luego, inicializa un objeto `FeatureSelector` con los parámetros necesarios y aplícalo a tus datos de entrenamiento.

```python
from FeatureSelector import FeatureSelector

# Define los parámetros para la selección de características
fs_params = {
    "selectors": ["model_based"],
    "model_based": {
        "bypass_features": ["passenger_sex_female"],
        "estimator": {
            "class": "xgboost.XGBClassifier",
            "kwargs": {
                "n_estimators": 10,
                "max_depth": 4,
                "random_state": 42,
            }
        },
        "threshold": 0.001,
        "prefit": False
    }
}


# Inicializa el transformador de selección de características
feature_selector = FeatureSelector(fs_params)

# Aplica el transformador a tus datos de entrenamiento
X_train_selected = feature_selector.fit_transform(X_train, y_train)
```

### Ejemplos
Puedes encontrar ejemplos prácticos de cómo utilizar `FeatureSelector` en el archivo `examples.ipynb`. Este archivo contiene casos de uso comunes junto con explicaciones detalladas sobre cómo aplicar la selección de características en diferentes conjuntos de datos.
