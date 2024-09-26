# Red Neuronal con keras - Machine Learning

Ficha Técnica: Proyecto de Análisis de Datos Predictivo

Título del Proyecto: Machine Learning Supervisado

Objetivo:
Construir un modelo de machine learning.

Equipo:
Trabajo Individual.

Herramientas y Tecnologías:
- Python
- sklearn
- Google Colab

Procesamiento y análisis:
- limpieza de datos
- preprocesado de datos
- exploración de datos
- Técnica de Análisis de datos predictivo
  
Resultados y Conclusiones:
se realizaron modelos de machine learning(Logistic Regression, Linear Regression), se realizaron métricas para medir el rendimiento y regularizacion L1(Lasso).

Linear Regression:

```python
# construccion y entrenamiento del modelo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
modelo_LinReg = LinearRegression()
modelo_LinReg.fit(X_train, y_train)

# evaluación del modelo
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")
```
Logistic Regression:
```python
# construccion y entrenamiento del modelo
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluación del modelo
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("la matriz de confusion:", cm)
```


Limitaciones/Próximos Pasos:
Identifica y describe cualquier limitación o desafío encontrado durante el proyecto.
Sugiere posibles próximos pasos para extender o mejorar el proyecto de análisis de datos.


