import numpy
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot   as plt

#carga los datos desde un archivo CSV 
datos = numpy.loadtxt("Videojuegos/examen.csv", delimiter=",")

X = datos[:, :-1]  # Variables independientes
y = datos[:, -1]   # Variable objetivo

# Define las reglas de asociación
# salud	cuchillo	arma	enemigos	accion
reglas = [
    [1, 0, 0, 3, 1],
    [2, 0, 0, 3, 1],
    [3, 0, 0, 3, 1],
    [1, 0, 0, 2, 2],
    [2, 0, 0, 2, 2],
    [3, 0, 0, 2, 2],
    [1, 0, 0, 1, 2],
    [2, 0, 0, 1, 2],
    [3, 0, 0, 1, 2],
    [1, 1, 0, 1, 3],
    [2, 1, 0, 1, 3],
    [3, 1, 0, 1, 3],
    [1, 1, 0, 2, 3],
    [2, 1, 0, 2, 3],
    [3, 1, 0, 2, 3],
    [1, 0, 1, 3, 3],
    [2, 0, 1, 3, 3],
    [3, 0, 1, 3, 3],
    [1, 0, 0, 3, 4],
    [2, 0, 0, 3, 4],
    [3, 0, 0, 3, 4],
    [1, 1, 0, 3, 4],
    [2, 1, 0, 3, 4],
    [3, 1, 0, 3, 4],
]


#Crea el modelo de red neuronal
model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))  
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Toma 120 datos aleatorios para entrenamiento
random_indices = numpy.random.choice(len(X_train), size=120, replace=False)
X_train = X_train[random_indices]
y_train = y_train[random_indices]

#Entrena el modelo con las 120 muestras y las reglas de asociación
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=1)

#Realiza predicciones en el conjunto de prueba
predictionY = model.predict(X_test)
predictionY = numpy.round(predictionY).astype(int)  # Redondear las probabilidades a 0 o 1

#Calcula el porcentaje de acierto
accuracy = accuracy_score(y_test, predictionY)
print(f"Porcentaje de acierto en la prueba: {accuracy * 100:.2f}%")

#Realiza la predicción en el conjunto de datos original
prediction = model.predict(X)
print(prediction)

#Redondea las predicciones
rounded = [round(x[0]) for x in prediction]
print(rounded)

#Evalua el modelo
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Evalua el modelo con 5 datos aleatorios
y_pred = model.predict(X_test[:5])
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
accuracy = accuracy_score(y_test[:5], y_pred_binary)
print("Precisión del modelo con 5 datos aleatorios:", accuracy)

#Genera un nuevo archivo CSV con las reglas de asociación aplicadas
df = pandas.DataFrame(reglas, columns=["Salud", "Tiene cuchillo", "Tiene arma", "Enemigos", "Acción"])
df.to_csv("Videojuegos/reglas_asociacion.csv", index=False)