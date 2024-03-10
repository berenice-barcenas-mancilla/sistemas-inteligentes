import numpy
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Carga los datos desde un archivo CSV
datos = numpy.loadtxt("RedesNeuronales\\Finanzas\\finanza.csv", delimiter=",")

# Divide los datos en variables independientes (X) y variable dependiente (y)
X = datos[:, :-1]  
y = (datos[:, 2] >= 25) & (datos[:, 3] >= 29000)  
y = y.astype(int)  

# Divide los datos en entrenamiento y prueba 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea el modelo de red neuronal
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compila el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrena el modelo
history = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, validation_data=(x_test, y_test))

# Realiza predicciones en el conjunto de prueba
predictionY = model.predict(x_test)
predictionY = numpy.round(predictionY).astype(int)  

# Calcula el porcentaje de acierto
accuracy = accuracy_score(y_test, predictionY)
print(f"Porcentaje de acierto de la prueba: {accuracy * 100:.2f}%")

# Realiza la predicción en el conjunto de datos original
prediction = model.predict(X)
rounded = [round(x[0]) for x in prediction]

# Evalua el modelo
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Tomar otros 5 datos aleatorios del archivo finanza.csv
indices_aleatorios = numpy.random.choice(len(X), 5)
datos_aleatorios = X[indices_aleatorios]

# Realizar predicciones en los datos aleatorios
predicciones_aleatorias = model.predict(datos_aleatorios)
predicciones_aleatorias_redondeadas = numpy.round(predicciones_aleatorias).astype(int)

# Imprimir los datos aleatorios y sus predicciones
print("\nDatos aleatorios y sus predicciones:")
for i in range(len(datos_aleatorios)):
    # Obtener los valores correspondientes a los datos aleatorios
    valores_aleatorios = datos[indices_aleatorios[i]]
    print(f"Datos: {valores_aleatorios}, Predicción: {predicciones_aleatorias_redondeadas[i]}")

# Agrega las predicciones como una nueva columna al conjunto de datos original
data_predictions = numpy.concatenate((datos, prediction.reshape(-1, 1)), axis=1)

# Convierte el array de datos a un DataFrame de pandas
column_names = ["Estado Civil", "Género", "Edad", "Ingresos", "Predicción"]
df = pandas.DataFrame(data_predictions, columns=column_names)

# Guarda el DataFrame en un nuevo archivo CSV
df.to_csv("finanzasP.csv", index=False)

# Gráfica del historial de entrenamiento
plt.figure(figsize=(12, 6))  
plt.plot(history.history['accuracy'], label='Precisión (datos de entrenamiento)')  
plt.plot(history.history['val_accuracy'], label='Precisión (datos de validación)')  
plt.title('Precisión del Modelo Finanzas - BBM')  
plt.ylabel('Precisión')  
plt.xlabel('Época')  
plt.legend(loc="lower right")  
plt.show()  

# Gráfica del historial de entrenamiento de pérdida
plt.figure(figsize=(12, 6))  
plt.plot(history.history['loss'], label='Pérdida (datos de entrenamiento)')  
plt.plot(history.history['val_loss'], label='Pérdida (datos de validación)')  
plt.title('Pérdida del Modelo Finanzas - BBM')  
plt.ylabel('Pérdida')  
plt.xlabel('Época')  
plt.legend(loc="upper right")  
plt.show()
