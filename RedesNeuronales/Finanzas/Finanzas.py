import numpy   #Librería para operaciones matemáticas en arrays
import pandas   #Librería para manipulación y análisis de datos
from sklearn.metrics import accuracy_score  #Función para calcular la precisión de clasificación
from sklearn.model_selection import train_test_split  #Función para dividir datos en entrenamiento y prueba
from keras.models import Sequential  #Modelo de red neuronal secuencial
from keras.layers import Dense  #Capa densa de la red neuronal
import matplotlib.pyplot as plt #Biblioteca  para graficar

#Carga los datos desde un archivo CSV
datos = numpy.loadtxt("RedesNeuronales\\Finanzas\\finanza.csv", delimiter=",")

X = datos[:, :-1]  #Estado Civil, Género, Edad, Ingresos - Variables independientes
y = (datos[:, 2] >= 25) & (datos[:, 3] >= 29000)  #Fiable (True o False) -Variable dependiente
y = y.astype(int)  #Convierte a 0 y 1 para uso en la clasificación binaria
print(y)

#Divide los datos en entrenamiento y prueba 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Crear el modelo
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=x_train.shape[1]))  #Capa oculta con activación Relu
model.add(Dense(16, activation='relu'))  #Otra capa oculta con activación relu
model.add(Dense(1, activation='sigmoid'))  #Capa de salida con activación sigmoide para clasificación binaria

#Compila el modelo con un optimizador (función de pérdida y métricas)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Entrena el modelo
history = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, validation_data=(x_test, y_test))
#Realiza predicciones en el conjunto de prueba
predictionY = model.predict(x_test)
predictionY = numpy.round(predictionY).astype(int)  #Redondea las probabilidades a 0 o 1

#Calcula el porcentaje de acierto
accuracy = accuracy_score(y_test, predictionY)
print(f"Porcentaje de acierto de la prueba: {accuracy * 100:.2f}%")

#Realiza la prediccion en el conjunto de datos original
prediction = model.predict(X)
print(prediction)
#Redondeamos las predicciones
rounded = [round(x[0]) for x in prediction]
print(rounded)

# Evalua el modelo
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#Toma otros 5 datos aleatorios del archivo finanza.csv
indices_aleatorios = numpy.random.choice(len(X), 5)
datos_aleatorios = X[indices_aleatorios]

#Realiza predicciones en los datos aleatorios
predicciones_aleatorias = model.predict(datos_aleatorios)
predicciones_aleatorias_redondeadas = numpy.round(predicciones_aleatorias).astype(int)

#Imprime los datos aleatorios y sus predicciones
print("\nDatos aleatorios y sus predicciones:")
for i in range(len(datos_aleatorios)):
    print(f"Datos: {datos_aleatorios[i]}, Ingresos: {datos[indices_aleatorios[i], 3]}, Predicción: {predicciones_aleatorias_redondeadas[i]}")

#Agrega las predicciones como una nueva columna al conjunto de datos original
data_predictions = numpy.concatenate((datos, prediction.reshape(-1, 1)), axis=1)

#Convierte el array de datos a un DataFrame de pandas
column_names = ["Estado Civil", "Género", "Edad", "Ingresos", "Predicción"]
df = pandas.DataFrame(data_predictions, columns=column_names)

#Guarda el DataFrame en un nuevo archivo CSV
df.to_csv("finanzasP.csv", index=False)

#Gráfica del historial de entrenamiento
plt.figure(figsize=(12, 6))  #Tamaño de la figura
plt.plot(history.history['accuracy'], label='Precisión (datos de entrenamiento)')  #Precisión en datos de entrenamiento
plt.plot(history.history['val_accuracy'], label='Precisión (datos de validación)')  #Precisión en datos de validación
plt.title('Precisión del Modelo Finanzas - BBM')  #Título  de la gráfica
plt.ylabel('Precisión')  #Etiqueta del eje y
plt.xlabel('Época')  #Etiqueta del eje x
plt.legend(loc="lower right")  #Ubicación de la leyenda
plt.show()  #Muestra la gráfica

#Gráfica del historial de entrenamiento de pérdida
plt.figure(figsize=(12, 6))  #Tamaño de la figura
plt.plot(history.history['loss'], label='Pérdida (datos de entrenamiento)')  #Pérdida en datos de entrenamiento
plt.plot(history.history['val_loss'], label='Pérdida (datos de validación)')  #Pérdida en datos de validación
plt.title('Pérdida del Modelo Finanzas - BBM')  #Título de la gráfica
plt.ylabel('Pérdida')  #Etiqueta del eje y
plt.xlabel('Época')  #Etiqueta del eje x\
plt.legend(loc="upper right")  # Ubicación de la leyenda
plt.show()  # Mostrar la gráfica