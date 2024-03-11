import numpy
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot   as plt

#carga los datos desde un archivo CSV 
datos = numpy.loadtxt("Videojuegos/videojuego.csv", delimiter=",")

#divide los datos en caracteristicas=X Y acciones Y
x=datos[:,:4] #contiene datos de salud,cuchillo,arma y no.enemigos
y=datos[:,4] #contiene las acciones

#definicion del modelo de la red nuronal
model = Sequential()
model.add(Dense(12,input_dim=4,activation='relu')) #Añade una capa densamente conectada con 12 neuronas y función de activación ReLU, 
model.add(Dense(8,activation='relu')) #Añade una capa densamente conectada con 8 neuronas y función de activación ReLU
model.add(Dense(4,activation='softmax')) #Añade una capa densamente conectada con 5 neuronas y función de activación softmax

#compila el modelo
#loss='sparse_categorical_crossentropy'`: Define la función de pérdida para calcular el error entre las predicciones del modelo y las etiquetas reales.
#optimizer=adam :Especifica el algoritmo de optimización Adam, que se utiliza para ajustar los pesos del modelo durante el entrenamiento con el fin de minimizar la función de pérdida.
#metrics=['accuracy']`: Define las métricas que se utilizarán para evaluar el rendimiento del modelo durante el entrenamiento y la evaluación, en este caso, se utiliza la precisión (accuracy) para medir la proporción de muestras correctamente clasificadas.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x, y, epochs=120, batch_size=10, verbose=1)

#Imprime la precisión de entrenamiento después de cada iteración
train_accuracy = history.history['accuracy']
for epoch, accuracy in enumerate(train_accuracy, 1):
    print(f'Iteración {epoch}: precisión de entrenamiento = {accuracy * 100:.2f}%')

#imprime la precisión final del entrenamiento
final_train_accuracy = train_accuracy[-1]
print(f'Precisión final del entrenamiento = {final_train_accuracy * 100:.2f}%')

#toma a 5 datos para hacer su evaluacion
indices_aleatorios = numpy.random.choice(x.shape[0], 5, replace=False)
x_evaluacion = x[indices_aleatorios]
y_evaluacion = y[indices_aleatorios]

#realiza prediccion a esos 5 datos
prediction = model.predict(x_evaluacion)

#define las acciones
acciones_dict = {
    0: "Escapar",
    1: "Andar",
    2: "Atacar",
    3: "Esconderse"
}

#aplica porcentaje de selección  para las predicciones
porcentaje = 0.75
accion = []


#imprime los datos de los registros evaluados junto con las acciones predichas y reales
for i, (datos, prediccion) in enumerate(zip(x_evaluacion, prediction), 1):
    accion_predicha = numpy.argmax(prediccion)
    if numpy.max(prediccion) >= porcentaje:
        accion.append((acciones_dict[accion_predicha], acciones_dict[int(y_evaluacion[i-1])]))
    else:
        accion.append(("No predicho ( porcentaje de selección  no alcanzado)", acciones_dict[int(y_evaluacion[i-1])]))

    print(f"Registro {i}: Datos: {datos}, Acción predicha: {accion[-1][0]}, Acción real en datos: {accion[-1][1]}")


#Evaluar precisión en la evaluación de los 5 registros
accuracy_evaluation = model.evaluate(x_evaluacion, y_evaluacion, verbose=0)
print(f'\nPrecisión de la evaluación de los 5 registros = {accuracy_evaluation[1] * 100:.2f}%')