from keras.models import Sequential
from keras.layers import Dense
import numpy


#fija las semillas aleaorias para la reproduccibilidad
numpy.random.seed(7)

#carga los datos
dataset = numpy.loadtxt(fname="RedesNeuronales\Actividad2\pima-indians-diabetes.csv",delimiter=",")
#dividido en variables de entrada (x) y de salida (y)
X=dataset[:,0:8]
Y=dataset[:,8]

#crea el modelo
model = Sequential()
model.add(Dense(units=12, input_dim=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

#ejecuta el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#ajusta el modelo
model.fit(X,Y,epochs=150, batch_size=10)

#evalua el modelo
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))

#calcula las predicciones
predictions =model.predict(X)
print(predictions)

#redondeamos las predicciones
rounded = [round(x[0]) for x in predictions]
print(rounded)
