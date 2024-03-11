import numpy
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot   as plt

#carga los datos desde un archivo CSV 
datos = numpy.loadtxt("Videojuegos/examen.csv", delimiter=",")
