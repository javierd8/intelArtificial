import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

#from keras.models import Sequential
#from keras.layers.core import Dense
from tensorflow import optimizers
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import keras

#Diccionarios con los datos de entreamiento
letra = { #Diccionario (mapa) con las letras [https://es.fonts2u.com/pixel-art-regular.fuente]
    "A" : [1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0, 1,0,0,1,0], #5x4 (filas x columnas (para representacion grafica))
    "B" : [1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0], #5x5
    "C" : [1,1,1,1,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,0], #5x4
    "D" : [1,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,0], #5x5
    "E" : [1,1,1,1,0, 1,0,0,0,0, 1,1,1,0,0, 1,0,0,0,0, 1,1,1,1,0], #5x4
    "F" : [1,1,1,1,0, 1,0,0,0,0, 1,1,1,0,0, 1,0,0,0,0, 1,0,0,0,0], #5x4
    "G" : [1,1,1,1,0, 1,0,0,0,0, 1,0,1,1,0, 1,0,0,1,0, 1,1,1,1,0], #5x4
    "H" : [1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0], #5x4
    "I" : [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0], #5x1
    "J" : [1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,0,0], #5x5
    "K" : [1,0,0,1,0, 1,0,1,0,0, 1,1,0,0,0, 1,0,1,0,0, 1,0,0,1,0], #5x4
    "L" : [1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,0], #5x4
    "M" : [1,1,1,1,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1], #5x5
    "N" : [1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0], #5x4
    "O" : [1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0], #5x4
    "P" : [1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0, 1,0,0,0,0], #5x4
    "Q" : [1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0, 0,0,0,1,0], #5x4 (Cambiado graficamente a una 'q' en vez de 'Q' porque sino quedaba 5x6)
    "R" : [1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,0,0, 1,0,0,1,0], #5x4
    "S" : [1,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,1,0, 1,1,1,1,0], #5x4
    "T" : [1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0], #5x5
    "U" : [1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,0], #5x4
    "V" : [1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0, 0,1,1,0,0], #5x4
    "W" : [1,0,0,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 0,1,1,1,0], #5x5
    "X" : [1,0,1,0,0, 1,0,1,0,0, 0,1,0,0,0, 1,0,1,0,0, 1,0,1,0,0], #5x3
    "Y" : [1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,1,0,0], #5x5
    "Z" : [1,1,1,1,1, 0,0,0,0,1, 0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,1]  #5x5
}
ascii = {}
for i in range(65,91): #Valor de las letras en ascii (A-Z), para minusculas(97-123) (https://simple.m.wikipedia.org/wiki/File:ASCII-Table-wide.svg)
  ascii[chr(i)] = i

#Convierte los diccionarios(mapas) a listas
dLetra = list(letra.values())
dAscii = list(ascii.values())

#Carga los datos de entrenamiento&prueba
training_data = np.asarray(dLetra) #datos de entrenamiento que son los inputs
target_data = np.asarray(dAscii) #salidas, datos de entrenamiento
test = np.array([letra["A"],letra["R"],letra["Q"],letra["U"],letra["I"],letra["T"],letra["E"],letra["C"],letra["T"],letra["U"],letra["R"],letra["A"]]) #datos de prueba, test input

#Crea el modelo
model = Sequential()
model.add(layers.Dense(25000, input_dim=25 , activation='relu')) #Reminder: +Neuronas = +potencia && +tiempoEsperaEntreEpocas
model.add(layers.Dense(1))

#Carga(o no) un modelo existente sobreescribiendo el antes creado
if(1): #Verdadero para cargar un modelo existente, Falso para generar uno nuevo con los datos de arriba
  model = keras.models.load_model('neuroRed.keras')
 
#Compila y entrena el modelo usando los datos de entrenamiento&prueba
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
history = model.fit(training_data, target_data, epochs=500)
predictions = model.predict(training_data)

#Ploteo de Accuracy
plt.plot(history.history["binary_accuracy"],label = "Train")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()
 
#Ploteo de Loss
plt.plot(history.history["loss"],label = "Train")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

#Evalua el modelo imprimiendo al final el resultado
scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
res = model.predict(test).round()
#print(res)
for x in res:
  print (chr(int(x)),end='')

#Guarda el modelo en el path ingresado
model.save('neuroRed.keras')
