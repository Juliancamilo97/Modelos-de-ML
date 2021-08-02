import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#Carga de data en archivo csv
df= pd.read_csv('ModeloSupervisado.csv')
print(df)
#Info sobre la data
df.info()
#Se ve cuantos espacios vacios existen en cada columna
df.isnull().sum()
#Se imprime los varoles que puede tomar como enteros
print(df.describe())
#Se saca promedio en Distancia para rellenar espacios vacios
promedio= int(df["DistanciaEnMetros"].mean())
#Se rellena espacios vacios de Distancia
df["DistanciaEnMetros"]=(df["DistanciaEnMetros"].replace(np.NaN, promedio)).astype(int)
#Las columnas que se toman como objetos por el signo "%" se le quita para poder convertilo en entero
df['Levnombreestablecimiento'] = df['Levnombreestablecimiento'].str.rstrip('%').astype('float') / 100.0
#Se saca el promedio para poder rellenar la columna de nombres establecimiento y se rellena
promedio1=int(df['Levnombreestablecimiento'].mean())
df['Levnombreestablecimiento']=df['Levnombreestablecimiento'].replace(np.NaN, promedio1)
#Se devuelve el valor de porcentaje a cada columna que lo tiene
df['Levnombreestablecimiento']=round(df['Levnombreestablecimiento']*100).astype(int)
df['Levnombreestablecimiento']=df['Levnombreestablecimiento'].astype(str) + '%'
df.info()

#Se definen las variables dependientes e independietes
#X = df[['Levdireccion','Levdirnum','Levpropietario','Levtelefono','Lev_Sticker','Levnombreestablecimiento','DistanciaEnMetros']]
X = df[['DistanciaEnMetros']]
y = df[['Match']]

#Se entrena el algoritmo
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2)

#Se escalan todos los datos
escalar= StandardScaler()
X_train= escalar.fit_transform(X_train)
X_test= escalar.transform(X_test)

#Definimos el algotirmo

algoritmo= LogisticRegression()

#Entrenamiento del modelo
algoritmo.fit(X_train,y_train)

#Realizar una predicción
y_pred= algoritmo.predict(X_test)

#Se ve la precisión del modelo
precision=algoritmo.score(X_train,y_train)
print('Presición del modelo', precision)
print('MSE:', metrics.mean_squared_error(y_test['Match'],y_pred))

Matriz = confusion_matrix(y_test,y_pred)
print('Matriz de confusión: ')
print(Matriz)

#Calculo la exactitud del modelo
exactitud= accuracy_score(y_test,y_pred)
print('La exactitud del modelo es: ', exactitud)

#a. ¿Cuáles con las variables más importantes para el modelo?
#La variable que concidero más relevante es distancia en metros, además que es la única que se esta tomando sin un porcentaje
#b. ¿Confiaría en la predicción de su modelo propuesto según los cálculos del punto 5?
#Pensaría que es una buena predicción, aunque por la matriz de confusión se estan teniendo valores erroneos al creen que todos los datos son cero