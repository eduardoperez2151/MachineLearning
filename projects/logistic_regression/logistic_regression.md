
## Introducción
---

Como primer proyecto vamos a entrenar un modelo de regresion logistica.
Para ello seleccionamos el siguiente dataset de [UCI](https://archive.ics.uci.edu/ml/datasets/adult). 
Antes de comenzar deberiamos de seguir algun proceso o métodologia que nos guie en el desarrollo de este proyecto.
Por suerte para nuestra ayuda existe una métodologia llamada **[CRISP-DM](http://crisp-dm.eu/)**, esta métodología cuenta con 6 etapas,las cuales son:

- Comprensión del negocio
- Entendimiento y analisis de los datos
- Preparacion de los datos
- Modelado
- Evaluación
- Despliegue

Iremos avanzando en el proyecto, paso a paso aplicando cada una de las etapa de **CRISP-DM**. 
Mas adelante dedicaremos un post para hablar de esta metodología en profundidad, asi que por el momento manos a la obra!!!.
![Proceso CRISP-DM](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/220px-CRISP-DM_Process_Diagram.png)

### Comprender el problema
---
Basandonos en la informacion adjunta al dataset el objetivo del problema es **clasificar** en base a **dos clases** si una persona gana mas, o menos de $50000 dolares al año, en base a un conjunto de atributos.

### Analisis de los datos
---
El dataset cuenta con 15 atributos incluyendo la variable Objetivo, esta dividio en un dos partes para el proposito de entrenamiento y el otro para test. La cantidad de instancias que posse cada conjunto es de
32560 y de 16281 respectivamente. Examinando los dataset podemos ver que los mismos no tiene los encabezados en la data, por lo que debemos leer toda documentación adjunta para poder saber los nombres de cada atributo y a que columna corresponde.


```python
#Vamos a leer nuestro archivo, para ello utilizaremos una libreria muy conocida llamada pandas.
import pandas as pd
pd.set_option('display.max_columns', 500)

#Leemos nuestro archivo para entrenamiento
ds_training=pd.read_csv("/home/eduardo/Escritorio/ML/Blog/problemas/Censo/adult.data")
print("Training data instances count:\t",ds_training.shape[0])
```

    Training data instances count:	 32560


Una vez que hemos leido toda la información y sabemos cuales son los cabezales de nuestro conjunto de datos se los agregaremos.


```python
#Estos son los headers de los datos
headers=['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION-NUM','MARITAL-STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL-GAIN','CAPITAL-LOSS','HOURS-PER-WEEK','NATIVE-COUNTRY','CLASS']

#Se lo agregamos al dataset
ds_training.columns=headers

#Chequeamos que el header fue agregado.
dataset=ds_training

dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>WORKCLASS</th>
      <th>FNLWGT</th>
      <th>EDUCATION</th>
      <th>EDUCATION-NUM</th>
      <th>MARITAL-STATUS</th>
      <th>OCCUPATION</th>
      <th>RELATIONSHIP</th>
      <th>RACE</th>
      <th>SEX</th>
      <th>CAPITAL-GAIN</th>
      <th>CAPITAL-LOSS</th>
      <th>HOURS-PER-WEEK</th>
      <th>NATIVE-COUNTRY</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>Private</td>
      <td>284582</td>
      <td>Masters</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



Como podemos **Education-NUM** y **EDUCATION** repesentan la misma informacion de diferente manera, por lo que podemos prescindir de una, en este caso eliminaremos la columna **EDUCATION** del dataset.

En cuanto a la variable **FNLWGT** es una apreciación subjetiva de cuantas personas afecta el censo ,segun las personas que realizan el censo, en este caso vamos a eliminarla. 


```python
dataset = dataset.drop(['FNLWGT','EDUCATION'],  axis=1)
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>WORKCLASS</th>
      <th>EDUCATION-NUM</th>
      <th>MARITAL-STATUS</th>
      <th>OCCUPATION</th>
      <th>RELATIONSHIP</th>
      <th>RACE</th>
      <th>SEX</th>
      <th>CAPITAL-GAIN</th>
      <th>CAPITAL-LOSS</th>
      <th>HOURS-PER-WEEK</th>
      <th>NATIVE-COUNTRY</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>Private</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>Private</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>Private</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>Private</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



#### Tipos de datos
---
Vamos a revisar los tipos de datos que tiene nuestro dataset.
Como podemos ver muchos de nuestros atributos son polinomiales, para poder utilizarlos en una regresión logistica,
debemos aplicarles algun tipo de transformacion de **Polinomial a Númerico**. Pero antes debemos hacer un chequeo de datos faltantes.


```python
dataset.dtypes
```




    AGE                int64
    WORKCLASS         object
    EDUCATION-NUM      int64
    MARITAL-STATUS    object
    OCCUPATION        object
    RELATIONSHIP      object
    RACE              object
    SEX               object
    CAPITAL-GAIN       int64
    CAPITAL-LOSS       int64
    HOURS-PER-WEEK     int64
    NATIVE-COUNTRY    object
    CLASS             object
    dtype: object



#### Datos Faltantes
---
Una vez que agregamos los headers al dataset comenzaremos a revisar mas en profundidad los datos, comenzaremos por cheuqear los datos faltantes. En el caso de este dataset los datos faltantes son marcado con la cadena **" ?"**, hagamos un recuento de estos.
Para ellos **reemplazaremos** la cadena **" ?"** por el valor **NaN de numpy**, esto lo hacemos ya que **pandas nos ofrece algunas funciones para hacer el recuento de estos valores**.


```python
import numpy as np
np.set_printoptions(threshold=np.inf)
dataset=dataset.replace(" ?", value=np.nan)
dataset.isna().sum()
```




    AGE                  0
    WORKCLASS         1836
    EDUCATION-NUM        0
    MARITAL-STATUS       0
    OCCUPATION        1843
    RELATIONSHIP         0
    RACE                 0
    SEX                  0
    CAPITAL-GAIN         0
    CAPITAL-LOSS         0
    HOURS-PER-WEEK       0
    NATIVE-COUNTRY     583
    CLASS                0
    dtype: int64



ahora vamos a ver que valores contienen dichos atributos, **agrupandolos** por tipo.


```python
print(dataset.groupby(['WORKCLASS']).size(),"\n")

print(dataset.groupby(['OCCUPATION']).size(),"\n")

print(dataset.groupby(['NATIVE-COUNTRY']).size())
```

    WORKCLASS
     Federal-gov           960
     Local-gov            2093
     Never-worked            7
     Private             22696
     Self-emp-inc         1116
     Self-emp-not-inc     2541
     State-gov            1297
     Without-pay            14
    dtype: int64 
    
    OCCUPATION
     Adm-clerical         3769
     Armed-Forces            9
     Craft-repair         4099
     Exec-managerial      4066
     Farming-fishing       994
     Handlers-cleaners    1370
     Machine-op-inspct    2002
     Other-service        3295
     Priv-house-serv       149
     Prof-specialty       4140
     Protective-serv       649
     Sales                3650
     Tech-support          928
     Transport-moving     1597
    dtype: int64 
    
    NATIVE-COUNTRY
     Cambodia                         19
     Canada                          121
     China                            75
     Columbia                         59
     Cuba                             95
     Dominican-Republic               70
     Ecuador                          28
     El-Salvador                     106
     England                          90
     France                           29
     Germany                         137
     Greece                           29
     Guatemala                        64
     Haiti                            44
     Holand-Netherlands                1
     Honduras                         13
     Hong                             20
     Hungary                          13
     India                           100
     Iran                             43
     Ireland                          24
     Italy                            73
     Jamaica                          81
     Japan                            62
     Laos                             18
     Mexico                          643
     Nicaragua                        34
     Outlying-US(Guam-USVI-etc)       14
     Peru                             31
     Philippines                     198
     Poland                           60
     Portugal                         37
     Puerto-Rico                     114
     Scotland                         12
     South                            80
     Taiwan                           51
     Thailand                         18
     Trinadad&Tobago                  19
     United-States                 29169
     Vietnam                          67
     Yugoslavia                       16
    dtype: int64


Podemos imputar valores utilizando sklearn, en este caso eliminaremos estas intancias.


```python
dataset=dataset.dropna()
dataset.isna().sum()
```




    AGE               0
    WORKCLASS         0
    EDUCATION-NUM     0
    MARITAL-STATUS    0
    OCCUPATION        0
    RELATIONSHIP      0
    RACE              0
    SEX               0
    CAPITAL-GAIN      0
    CAPITAL-LOSS      0
    HOURS-PER-WEEK    0
    NATIVE-COUNTRY    0
    CLASS             0
    dtype: int64



# Detección de Outliers


```python
import seaborn as sbn
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,5))
sbn.boxplot(ax=ax, data=dataset,orient="h")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f6084b20208>




![png](output_17_1.png)


Vamos a obtener el percentil 0.98 de cada atributo de nuestro dataset


```python
dataset.quantile([0.25,0.5,0.75,0.95])
dataset=dataset[dataset['AGE']< 68]
dataset=dataset[dataset['EDUCATION-NUM']< 15]
dataset=dataset[dataset['CAPITAL-GAIN']< 14344]
dataset=dataset[dataset['CAPITAL-LOSS']< 1902]
dataset=dataset[dataset['HOURS-PER-WEEK']< 70]

```

Vamos a verificar la correlacion entre los atributos y a graficar los mismos.

# Correlación entre los atributos


```python
correlationMatrix=dataset.corr()
mask = np.zeros_like(correlationMatrix)
mask[np.triu_indices_from(mask)] = True
with sbn.axes_style("white"):
    ax = sbn.heatmap(correlationMatrix, mask=mask,cmap="Greens",annot=True)
```


![png](output_22_0.png)


# Conversión de datos
Ahora vamos a transformar los atributos categoricos a númericos.


```python
dataset=pd.get_dummies(data=dataset,columns=['MARITAL-STATUS','RELATIONSHIP','RACE','SEX','WORKCLASS','OCCUPATION','NATIVE-COUNTRY','CLASS'],drop_first=True)
```


```python
dataset = dataset.rename(columns={'CLASS_ >50K': 'CLASS'})
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EDUCATION-NUM</th>
      <th>CAPITAL-GAIN</th>
      <th>CAPITAL-LOSS</th>
      <th>HOURS-PER-WEEK</th>
      <th>MARITAL-STATUS_ Married-AF-spouse</th>
      <th>MARITAL-STATUS_ Married-civ-spouse</th>
      <th>MARITAL-STATUS_ Married-spouse-absent</th>
      <th>MARITAL-STATUS_ Never-married</th>
      <th>MARITAL-STATUS_ Separated</th>
      <th>MARITAL-STATUS_ Widowed</th>
      <th>RELATIONSHIP_ Not-in-family</th>
      <th>RELATIONSHIP_ Other-relative</th>
      <th>RELATIONSHIP_ Own-child</th>
      <th>RELATIONSHIP_ Unmarried</th>
      <th>RELATIONSHIP_ Wife</th>
      <th>RACE_ Asian-Pac-Islander</th>
      <th>RACE_ Black</th>
      <th>RACE_ Other</th>
      <th>RACE_ White</th>
      <th>SEX_ Male</th>
      <th>WORKCLASS_ Local-gov</th>
      <th>WORKCLASS_ Private</th>
      <th>WORKCLASS_ Self-emp-inc</th>
      <th>WORKCLASS_ Self-emp-not-inc</th>
      <th>WORKCLASS_ State-gov</th>
      <th>WORKCLASS_ Without-pay</th>
      <th>OCCUPATION_ Armed-Forces</th>
      <th>OCCUPATION_ Craft-repair</th>
      <th>OCCUPATION_ Exec-managerial</th>
      <th>OCCUPATION_ Farming-fishing</th>
      <th>OCCUPATION_ Handlers-cleaners</th>
      <th>OCCUPATION_ Machine-op-inspct</th>
      <th>OCCUPATION_ Other-service</th>
      <th>OCCUPATION_ Priv-house-serv</th>
      <th>OCCUPATION_ Prof-specialty</th>
      <th>OCCUPATION_ Protective-serv</th>
      <th>OCCUPATION_ Sales</th>
      <th>OCCUPATION_ Tech-support</th>
      <th>OCCUPATION_ Transport-moving</th>
      <th>NATIVE-COUNTRY_ Canada</th>
      <th>NATIVE-COUNTRY_ China</th>
      <th>NATIVE-COUNTRY_ Columbia</th>
      <th>NATIVE-COUNTRY_ Cuba</th>
      <th>NATIVE-COUNTRY_ Dominican-Republic</th>
      <th>NATIVE-COUNTRY_ Ecuador</th>
      <th>NATIVE-COUNTRY_ El-Salvador</th>
      <th>NATIVE-COUNTRY_ England</th>
      <th>NATIVE-COUNTRY_ France</th>
      <th>NATIVE-COUNTRY_ Germany</th>
      <th>NATIVE-COUNTRY_ Greece</th>
      <th>NATIVE-COUNTRY_ Guatemala</th>
      <th>NATIVE-COUNTRY_ Haiti</th>
      <th>NATIVE-COUNTRY_ Honduras</th>
      <th>NATIVE-COUNTRY_ Hong</th>
      <th>NATIVE-COUNTRY_ Hungary</th>
      <th>NATIVE-COUNTRY_ India</th>
      <th>NATIVE-COUNTRY_ Iran</th>
      <th>NATIVE-COUNTRY_ Ireland</th>
      <th>NATIVE-COUNTRY_ Italy</th>
      <th>NATIVE-COUNTRY_ Jamaica</th>
      <th>NATIVE-COUNTRY_ Japan</th>
      <th>NATIVE-COUNTRY_ Laos</th>
      <th>NATIVE-COUNTRY_ Mexico</th>
      <th>NATIVE-COUNTRY_ Nicaragua</th>
      <th>NATIVE-COUNTRY_ Outlying-US(Guam-USVI-etc)</th>
      <th>NATIVE-COUNTRY_ Peru</th>
      <th>NATIVE-COUNTRY_ Philippines</th>
      <th>NATIVE-COUNTRY_ Poland</th>
      <th>NATIVE-COUNTRY_ Portugal</th>
      <th>NATIVE-COUNTRY_ Puerto-Rico</th>
      <th>NATIVE-COUNTRY_ Scotland</th>
      <th>NATIVE-COUNTRY_ South</th>
      <th>NATIVE-COUNTRY_ Taiwan</th>
      <th>NATIVE-COUNTRY_ Thailand</th>
      <th>NATIVE-COUNTRY_ Trinadad&amp;Tobago</th>
      <th>NATIVE-COUNTRY_ United-States</th>
      <th>NATIVE-COUNTRY_ Vietnam</th>
      <th>NATIVE-COUNTRY_ Yugoslavia</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Normalizacion de los datos


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(dataset)

#Obtengo las columnas 
columns=dataset.columns;
normalizedData=scaler.transform(dataset)

#Obtengo un nuevo DataFrame a aprtir de los datos Normalizados entre 0 y 1
dataset=pd.DataFrame(columns=columns,data=normalizedData)
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EDUCATION-NUM</th>
      <th>CAPITAL-GAIN</th>
      <th>CAPITAL-LOSS</th>
      <th>HOURS-PER-WEEK</th>
      <th>MARITAL-STATUS_ Married-AF-spouse</th>
      <th>MARITAL-STATUS_ Married-civ-spouse</th>
      <th>MARITAL-STATUS_ Married-spouse-absent</th>
      <th>MARITAL-STATUS_ Never-married</th>
      <th>MARITAL-STATUS_ Separated</th>
      <th>MARITAL-STATUS_ Widowed</th>
      <th>RELATIONSHIP_ Not-in-family</th>
      <th>RELATIONSHIP_ Other-relative</th>
      <th>RELATIONSHIP_ Own-child</th>
      <th>RELATIONSHIP_ Unmarried</th>
      <th>RELATIONSHIP_ Wife</th>
      <th>RACE_ Asian-Pac-Islander</th>
      <th>RACE_ Black</th>
      <th>RACE_ Other</th>
      <th>RACE_ White</th>
      <th>SEX_ Male</th>
      <th>WORKCLASS_ Local-gov</th>
      <th>WORKCLASS_ Private</th>
      <th>WORKCLASS_ Self-emp-inc</th>
      <th>WORKCLASS_ Self-emp-not-inc</th>
      <th>WORKCLASS_ State-gov</th>
      <th>WORKCLASS_ Without-pay</th>
      <th>OCCUPATION_ Armed-Forces</th>
      <th>OCCUPATION_ Craft-repair</th>
      <th>OCCUPATION_ Exec-managerial</th>
      <th>OCCUPATION_ Farming-fishing</th>
      <th>OCCUPATION_ Handlers-cleaners</th>
      <th>OCCUPATION_ Machine-op-inspct</th>
      <th>OCCUPATION_ Other-service</th>
      <th>OCCUPATION_ Priv-house-serv</th>
      <th>OCCUPATION_ Prof-specialty</th>
      <th>OCCUPATION_ Protective-serv</th>
      <th>OCCUPATION_ Sales</th>
      <th>OCCUPATION_ Tech-support</th>
      <th>OCCUPATION_ Transport-moving</th>
      <th>NATIVE-COUNTRY_ Canada</th>
      <th>NATIVE-COUNTRY_ China</th>
      <th>NATIVE-COUNTRY_ Columbia</th>
      <th>NATIVE-COUNTRY_ Cuba</th>
      <th>NATIVE-COUNTRY_ Dominican-Republic</th>
      <th>NATIVE-COUNTRY_ Ecuador</th>
      <th>NATIVE-COUNTRY_ El-Salvador</th>
      <th>NATIVE-COUNTRY_ England</th>
      <th>NATIVE-COUNTRY_ France</th>
      <th>NATIVE-COUNTRY_ Germany</th>
      <th>NATIVE-COUNTRY_ Greece</th>
      <th>NATIVE-COUNTRY_ Guatemala</th>
      <th>NATIVE-COUNTRY_ Haiti</th>
      <th>NATIVE-COUNTRY_ Honduras</th>
      <th>NATIVE-COUNTRY_ Hong</th>
      <th>NATIVE-COUNTRY_ Hungary</th>
      <th>NATIVE-COUNTRY_ India</th>
      <th>NATIVE-COUNTRY_ Iran</th>
      <th>NATIVE-COUNTRY_ Ireland</th>
      <th>NATIVE-COUNTRY_ Italy</th>
      <th>NATIVE-COUNTRY_ Jamaica</th>
      <th>NATIVE-COUNTRY_ Japan</th>
      <th>NATIVE-COUNTRY_ Laos</th>
      <th>NATIVE-COUNTRY_ Mexico</th>
      <th>NATIVE-COUNTRY_ Nicaragua</th>
      <th>NATIVE-COUNTRY_ Outlying-US(Guam-USVI-etc)</th>
      <th>NATIVE-COUNTRY_ Peru</th>
      <th>NATIVE-COUNTRY_ Philippines</th>
      <th>NATIVE-COUNTRY_ Poland</th>
      <th>NATIVE-COUNTRY_ Portugal</th>
      <th>NATIVE-COUNTRY_ Puerto-Rico</th>
      <th>NATIVE-COUNTRY_ Scotland</th>
      <th>NATIVE-COUNTRY_ South</th>
      <th>NATIVE-COUNTRY_ Taiwan</th>
      <th>NATIVE-COUNTRY_ Thailand</th>
      <th>NATIVE-COUNTRY_ Trinadad&amp;Tobago</th>
      <th>NATIVE-COUNTRY_ United-States</th>
      <th>NATIVE-COUNTRY_ Vietnam</th>
      <th>NATIVE-COUNTRY_ Yugoslavia</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.66</td>
      <td>0.923077</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.179104</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.42</td>
      <td>0.615385</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.582090</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.72</td>
      <td>0.461538</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.582090</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.22</td>
      <td>0.923077</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.582090</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.40</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.582090</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



# Entrenamiento y Validación del Modelo de Regresión Logística

Separo los atributos de la variable objetivo, para el entrenamiento del modelo.


```python
from sklearn.linear_model import LogisticRegression

X= np.array(dataset.drop(['CLASS'],1))
Y = np.array(dataset['CLASS'])

model = LogisticRegression()
model.fit(X,Y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



Ahora verificamos la precisión de nuestro modelo.

## Validación del modelo


```python
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```


```python
validation_size = 0.30
seed = 2018
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
results = model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring='accuracy')
print("%s: %f (%f)" % ("Regresión Logística",results.mean(), results.std()))
```

    Regresión Logística: 0.852348 (0.011346)



```python
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
```

    0.8520527678461349


Ahora veremos la performance de nuestro modelo, mediante el reporte de clasificacion


```python
print(classification_report(Y_validation, predictions))
```

                 precision    recall  f1-score   support
    
            0.0       0.88      0.94      0.91      6415
            1.0       0.69      0.53      0.60      1696
    
    avg / total       0.84      0.85      0.84      8111
    

