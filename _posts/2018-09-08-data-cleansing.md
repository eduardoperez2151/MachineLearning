---
featured_image: '/images/blog/data-cleansing.png'
excerpt : Uno los puntos fundamentales en Machine Learning es el tratamiento de los datos. En este Post veremos un poco de estadística descriptiva aplicada sobre nuestros datos y sus distribuciones. Además veremos por qué y cómo Normalizar nuestros datos. Veremos conversiones entre tipos de datos y por último veremos cómo detectar y eliminar los Outliers o valores atípicos de nuestro conjuntos de datos.

---

![](/images/blog/cleaning/data-cleansing.png)

## Introducción
Una de las partes más importantes sobre la preparación de los datos, es comprender el negocio, para que de esta manera tengamos una mejor interpretación de los datos. Esta comprensión de los datos nos ayudará a decidir qué técnicas y/o transformaciones aplicaremos a nuestro conjunto de datos. Una pregunta frecuente es por que tanto trabajo sobre los datos, la respuesta es sencilla y se explica mediante el principio [GIGO(Garbage In - Garbage Out)](https://www.semantics3.com/blog/thoughts-on-the-gigo-principle-in-machine-learning-4fbd3af43dc4).

<img src="/images/blog/cleaning/gigo.png" width="200"/>

Básicamente si no somos capaces de comprender y analizar datos que le proveemos a nuestros modelos, tampoco vamos hacer capaces de analizar, comprender, interpretar y evaluar la salida de los mismos.

Como se mencionó anteriormente en este post nos preocupamos únicamente en abordar las técnicas( y los fundamentos que las sustentan), que nos ayudarán en la limpieza de nuestros conjuntos de datos.   

## <center><strong>Estadística descriptiva</strong></center>

Una de las herramientas más importantes que tenemos es la estadística descriptiva. Para poder hacer uso de esta, debemos comprender sus conceptos básicos, para poder aplicarla a nuestros conjuntos de datos de manera correcta.

 La estadística descriptiva es una rama de las Matemáticas cuyo objetivo es **recolectar**, **resumir** y **analizar** un conjunto de datos con el fin de **describir las características que este posee**.
Los conceptos más importantes que debemos comprender acerca de la estadística descriptiva son los siguientes:

  **Población:** Es una colección de un número finito de mediciones o una colección grande, virtualmente infinita de datos sobre algún objeto de interés.

  **Muestra:** Es un *subconjunto representativo* seleccionado de una población.

### **Medidas descriptivas numéricas**

Las medidas descriptivas numéricas son la herramienta fundamental la cual nos ayudaran a describir un conjunto de datos.
Existen para este propósito dos tipos de medidas, la primera segun **la ubicación de su centro** y la segunda **la variabilidad**. La ***tendencia central*** de un conjunto es la disposición de los datos de agruparse alrededor del centro de o determinados valores numéricos. ***La variabilidad*** de un conjunto de datos es la dispersión de los mismos.

A continuación vamos definir tres medidas al momento de hablar de **la tendencia central**.

#### Media:

La media de las observaciones **$$x_1, x_2, ... , x_n$$** es el promedio aritmético y se denota como:

**$$\overline{x}=\sum_{i=1}^{n} x_{i}/{n}$$**

dado que se utiliza todo el conjunto de datos para este cálculo debemos tener cuidado con los **valores extremos** o **outliers**, ya que estos pueden afectar de manera considerable el valor de la **media**.

#### Mediana

La mediana de un conjunto es el valor para el cual todo el conjunto se ordena de manera creciente de modo que la mitad de los elementos del conjunto es menor que la mediana y la otra mitad mayor.

#### Moda
La moda de un conjunto es el valor que ocurre con mayor frecuencia en el conjunto.

### **Dispersión de un conjunto**

Para poder hablar de la **dispersión** de un conjunto una de las medidas mas útiles es la varianza.

#### Varianza

La varianza de las observaciones **$$x_1, x_2, ... , x_n$$** es el promedio del cuadrado de las distancias entre cada observación y la media del conjunto :

**$$s^2=\sum_{i=1}^{n} (x_{i}-\overline{x})^2/({n}-1)$$**

La varianza es una medida confiable sobre la variabilidad de nuestro conjunto ya que si muchas diferencias son grandes entonces la varianza será grande ( o pequeña en caso contrario). **La varianza puede sufrir cambios muy desproporcionado, aún más que la media**, por la existencia de valor extremos o outliers

#### Desvío Estándar

La raíz cuadrada positiva de la varianza recibe el nombre de desviación estándar y se denota de la siguiente manera.

**$$s=+\sqrt{\sum_{i=1}^{n} (x_{i}-\overline{x})^2/({n}-1)}$$**

 **$$s=+\sqrt{s^2} $$**


#### Rango

El Rango es la diferencia entre el mayor valor  y el mínimo de un conjunto.

**$$R=X_{(n)}-X_{(1)}$$**

donde $$X{(i)}$$ es el dato ordenado en la posición i.

Hasta este punto hemos dado solo un repaso básico de la estadistica descriptiva entonces ...
![Hands-On](/images/blog/cleaning/evilkermit-cleansing.jpeg)

Para poner en practica los conceptos vistos hasta el momento, haremos uso de un conjunto de datos estraidos del sitio [UCI](https://archive.ics.uci.edu/ml/datasets.html?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=list), este sitio contiene enlaces a una buena cantidad de datasets, mas adelante le dedicaremos un pequeño post a esta página.

## Dataset (Conjunto de datos)
Examinando los datasets disponibles en UCI, seleccione uno el cual esta asociado al problema de determinar si una persona gana o no mas de 50000 al año dependiendo de algunos atributos como por ejemplo, la edad, sexo, estado civil, acupacion y nivel educativo entre otros. **Como lo comente al inicio es muy importante conocer el negocio y comprender los datos, en este caso vamos a enfocarnos en las técnicas de limpieza de los mismos, por lo que dejaremos los pasos previos a esta etapa para otro post en donde entrenaremos un modelo de clasificación**.

El Dataset lo descargaremos desde la siguiente ubicación:
[https://archive.ics.uci.edu/ml/datasets/Adult](https://archive.ics.uci.edu/ml/datasets/Adult).
Luego de descargar el dataset como primer paso debemos leer cualquier tipo de información que venga adjunta a nuestros datos. En este caso tenemos un archivo llamado **adult.names** que nos brina la informacion general del dataset así también como los atributos que lo componen.
Nuestro Dataset se encuentra en el archivo **adult.data**,
este archivo se encuentra en formato **CSV** (Comma Separated Values), para su manipulación utilizaremos una planilla electronica.

![raw-data](/images/blog/cleaning/raw-dataset.png)
Como podemos observar al abrir nuestros datos estos no possen la cabecera correspondiente, por lo que procedemos a agregarselas basandonos en la información adjunta a el.
