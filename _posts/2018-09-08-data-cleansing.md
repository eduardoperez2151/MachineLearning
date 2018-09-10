---
featured_image: '/images/blog/data-cleansing.png'
excerpt : Uno los puntos fundamentales en Machine Learning es el tratamiento de los datos. En este Post veremos un poco de estadística descriptiva aplicada sobre nuestros datos y sus distribuciones. Además veremos por qué y cómo Normalizar nuestros datos. Veremos conversiones entre tipos de datos y por último veremos cómo detectar y eliminar los Outliers o valores atípicos de nuestro conjuntos de datos.

---

![](/images/blog/data-cleansing.png)

## Introducción
Una de las partes más importantes sobre la preparación de los datos, es comprender el negocio, para que de esta manera tengamos una mejor interpretación de los datos. Esta comprensión de los datos nos ayudará a decidir qué técnicas y/o transformaciones aplicaremos a nuestro conjunto de datos. Una pregunta frecuente es por que tanto trabajo sobre los datos, la respuesta es sencilla y se explica mediante el principio [GIGO(Garbage In - Garbage Out)](https://www.semantics3.com/blog/thoughts-on-the-gigo-principle-in-machine-learning-4fbd3af43dc4).

![GIGO](/images/blog/gigo.png =200x100)

Básicamente si no somos capaces de comprender y analizar datos que le proveemos a nuestros modelos, tampoco vamos hacer capaces de analizar, comprender, interpretar y evaluar la salida de los mismos.

Como se mencionó anteriormente en este post nos preocupamos únicamente en abordar las técnicas( y los fundamentos que las sustentan), que nos ayudarán en la limpieza de nuestros conjuntos de datos.   

## <center><strong>Estadística descriptiva</strong></center>

Una de las herramientas más importantes que tenemos es la estadística descriptiva. Para poder hacer uso de esta, debemos comprender sus conceptos básicos, para poder aplicarla a nuestros conjuntos de datos de manera correcta.

 La estadística descriptiva es una rama de las Matemáticas cuyo objetivo es **recolectar**, **resumir** y **analizar** un conjunto de datos con el fin de **describir las características que este posee**.
Los conceptos más importantes que debemos comprender acerca de la estadística descriptiva son los siguientes:

- **Población:** Es una colección de un número finito de mediciones o una colección grande, virtualmente infinita de datos sobre algún objeto de interés.

- **Muestra:** Es un *subconjunto representativo* seleccionado de una población.

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
