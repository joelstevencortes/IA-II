"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 4: CONVOLUCIÓN Y FILTRADO ESPACIAL
===============================================================================

===============================================================================
SECCIÓN 1: RESOLUCIÓN Y DEMOSTRACIÓN DEL TALLER ANALÍTICO
===============================================================================

-------------------------------------------------------------------------------
TALLER ANALÍTICO: CALCULANDO LA CONVOLUCIÓN 2D (FILTRO DE MEDIA)
-------------------------------------------------------------------------------
Dada la matriz de Imagen I (3x3):
I = [[ 10,  20,  30],
     [ 15, 250,  15],
     [ 20,  10,  20]]

Y el Kernel de Filtro de Media K (3x3) con escala (1/9):
K = [[ 1/9, 1/9, 1/9 ],
     [ 1/9, 1/9, 1/9 ],
     [ 1/9, 1/9, 1/9 ]]

1. Cálculo Matemático del nuevo valor del píxel central (I[1,1] = 250):
   Formula de Convolución Puntual:
   Nuevo_Valor = \sum_{i,j} (I[i,j] * K[i,j])
               = (1/9) * \sum (Todos los elementos de I)

   Desarrollo paso a paso de la sumatoria:
   Suma_Elementos = 10 + 20 + 30 + 15 + 250 + 15 + 20 + 10 + 20
   Suma_Elementos = 390

   Nuevo_Valor_Central = 390 / 9 = 43.3333...
   Al cuantizar a entero de 8 bits (uint8), el valor final procesado es: 43

2. Análisis de Suavizado y Difuminado:
   - ¿Por qué el filtro de media difumina o suaviza la imagen?
     El píxel central original (250) representaba un "ruido de sal" (un pico extremo
     de intensidad blanca aislado en un entorno oscuro). 
     El filtro de media redistribuye el valor del píxel atípico promediándolo de 
     manera uniforme entre sus 8 vecinos. Como resultado, atenúa drásticamente la 
     intensidad de 250 a 43, homogeneizando la vecindad pero produciendo un efecto de
     difuminado (blurring) que atenúa los bordes o variaciones abruptas del objeto.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SECCIÓN 2: DEMOSTRACIÓN EN CÓDIGO DE LA CONVOLUCIÓN MATEMÁTICA
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO: DEMOSTRACIÓN MATEMÁTICA DE CONVOLUCIÓN 2D")
print("=========================================================")

# 1. Definición de la sección de matriz de imagen I (3x3)
I_matriz = np.array([[10, 20, 30],
                     [15, 250, 15],
                     [20, 10, 20]], dtype=np.float32) # [Bloque 1: Entrada] - Sub-matriz de imagen con ruido puntual de 250

# 2. Definición del Kernel de Promedio 3x3
kernel_media_3x3 = np.ones((3, 3), dtype=np.float32) / 9.0 # [Bloque 2: Kernel] - Kernel uniforme normalizado (1/9)

# 3. Producto Hadamard y Sumatoria (Operación de Convolución Puntual)
producto_hadamard = I_matriz * kernel_media_3x3 # [Bloque 3: Multiplicación] - Multiplicación punto a punto
pixel_promediado = np.sum(producto_hadamard) # [Bloque 3: Sumatoria] - Reducción a escalar central
pixel_final_uint8 = np.clip(pixel_promediado, 0, 255).astype(np.uint8) # [Bloque 3: Cuantización] - Truncamiento a 8 bits

print("Matriz de Entrada I (3x3):\n", I_matriz.astype(int)) # [Bloque 4: Salida] - Despliega matriz original
print(f"Valor central antes de la convolución: {I_matriz[1, 1]}") # [Bloque 4: Salida] - Muestra pico de ruido (250)
print(f"Suma total de la vecindad: {np.sum(I_matriz)}") # [Bloque 4: Salida] - Muestra suma (390)
print(f"Nuevo valor central calculado (Float): {pixel_promediado:.4f}") # [Bloque 4: Salida] - Resultado exacto (43.3333)
print(f"Nuevo valor central final (uint8): {pixel_final_uint8}") # [Bloque 4: Salida] - Resultado entero (43)


# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO (ESTRATEGIAS DE SUAVIZADO Y RUIDO)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO: FILTRADO DE MEDIA, GAUSSIANO Y MEDIANA")
print("=========================================================")

# 1. Generación de una Imagen Sintética con Ruido de Sal y Pimienta
np.random.seed(42) # [Bloque 5: Semilla] - Garantiza reproducibilidad
imagen_limpia = np.full((300, 300), 50, dtype=np.uint8) # [Bloque 5: Lienzo] - Fondo gris oscuro (50)
imagen_limpia[80:220, 80:220] = 180 # [Bloque 5: Objeto] - Cuadrado gris claro central (180)

# Copia para inyectar ruido sintético
imagen_ruidosa = imagen_limpia.copy()

# Inyección de Ruido de Sal (píxeles blancos = 255)
prob_sal = 0.05
mascara_sal = np.random.rand(300, 300) < prob_sal # [Bloque 6: Ruido Sal] - Puntos blancos aleatorios
imagen_ruidosa[mascara_sal] = 255

# Inyección de Ruido de Pimienta (píxeles negros = 0)
prob_pimienta = 0.05
mascara_pimienta = np.random.rand(300, 300) < prob_pimienta # [Bloque 6: Ruido Pimienta] - Puntos negros aleatorios
imagen_ruidosa[mascara_pimienta] = 0

# 2. Aplicación de los 3 Filtros con un Kernel Agresivo de 7x7
tamaño_kernel = 7 # [Bloque 7: Parámetro Kernel] - Dimensión 7x7

# A) Filtro de Media (Promedio)
blur_media = cv2.blur(imagen_ruidosa, (tamaño_kernel, tamaño_kernel)) # [Bloque 8: Filtro Media] - cv2.blur

# B) Filtro Gaussiano
blur_gauss = cv2.GaussianBlur(imagen_ruidosa, (tamaño_kernel, tamaño_kernel), 0) # [Bloque 8: Filtro Gaussiano] - cv2.GaussianBlur

# C) Filtro de Mediana
blur_mediana = cv2.medianBlur(imagen_ruidosa, tamaño_kernel) # [Bloque 8: Filtro Mediana] - cv2.medianBlur

# 3. Guardado de Resultados en Archivos de Imagen para Inspección
cv2.imwrite("1_imagen_ruidosa_sal_pimienta.png", imagen_ruidosa)
cv2.imwrite("2_resultado_filtro_media_7x7.png", blur_media)
cv2.imwrite("3_resultado_filtro_gaussiano_7x7.png", blur_gauss)
cv2.imwrite("4_resultado_filtro_mediana_7x7.png", blur_mediana)

print("Procesamiento de filtrado espacial completado.")
print("Imágenes exportadas: '1_imagen_ruidosa_sal_pimienta.png', '2_resultado_filtro_media_7x7.png', '3_resultado_filtro_gaussiano_7x7.png', '4_resultado_filtro_mediana_7x7.png'.")

"""
===============================================================================
ANÁLISIS CRÍTICO Y DEDUCCIÓN ALGORÍTMICA (RESPUESTA AL PUNTO 4 DEL LABORATORIO)
===============================================================================
Comparación crítica de los tres filtros sobre el ruido de Sal y Pimienta (Kernel 7x7):

1. Filtro de Media (Promedio Linear):
   - Comportamiento: Suma todos los píxeles dentro de la ventana de 7x7 (incluyendo los 
     picos 0 y 255 del ruido) y divide entre 49.
   - Resultado: Al promediar valores atípicos extremos con valores normales, crea 
     "manchas" o "halos" grises difuminados alrededor de donde estaban los puntos de ruido. 
     Además, degrada significativamente la nitidez de las bordes del objeto.

2. Filtro Gaussiano:
   - Comportamiento: Asigna mayor peso al píxel central y menor peso a la periferia.
   - Resultado: Aunque preserva mejor los bordes estructurales que la media, si el pico de 
     ruido cae en el centro del kernel, le asigna el peso más alto, propagando el error de 
     manera acentuada y dejando rastros borrosos de ruido.

3. Filtro de Mediana (Filtro No Lineal):
   - Comportamiento Algorítmico: Ordena los 49 valores de la vecindad de menor a mayor y 
     selecciona la posición central (la mediana estadística, elemento en la posición 25).
   - DEDUCCIÓN CLAVE: El ruido de impulsos (Sal = 255, Pimienta = 0) siempre se ubica en 
     los extremos absolutos del arreglo ordenado (las primeras o últimas posiciones). Por 
     lo tanto, la mediana ignora por completo estos picos extremos y selecciona un valor 
     representativo de la masa principal de datos de la vecindad.

CONCLUSIÓN FINAL:
El Filtro de Mediana es algorítmicamente superior para eliminar ruido de Sal y Pimienta, 
reconstruyendo la superficie lisa y preservando intactos los bordes del objeto sin generar 
ninguna clase de manchas o difuminado indeseado.
"""