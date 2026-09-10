"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 3: SEGMENTACIÓN Y MORFOLOGÍA MATEMÁTICA
===============================================================================

===============================================================================
SECCIÓN 1: RESPUESTAS TEÓRICAS Y ANÁLISIS CONCEPTUAL
===============================================================================

-------------------------------------------------------------------------------
TALLER ANALÍTICO 1: UMBRALIZACIÓN Y FUNCIÓN ESCALÓN
-------------------------------------------------------------------------------
Dada la sub-matriz I de 3x3 (intensidades de gris):
I = [[ 80, 120, 140],
     [ 90, 200, 210],
     [ 50, 130, 250]]

1. Binarización analítica con Umbral T = 135:
   Definición de la función escalón:
   - f(x,y) = 255 si I(x,y) >= 135
   - f(x,y) = 0   si I(x,y) < 135

   Evaluación elemento a elemento:
   - Fila 0: 80 < 135 -> 0  | 120 < 135 -> 0  | 140 >= 135 -> 255
   - Fila 1: 90 < 135 -> 0  | 200 >= 135 -> 255 | 210 >= 135 -> 255
   - Fila 2: 50 < 135 -> 0  | 130 < 135 -> 0  | 250 >= 135 -> 255

   MATRIZ RESULTANTE (Binarizada):
   [[  0,   0, 255],
    [  0, 255, 255],
    [  0,   0, 255]]

2. Análisis de Error de Segmentación (Si el objetivo era aislar I(x,y) > 100):
   - Justificación del Error:
     Al elegir T = 135, los valores de intensidad 120 y 130 quedaron por debajo del 
     umbral (120 < 135 y 130 < 135) y se transformaron falsamente en 0 (Fondo).
   - Consecuencia Visual sobre el Objeto:
     Ocurre una "sub-segmentación" o fragmentación del objeto. Píxeles que pertenecen 
     al objeto de interés fueron eliminados, generando huecos internos, pérdida de bordes 
     y encogimiento del área real.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SECCIÓN 2: DEMOSTRACIÓN ALGORÍTMICA DE LA UMBRALIZACIÓN (TALLER ANALÍTICO)
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO 1: DEMOSTRACIÓN DE UMBRALIZACIÓN BINARIA")
print("=========================================================")

# 1. Definición de la matriz de entrada 3x3
I_analitica = np.array([[80, 120, 140],
                        [90, 200, 210],
                        [50, 130, 250]], dtype=np.uint8) # [Bloque 1: Entradas] - Matriz bidimensional 3x3 de intensidades en uint8

# 2. Aplicación de Umbralización con T = 135
T_fijo = 135 # [Bloque 2: Parámetro] - Umbral manual fijado
_, I_binaria = cv2.threshold(I_analitica, T_fijo, 255, cv2.THRESH_BINARY) # [Bloque 2: Binarización] - Evaluación escalón | f(x,y) = 255 si I >= T else 0

print("Matriz Original I (3x3):\n", I_analitica) # [Bloque 3: Salida] - Despliega valores iniciales de gris
print(f"\nMatriz Resultante Binarizada con T={T_fijo}:\n", I_binaria) # [Bloque 3: Salida] - Muestra la matriz binarizada en {0, 255}


# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO FINAL (TRATAMIENTO DE RUIDO Y MORFOLOGÍA)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO FINAL: TRATAMIENTO DE RUIDO Y MORFOLOGÍA")
print("=========================================================")

# 1. Generación de Imagen Sintética con Objeto y Ruido
np.random.seed(42) # [Bloque 4: Semilla] - Reproducibilidad estocástica
imagen_base = np.zeros((200, 200), dtype=np.uint8) # [Bloque 4: Fondo] - Lienzo negro (0)

# Dibujar un objeto principal (Cuadrado central blanco)
imagen_base[50:150, 50:150] = 200 # [Bloque 4: Objeto] - Región de interés con intensidad 200

# Agregar ruido de "Sal" (puntos blancos en el fondo)
ruido_sal = np.random.rand(200, 200) > 0.98 # [Bloque 4: Ruido Fondo] - Máscara aleatoria
imagen_base[ruido_sal] = 220

# Agregar ruido de "Pimienta" (huecos negros dentro del objeto)
ruido_pimienta = np.random.rand(200, 200) > 0.98 # [Bloque 4: Ruido Objeto] - Máscara aleatoria
imagen_base[50:150, 50:150][ruido_pimienta[50:150, 50:150]] = 20

# 2. Binarización Estática con Umbral T = 100
_, img_binaria_ruido = cv2.threshold(imagen_base, 100, 255, cv2.THRESH_BINARY) # [Bloque 5: Binarización] - Conversión a blanco y negro absolutos

# 3. Construcción del Elemento Estructurante (Kernel 3x3)
kernel_3x3 = np.ones((3, 3), dtype=np.uint8) # [Bloque 6: Kernel] - Matriz booleana rectangular de unos

# 4. Operación Morfológica de APERTURA (Erosión seguida de Dilatación)
# Se utiliza para eliminar ruido blanco aislado del fondo (Sal)
img_apertura = cv2.morphologyEx(img_binaria_ruido, cv2.MORPH_OPEN, kernel_3x3) # [Bloque 7: Apertura] - Opening = Erode + Dilate

# 5. Operación Morfológica de CIERRE (Dilatación seguida de Erosión)
# Se utiliza para rellenar huecos negros dentro del objeto (Pimienta)
img_cierre = cv2.morphologyEx(img_binaria_ruido, cv2.MORPH_CLOSE, kernel_3x3) # [Bloque 8: Cierre] - Closing = Dilate + Erode

# 6. Guardado de Resultados para inspección visual
cv2.imwrite("1_binaria_con_ruido.png", img_binaria_ruido)
cv2.imwrite("2_operacion_apertura.png", img_apertura)
cv2.imwrite("3_operacion_cierre.png", img_cierre)

print("Procesamiento morfológico completado.")
print("Imágenes resultantes exportadas: '1_binaria_con_ruido.png', '2_operacion_apertura.png' y '3_operacion_cierre.png'.")

"""
===============================================================================
CONCLUSIONES Y COMPARATIVA MORFOLÓGICA (RESPUESTA AL PUNTO 6 DEL LABORATORIO)
===============================================================================
Análisis comparativo de las operaciones aplicadas:

1. Operación de Apertura (Erosión + Dilatación):
   - Efecto: Elimina eficazmente los pequeños puntos blancos dispersos en el fondo 
     (ruido de sal) porque la erosión inicial elimina regiones más pequeñas que 
     el elemento estructurante. Luego, la dilatación restaura la escala del objeto.
   - Limitación: Mantiene o amplía ligeramente los huecos negros dentro del objeto.

2. Operación de Cierre (Dilatación + Erosión):
   - Efecto: Rellena eficazmente las grietas y perforaciones negras dentro del objeto 
     (ruido de pimienta) gracias a que la dilatación conecta las fronteras internas.
   - Limitación: Consolida o agranda el ruido blanco del fondo.

CONCLUSIÓN FINAL:
Para esta imagen específica que contenía ruido de fondo (puntos blancos), la operación 
más efectiva fue la APERTURA (Opening). Logró limpiar por completo el fondo sin 
reducir ni deformar permanentemente la estructura geométrica del objeto principal. 
En proyectos de Visión por Computadora, se recomienda encadenar ambas (Apertura seguida de Cierre)
cuando coexistan ruidos de sal y pimienta de forma simultánea.
"""