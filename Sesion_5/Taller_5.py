"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 5: GRADIENTES ESPACIALES Y DETECCIÓN DE BORDES
===============================================================================

===============================================================================
SECCIÓN 1: RESOLUCIÓN Y CÁLCULOS DEL TALLER ANALÍTICO
===============================================================================

-------------------------------------------------------------------------------
TALLER ANALÍTICO: CALCULANDO EL GRADIENTE (SOBEL X Y SOBEL Y)
-------------------------------------------------------------------------------
Dada la matriz de imagen I (3x3) con un borde vertical perfecto:
I = [[   0,   0, 255 ],
     [   0,   0, 255 ],
     [   0,   0, 255 ]]

1. Cálculo de Convolución con Kernel Sobel X (Gx) para el píxel central I[1,1]:
   Kernel Gx = [[ -1,  0,  1 ],
                [ -2,  0,  2 ],
                [ -1,  0,  1 ]]

   Multiplicación punto a punto (Hadamard) entre I y Gx:
   I * Gx = [[ (0*-1), (0*0), (255*1) ],
             [ (0*-2), (0*0), (255*2) ],
             [ (0*-1), (0*0), (255*1) ]]
          = [[ 0, 0, 255 ],
             [ 0, 0, 510 ],
             [ 0, 0, 255 ]]

   Sumatoria total:
   Gx_central = 255 + 510 + 255 = 1020

   VALOR DEL GRADIENTE EN X = 1020
   (Indica un cambio de intensidad fuerte y positivo en la dirección horizontal).

2. Cálculo de Convolución con Kernel Sobel Y (Gy) para el píxel central I[1,1]:
   Kernel Gy = [[ -1, -2, -1 ],
                [  0,  0,  0 ],
                [  1,  2,  1 ]]

   Multiplicación punto a punto entre I y Gy:
   I * Gy = [[ (0*-1), (0*-2), (255*-1) ],
             [ (0*0) , (0*0) , (255*0)  ],
             [ (0*1) , (0*2) , (255*1)  ]]
          = [[ 0, 0, -255 ],
             [ 0, 0,    0 ],
             [ 0, 0,  255 ]]

   Sumatoria total:
   Gy_central = -255 + 0 + 255 = 0

   VALOR DEL GRADIENTE EN Y = 0

   ¿Por qué el resultado en Y es cero y qué indica sobre la dirección del borde?
   El resultado es 0 porque las variaciones de intensidad a lo largo de las filas
   (dirección vertical) son nulas; no hay cambios de luz de arriba hacia abajo.
   Esto indica matemáticamente que el borde es estrictamente VERTICAL. La dirección
   del vector gradiente es perpendicular al borde (apunta horizontalmente en dirección a X).
"""

import cv2
import numpy as np

# ==========================================
# SECCIÓN 2: DEMOSTRACIÓN ALGORÍTMICA DEL GRADIENTE SOBEL (ANALÍTICO)
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO: DEMOSTRACIÓN DE GRADIENTES SOBEL X Y Y")
print("=========================================================")

# 1. Definición de la matriz I (3x3)
I_matriz = np.array([[0, 0, 255],
                     [0, 0, 255],
                     [0, 0, 255]], dtype=np.float32) # [Bloque 1: Entrada] - Sub-matriz con borde vertical

# 2. Definición manual de Kernels Sobel X y Y
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32) # [Bloque 2: Kernel Gx] - Operador diferencial horizontal

Gy = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32) # [Bloque 2: Kernel Gy] - Operador diferencial vertical

# 3. Convolución puntual en el centro
gx_res = np.sum(I_matriz * Gx) # [Bloque 3: Convolución X] - Multiplicación y suma puntual
gy_res = np.sum(I_matriz * Gy) # [Bloque 3: Convolución Y] - Multiplicación y suma puntual
magnitud_g = np.sqrt(gx_res**2 + gy_res**2) # [Bloque 3: Magnitud] - Teorema de Pitágoras G = sqrt(Gx^2 + Gy^2)

print("Matriz de entrada I:\n", I_matriz.astype(int))
print(f"Gradiente Calculado Gx (Horizontal): {gx_res:.1f}")
print(f"Gradiente Calculado Gy (Vertical):   {gy_res:.1f}")
print(f"Magnitud Total del Gradiente G:      {magnitud_g:.1f}")


# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO (INSPECTOR DE BORDES Y EXPERIMENTACIÓN CANNY)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO: INSPECTOR DE BORDES Y HISTÉRESIS")
print("=========================================================")

# 1. Generación de Imagen Sintética con Edificios y Formas Geométricas
np.random.seed(42) # [Bloque 4: Semilla]
escena = np.full((300, 300), 40, dtype=np.uint8) # [Bloque 4: Fondo] - Cielo/Noche (40)

# Edificio 1 (Rectángulo con contraste medio)
escena[100:280, 40:120] = 130 # [Bloque 4: Geometría] - Bloque vertical

# Edificio 2 (Rectángulo con contraste alto)
escena[60:280, 160:260] = 220 # [Bloque 4: Geometría] - Bloque alto

# Ventanas (Detalles de alta frecuencia / textura)
escena[80:110, 180:210] = 50
escena[130:160, 180:210] = 50

# Línea de carretera horizontal (Borde horizontal)
escena[280:290, :] = 200

# 2. Detección de Bordes con Sobel
sobel_x_64f = cv2.Sobel(escena, cv2.CV_64F, 1, 0, ksize=3) # [Bloque 5: Sobel X] - Derivada respecto a X
sobel_y_64f = cv2.Sobel(escena, cv2.CV_64F, 0, 1, ksize=3) # [Bloque 5: Sobel Y] - Derivada respecto a Y

sobel_x_abs = cv2.convertScaleAbs(sobel_x_64f) # [Bloque 5: Conversión] - Valor absoluto y uint8
sobel_y_abs = cv2.convertScaleAbs(sobel_y_64f) # [Bloque 5: Conversión] - Valor absoluto y uint8

# 3. Experimentación con Algoritmo Canny y Umbrales de Histéresis
# A) Umbrales Permisivos/Bajos (Captura ruido y bordes débiles)
canny_bajo = cv2.Canny(escena, 10, 50) # [Bloque 6: Canny Bajo] - T_low=10, T_high=50

# B) Umbrales Óptimos/Balanceados (Bordes definidos y limpios)
canny_optimo = cv2.Canny(escena, 50, 150) # [Bloque 6: Canny Óptimo] - T_low=50, T_high=150

# C) Umbrales Restrictivos/Altos (Solo conserva gradientes muy fuertes)
canny_alto = cv2.Canny(escena, 200, 250) # [Bloque 6: Canny Alto] - T_low=200, T_high=250

# 4. Guardado de Resultados en Archivos de Imagen
cv2.imwrite("1_imagen_escena_original.png", escena)
cv2.imwrite("2_sobel_x_verticales.png", sobel_x_abs)
cv2.imwrite("3_sobel_y_horizontales.png", sobel_y_abs)
cv2.imwrite("4_canny_umbrales_bajos_10_50.png", canny_bajo)
cv2.imwrite("5_canny_umbrales_optimos_50_150.png", canny_optimo)
cv2.imwrite("6_canny_umbrales_altos_200_250.png", canny_alto)

print("Procesamiento de detección de bordes completado.")
print("Resultados guardados correctamente como archivos PNG.")

"""
===============================================================================
ANÁLISIS EXPERIMENTAL DE UMBRALES DE CANNY (RESPUESTA AL PUNTO 4 DEL LABORATORIO)
===============================================================================
Resultados observados tras variar los umbrales de histéresis (T_low, T_high):

1. Umbrales Bajos (T_low = 10, T_high = 50):
   - Comportamiento: El algoritmo es extremadamente sensible a variaciones leves 
     de intensidad.
   - Consecuencia: Detecta texturas secundarias, ruido de fondo y falsos bordes. 
     Los contornos lucen recargados y no aíslan adecuadamente la geometría principal.

2. Umbrales Altos (T_low = 200, T_high = 250):
   - Comportamiento: El algoritmo únicamente acepta transiciones de luz drásticas.
   - Consecuencia: Se pierden estructuras reales de bajo contraste (como bordes de 
     edificios de color medio o detalles de ventanas). Aparecen discontinuidades y 
     segmentos de bordes fragmentados.

3. Umbrales Óptimos para esta Escena (T_low = 50, T_high = 150):
   - Justificación: Mantiene la continuidad de los contornos (gracias al proceso de 
     histéresis que conecta píxeles entre 50 y 150 si están unidos a un borde fuerte > 150) 
     y suprime el ruido del fondo, logrando trazos delgados de 1 píxel de grosor.
"""