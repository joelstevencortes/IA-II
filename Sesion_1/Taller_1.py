"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
TALLER 1: REPASO DE ÁLGEBRA LINEAL Y PYTHON
===============================================================================

===============================================================================
SECCIÓN 1: RESPUESTAS A TALLERES ANALÍTICOS (TEORÍA)
===============================================================================

-------------------------------------------------------------------------------
TALLER ANALÍTICO 1: INDEXACIÓN Y TENSORES
-------------------------------------------------------------------------------
1. En la Matriz A (5x5), ¿cuál es el valor exacto del elemento A_{2,3}?
   - Respuesta: 0.
   - Justificación:
     La matriz A dada es:
     [[  0, 255, 255, 255,   0],
      [255,   0,   0,   0, 255],
      [255,   0, 128,   0, 255],
      [255,   0,   0,   0, 255],
      [  0, 255, 255, 255,   0]]
     Indizando en base 0, la fila 2 es [255, 0, 128, 0, 255].
     El elemento en la columna 3 es 0.
     Visualmente representa un píxel completamente negro (ausencia de luz)
     ubicado a la derecha del centro de la matriz.

2. Bytes almacenados para un Tensor RGB (1080 x 1920 x 3):
   - Respuesta: 6,220,800 bytes (~5.93 MB).
   - Cálculo:
     Total Bytes = Alto * Ancho * Canales = 1080 * 1920 * 3 = 6,220,800 bytes.
     Cada valor en formato uint8 ocupa 1 byte de memoria.

-------------------------------------------------------------------------------
TALLER ANALÍTICO 2: TRANSFORMACIONES ESPACIALES
-------------------------------------------------------------------------------
1. Transposición de una matriz identidad I_4:
   - Respuesta: Mismas dimensiones y valores (I_4^T = I_4).
   - Justificación:
     La transposición intercambia filas por columnas (A^T_{i,j} = A_{j,i}). 
     Geométricamente refleja los elementos sobre la diagonal principal.
     Como la matriz identidad posee únicamente valores 1 en su diagonal principal
     y 0 fuera de ella, la reflexión no altera ningún elemento.

2. Número de neuronas para vectorizar imagen RGB (200, 200, 3):
   - Respuesta: 120,000 neuronas.
   - Cálculo:
     Capa de Entrada = 200 * 200 * 3 = 120,000 elementos (vector unidimensional 1D).
"""

import numpy as np

# ==========================================
# SECCIÓN 2: TALLER DE LABORATORIO 1 (TRANSFORMACIONES AFINES)
# ==========================================
np.random.seed(42) # [Bloque 1: Semilla] - Asegura reproducibilidad en la generación aleatoria
A = np.random.randint(200, 256, (5, 5)).astype(np.float32) # [Bloque 1: Entradas] - Matriz 5x5 sobreexpuesta en rango [200, 255]

alpha = 0.5 # [Bloque 2: Parámetros] - Factor escalar de ganancia para comprimir contraste al 50% | Fórmula: \alpha = 0.5
beta = -50.0 # [Bloque 2: Parámetros] - Escalar de sesgo para atenuar luminosidad global en 50 unidades | Fórmula: \beta = -50.0

A_nueva = (alpha * A) + beta # [Bloque 2: Transformación Afín] - Mapeo afín punto a punto en el tensor bidimensional | Fórmula: A' = \alpha * A + \beta

A_nueva = np.clip(A_nueva, 0, 255) # [Bloque 3: Regularización] - Función de saturación para truncar cotas fuera de rango | Fórmula: min(max(x, 0), 255)
A_nueva = A_nueva.astype(np.uint8) # [Bloque 3: Cuantización] - Convierte coma flotante a enteros sin signo de 8 bits (rango estándar 0-255)

print("=== MATRIZ ORIGINAL (SOBREEXPUESTA) ===\n", A.astype(int)) # [Bloque 3: Salida] - Imprime matriz inicial
print("\n=== MATRIZ PROCESADA (AJUSTADA) ===\n", A_nueva) # [Bloque 3: Salida] - Imprime la matriz ajustada con rango dinámico normalizado


# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO FINAL (PROGRAMANDO UN KERNEL)
# ==========================================
print("\n--- LABORATORIO FINAL: KERNEL Y PRODUCTO HADAMARD ---") # [Bloque 4: Salida] - Imprime cabecera de convolución

I = np.array([[100, 100, 100], [100, 200, 100], [100, 100, 100]], dtype=np.float32) # [Bloque 5: Inicialización] - Declara la matriz de vecindad local de la imagen
K = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32) # [Bloque 5: Inicialización] - Declara el kernel Laplaciano de realce

producto_hadamard = I * K # [Bloque 6: Producto de Hadamard] - Multiplicación elemento a elemento de matrices del mismo orden | Fórmula: C_{i,j} = A_{i,j} * B_{i,j}

pixel_central = np.sum(producto_hadamard) # [Bloque 7: Reducción Tensorial] - Sumatoria de todos los elementos del producto de Schur | Fórmula: \sum C_{i,j}
pixel_central_clipped = np.clip(pixel_central, 0, 255).astype(np.uint8) # [Bloque 7: Normalización] - Truncamiento de la respuesta espacial a rango uint8

print("\nSección de Imagen (I):\n", I.astype(int)) # [Bloque 7: Salida] - Muestra la matriz I
print("Kernel de Realce (K):\n", K.astype(int)) # [Bloque 7: Salida] - Muestra la matriz K
print("Producto Hadamard:\n", producto_hadamard) # [Bloque 7: Salida] - Muestra el producto Hadamard
print("Valor del píxel central calculado:", pixel_central) # [Bloque 7: Salida] - Despliega el escalar obtenido tras la operación de convolución
print("Valor final procesado (uint8):", pixel_central_clipped) # [Bloque 7: Salida] - Despliega el píxel acotado