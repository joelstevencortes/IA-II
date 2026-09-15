"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 9: KNN - ALGORITMO DE LOS K VECINOS MÁS CERCANOS
===============================================================================

===============================================================================
SECCIÓN 1: RESOLUCIÓN Y CÁLCULOS DEL TALLER ANALÍTICO (VOTACIÓN ESPACIAL)
===============================================================================

Dataset histórico:
- Cliente 1 (A): (20, 30) -> Clase 0 (NO COMPRA)
- Cliente 2 (B): (40, 50) -> Clase 1 (COMPRA)
- Cliente 3 (C): (35, 45) -> Clase 1 (COMPRA)
Punto Nuevo (P): (30, 40)

1. CÁLCULO DE DISTANCIAS EUCLIDIANAS:
   Fórmula general: d(P, X) = sqrt((x2 - x1)^2 + (y2 - y1)^2)

   A) Distancia a Cliente A (20, 30):
      d(P, A) = sqrt((30 - 20)^2 + (40 - 30)^2)
              = sqrt(10^2 + 10^2) = sqrt(100 + 100) = sqrt(200) ≈ 14.1421

   B) Distancia a Cliente B (40, 50):
      d(P, B) = sqrt((30 - 40)^2 + (40 - 50)^2)
              = sqrt((-10)^2 + (-10)^2) = sqrt(100 + 100) = sqrt(200) ≈ 14.1421

   C) Distancia a Cliente C (35, 45):
      d(P, C) = sqrt((30 - 35)^2 + (40 - 45)^2)
              = sqrt((-5)^2 + (-5)^2) = sqrt(25 + 25) = sqrt(50) ≈ 7.0711

   Resumen de Orden de Proximidad:
   1. Cliente C (d ≈ 7.07)   -> Clase 1 (COMPRA)
   2. Cliente A (d ≈ 14.14)  -> Clase 0 (NO COMPRA) [Empate en distancia con B]
   3. Cliente B (d ≈ 14.14)  -> Clase 1 (COMPRA)    [Empate en distancia con A]

2. CLASIFICACIÓN CON K = 1:
   - El vecino más cercano es C (distancia 7.07), cuya clase es COMPRA (1).
   - Resultado: Clase 1 (COMPRA).

3. CLASIFICACIÓN CON K = 3 Y ANÁLISIS DE CAMBIO:
   - Se toman los 3 vecinos: C (COMPRA), A (NO COMPRA), B (COMPRA).
   - Votación democrática: 2 votos para COMPRA (1) vs 1 voto para NO COMPRA (0).
   - Resultado: Clase 1 (COMPRA).
   - ¿Hubo cambio en la decisión?: No, la predicción se mantuvo en COMPRA (1) debido a que 
     dos de los tres clientes históricos más cercanos pertenecen a esa categoría.
"""

import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# SECCIÓN 2: VERIFICACIÓN MATEMÁTICA DEL TALLER ANALÍTICO
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO: VERIFICACIÓN DE DISTANCIAS Y VOTACIÓN")
print("=========================================================")

punto_nuevo = np.array([30, 40])
clientes = {
    'A (20, 30) - NO COMPRA': np.array([20, 30]),
    'B (40, 50) - COMPRA':    np.array([40, 50]),
    'C (35, 45) - COMPRA':    np.array([35, 45])
}

for nombre, coords in clientes.items():
    dist = math.sqrt(np.sum((punto_nuevo - coords) ** 2))
    print(f"Distancia Euclidiana a Cliente {nombre}: {dist:.4f}")

# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO (CLASIFICADOR UNIVERSAL AMPLIADO)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO: CLASIFICADOR KNN EN 3D (10 REGISTROS)")
print("=========================================================")

# 1. Dataset ampliado: 10 registros y 3 columnas [Edad, Salario (en miles), Nro_Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],  # Registro 1
    [40, 50, 2],  # Registro 2
    [35, 45, 1],  # Registro 3
    [22, 25, 0],  # Registro 4
    [50, 80, 3],  # Registro 5
    [18, 20, 0],  # Registro 6
    [45, 60, 2],  # Registro 7
    [30, 35, 1],  # Registro 8
    [55, 90, 2],  # Registro 9
    [28, 42, 0]   # Registro 10
])

# Etiquetas correspondientes: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])

# Nuevo sujeto de prueba [Edad=30, Salario=40, Nro_Hijos=1]
nuevo_sujeto = np.array([[30, 40, 1]])

print(f"Punto a clasificar [Edad, Salario, Hijos]: {nuevo_sujeto[0].tolist()}\n")

# Experimento A: Evaluación con K = 1
knn_k1 = KNeighborsClassifier(n_neighbors=1)
knn_k1.fit(X_entrenamiento, Y_entrenamiento)
pred_k1 = knn_k1.predict(nuevo_sujeto)
etiqueta_k1 = "COMPRA (1)" if pred_k1[0] == 1 else "NO COMPRA (0)"
print(f"Predicción con K = 1: {etiqueta_k1}")

# Experimento B: Evaluación con K = 5
knn_k5 = KNeighborsClassifier(n_neighbors=5)
knn_k5.fit(X_entrenamiento, Y_entrenamiento)
pred_k5 = knn_k5.predict(nuevo_sujeto)
etiqueta_k5 = "COMPRA (1)" if pred_k5[0] == 1 else "NO COMPRA (0)"
print(f"Predicción con K = 5: {etiqueta_k5}")

"""
===============================================================================
SECCIÓN 4: PREGUNTA DE ANÁLISIS - LA MALDICIÓN DE LA DIMENSIONALIDAD
===============================================================================
PREGUNTA:
Si en lugar de 3 columnas tuvieran 1,000 columnas (como los píxeles de una imagen), 
¿qué pasaría matemáticamente con la Distancia Euclidiana entre los puntos?

RESPUESTA TÉCNICA Y DISCUSIÓN:
1. Fenómeno de Esparcimiento Espacial:
   A medida que el número de dimensiones (características) crece exponencialmente hacia 1,000, 
   el volumen del espacio vectorial se vuelve tan inmenso que los datos disponibles se vuelven 
   extremadamente dispersos.

2. Concentración de Distancias:
   Matemáticamente, la Distancia Euclidiana acumula diferencias al cuadrado a lo largo de 
   cada dimensión extra:
   d(A, B) = sqrt(sum_{i=1}^{N} (A_i - B_i)^2)
   Con N = 1,000, la distancia entre cualquier par de puntos tiende a concentrarse en valores 
   muy similares. La diferencia relativa entre la distancia al vecino más cercano y al más 
   lejano se reduce casi a cero (d_max - d_min -> 0).

3. Pérdida del Sentido de Vecindad en KNN:
   Como todos los puntos terminan estando conceptualmente a la "misma distancia" unos de otros, 
   el concepto de "vecino más cercano" pierde su significado geométrico. Esto degrada la 
   capacidad predictiva del algoritmo KNN y eleva drásticamente el costo computacional.
"""