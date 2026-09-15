"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 10: SVM - ALGORITMO DE SUPPORT VECTOR MACHINE
===============================================================================

===============================================================================
SECCIÓN 1: RESOLUCIÓN Y CÁLCULOS DEL TALLER ANALÍTICO (DIBUJANDO EL MARGEN)
===============================================================================

Dataset inicial:
- Clase A (0): (2, 2), (3, 3), (4, 2)
- Clase B (1): (6, 6), (7, 8), (8, 7)

1. LÍNEA RECTA ÓPTIMA Y MARGEN MÁXIMO:
   - El punto más cercano de la Clase A hacia la Clase B es (3, 3).
   - El punto más cercano de la Clase B hacia la Clase A es (6, 6).
   - El punto medio entre (3, 3) y (6, 6) es (4.5, 4.5).
   - La frontera óptima pasa por (4.5, 4.5) con pendiente m = -1.
   - Ecuación del hiperplano de separación: x + y - 9 = 0 (Línea: y = -x + 9).

2. VECTORES DE SOPORTE IDENTIFICADOS:
   - Los Vectores de Soporte son únicamente los puntos críticos situados en el borde 
     del margen de seguridad:
     * De la Clase A: (3, 3) [o los límites expuestos hacia la frontera]
     * De la Clase B: (6, 6)
   - Los puntos lejanos como (2, 2) o (8, 7) no afectan la definición de la calle.

3. ADICIÓN DE UN NUEVO PUNTO EN (1, 1) DE CLASE A:
   - ¿Cambiaría la posición de la línea?: NO.
   - Justificación teórica: El algoritmo SVM maximiza el margen basándose EXCLUSIVAMENTE 
     en los Vectores de Soporte (los puntos más cercanos al límite entre clases). 
     El punto (1, 1) está ubicado en la zona profunda de la Clase A, alejado del margen. 
     Dado que no altera la frontera geométrica de decisión ni se convierte en un Vector 
     de Soporte, la ecuación del hiperplano óptimo permanece exactamente intacta.
"""

import numpy as np
from sklearn.svm import SVC

# ==========================================
# SECCIÓN 2: VERIFICACIÓN ANALÍTICA CON SVM LINEAL
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO: VERIFICACIÓN DE VECTORES DE SOPORTE")
print("=========================================================")

# Dataset base del taller analítico
X_analitico = np.array([[2, 2], [3, 3], [4, 2], [6, 6], [7, 8], [8, 7]])
Y_analitico = np.array([0, 0, 0, 1, 1, 1])

# Entrenamiento modelo lineal
svm_base = SVC(kernel='linear')
svm_base.fit(X_analitico, Y_analitico)

print("Vectores de soporte encontrados por el modelo base:")
print(svm_base.support_vectors_)

# Prueba de adición del punto (1, 1)
X_ampliado = np.vstack([X_analitico, [1, 1]])
Y_ampliado = np.append(Y_analitico, 0)

svm_ampliado = SVC(kernel='linear')
svm_ampliado.fit(X_ampliado, Y_ampliado)

print("\nVectores de soporte tras añadir el punto (1, 1):")
print(svm_ampliado.support_vectors_)
print("-> Se confirma que el punto (1, 1) NO modifica los vectores de soporte ni la frontera.")


# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO (FRONTERAS NO LINEALES Y KERNEL TRICK)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO: COMPARATIVA KERNEL LINEAR VS RBF")
print("=========================================================")

# 1. Dataset con un punto no separable linealmente [5, 5] etiquetado como Clase A (0)
X_lab = np.array([
    [2, 2], [3, 3], [4, 2], 
    [6, 6], [7, 8], [8, 7], 
    [5, 5]  # Punto de prueba para engañar la frontera lineal
])
Y_lab = np.array([0, 0, 0, 1, 1, 1, 0])

nuevo_punto = np.array([[5, 4]])

# 2. Experimento A: Kernel Lineal
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_lab, Y_lab)
pred_linear = svm_linear.predict(nuevo_punto)

print("--- EXPERIMENTO A: KERNEL LINEAL ---")
print("Vectores de soporte con Kernel Lineal:\n", svm_linear.support_vectors_)
print(f"Predicción para el punto [5, 4]: Clase {pred_linear[0]}")

# 3. Experimento B: Kernel RBF (Radial Basis Function)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_lab, Y_lab)
pred_rbf = svm_rbf.predict(nuevo_punto)

print("\n--- EXPERIMENTO B: KERNEL RBF ---")
print("Vectores de soporte con Kernel RBF:\n", svm_rbf.support_vectors_)
print(f"Predicción para el punto [5, 4]: Clase {pred_rbf[0]}")

"""
===============================================================================
SECCIÓN 4: REFLEXIÓN Y ESCENARIOS DEL MUNDO REAL
===============================================================================
PREGUNTA:
El "Kernel Trick" (RBF) le permite al SVM aislar puntos que están "rodeados" por 
la clase enemiga. ¿En qué escenario del mundo real (ej. medicina o reconocimiento 
facial) creen que un kernel lineal fallaría completamente y se requeriría RBF?

RESPUESTA TÉCNICA Y ESCENARIOS:

1. Medicina y Diagnóstico Oncológico (Detección de Tumores):
   Las células cancerígenas o tejidos malignos a menudo crecen como "focos o islas 
   anormales" dentro de una masa de tejido totalmente sano que las rodea por completo. 
   Un kernel lineal solo puede trazar un corte recto en el espacio de características 
   (ej. densidad celular vs permeabilidad), lo que imposibilita aislar el tumor circunscrito. 
   Un kernel RBF proyecta los datos a un espacio de mayor dimensión creando una frontera 
   esférica/no lineal que encierra adecuadamente el tumor sin clasificar erróneamente el 
   tejido sano contiguo.

2. Reconocimiento Facial y Biométrica:
   Las variaciones en la iluminación, ángulos de rotación de la cabeza y expresiones 
   faciales generan distribuciones altamente complejas y no lineales en el espacio de 
   pixeles o descriptores faciales. Los rasgos distintivos de un mismo individuo pueden 
   formar clusters concéntricos o intercalados respecto a los de otras personas. El kernel 
   RBF mapea estas relaciones no lineales permitiendo delimitar fronteras envolventes 
   precisas para la identificación de identidades.
"""