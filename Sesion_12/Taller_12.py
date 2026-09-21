"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 12: REDES NEURONALES DENSAS O MULTICAPA (MLP)
===============================================================================

===============================================================================
SECCIÓN 1: RESOLUCIÓN Y CÁLCULOS DEL TALLER ANALÍTICO (CONTANDO PARÁMETROS)
===============================================================================

Arquitectura definida:
  - Capa de Entrada: 3 variables (Edad, Ingresos, Deuda) -> N_in = 3
  - Capa Oculta   : 1 capa con 4 neuronas             -> N_hidden = 4
  - Capa de Salida: 1 neurona (Aprobado/Rechazado)     -> N_out = 1

1. PESOS (W1) ENTRE ENTRADA Y CAPA OCULTA:
   Cada una de las 3 entradas se conecta con cada una de las 4 neuronas ocultas.
   Pesos W1 = 3 entradas * 4 neuronas = 12 pesos.

2. SESGOS (b1) EN LA CAPA OCULTA:
   Cada neurona oculta requiere 1 parámetro de sesgo (bias).
   Sesgos b1 = 4 sesgos.

3. PESOS (W2) Y SESGOS (b2) ENTRE CAPA OCULTA Y CAPA DE SALIDA:
   - Pesos W2 = 4 neuronas ocultas * 1 neurona de salida = 4 pesos.
   - Sesgos b2 = 1 neurona de salida = 1 sesgo.

4. TOTAL DE PARÁMETROS ENTRENABLES:
   Total = (W1) + (b1) + (W2) + (b2)
   Total = 12 + 4 + 4 + 1 = 21 parámetros entrenables.
"""

import numpy as np

# ==========================================
# FUNCIONES BÁSICAS DE LA RED NEURONAL
# ==========================================
def sigmoide(x):
    """Función de activación Sigmoide: mapea valores al rango (0, 1)."""
    return 1 / (1 + np.exp(-x))


# ==========================================
# SECCIÓN 2: VERIFICACIÓN ANALÍTICA DE LA ARQUITECTURA
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO: RESUMEN DE PARÁMETROS ENTRENABLES")
print("=========================================================")

N_in = 3
N_hidden = 4
N_out = 1

pesos_capa_oculta = N_in * N_hidden
sesgos_capa_oculta = N_hidden
pesos_capa_salida = N_hidden * N_out
sesgos_capa_salida = N_out

total_parametros = (pesos_capa_oculta + sesgos_capa_oculta + 
                    pesos_capa_salida + sesgos_capa_salida)

print(f"Pesos Capa Oculta (W1) : {pesos_capa_oculta}")
print(f"Sesgos Capa Oculta (b1): {sesgos_capa_oculta}")
print(f"Pesos Capa Salida (W2) : {pesos_capa_salida}")
print(f"Sesgos Capa Salida (b2): {sesgos_capa_salida}")
print(f"TOTAL PARÁMETROS       : {total_parametros}\n")


# ==========================================
# SECCIÓN 3: CÓDIGO BASE (EVALUACIÓN DE 1 CLIENTE)
# ==========================================
print("=========================================================")
print("EVALUACIÓN INDIVIDUAL: CÓDIGO BASE (1 CLIENTE)")
print("=========================================================")

# 1. ENTRADA (X): 1 cliente con 3 características
X_1_cliente = np.array([0.5, 0.8, 0.2])

# 2. PARÁMETROS CAPA OCULTA (4 Neuronas)
W1 = np.array([
    [0.1,  0.2,  0.3,  0.4],
    [-0.5, 0.6,  0.7, -0.8],
    [0.9, -0.1,  0.2,  0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])

# Proceso Capa Oculta
Z1_individual = np.dot(X_1_cliente, W1) + b1
A1_individual = sigmoide(Z1_individual)

# 3. PARÁMETROS CAPA DE SALIDA (1 Neurona)
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# Proceso Capa Final
Z2_individual = np.dot(A1_individual, W2) + b2
Salida_individual = sigmoide(Z2_individual)

print(f"Combinación lineal Z1 : {Z1_individual}")
print(f"Activación Sigmoide A1: {A1_individual}")
print(f"Predicción de la Red (Probabilidad): {np.round(Salida_individual[0], 4)}\n")


# ==========================================
# SECCIÓN 4: TALLER DE LABORATORIO (PROCESAMIENTO EN LOTE / BATCH DE 2 CLIENTES)
# ==========================================
print("=========================================================")
print("TALLER DE LABORATORIO: RETO DIMENSIONAL (2 CLIENTES EN LOTE)")
print("=========================================================")

# 1. Matriz de Entradas X de 2x3 (2 clientes, 3 características cada uno)
X_batch = np.array([
    [0.5, 0.8, 0.2],  # Cliente 1
    [0.1, 0.9, 0.9]   # Cliente 2
])

# 2. Propagación Capa Oculta (Las matrices W1 y b1 permanecen idénticas)
Z1_batch = np.dot(X_batch, W1) + b1
A1_batch = sigmoide(Z1_batch)

# 3. Propagación Capa de Salida (Las matrices W2 y b2 permanecen idénticas)
Z2_batch = np.dot(A1_batch, W2) + b2
Salida_batch = sigmoide(Z2_batch)

print("--- RESULTADOS DEL CÁLCULO TENSERIAL SIMULTÁNEO ---")
print("Matriz Z1 (Combinación Lineal Capa Oculta 2x4):\n", Z1_batch)
print("\nMatriz A1 (Activaciones Sigmoide Capa Oculta 2x4):\n", A1_batch)

print("\n--- PREDICCIONES FINALES ---")
for i, proba in enumerate(Salida_batch):
    print(f"Cliente {i+1}: Probabilidad = {np.round(proba, 4)}")