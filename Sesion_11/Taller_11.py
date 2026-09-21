"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 11: REDES NEURONALES - EL PERCEPTRÓN
===============================================================================

===============================================================================
SECCIÓN 1: RESOLUCIÓN Y CÁLCULOS DEL TALLER ANALÍTICO (CALCULANDO EL DISPARO)
===============================================================================

Escenario: Evaluación de Aprobación de Crédito
Entradas:
  - X1 (Ingresos) = 50
  - X2 (Deudas)   = 20
Parámetros internos de la neurona:
  - W1 (Peso de ingresos) = 0.8
  - W2 (Peso de deudas)   = -0.5
  - b  (Sesgo/Bias)        = -10

1. CÁLCULO DE LA COMBINACIÓN LINEAL (Z):
   Z = (X1 * W1) + (X2 * W2) + b
   Z = (50 * 0.8) + (20 * -0.5) + (-10)
   Z = 40 + (-10) - 10
   Z = 40 - 10 - 10 = 20

2. EVALUACIÓN EN LA FUNCIÓN ESCALÓN (ACTIVACIÓN):
   f(Z) = 1 si Z >= 0
   f(Z) = 0 si Z < 0

   Como Z = 20 >= 0:
   Salida = 1 (La neurona dispara aprobando el crédito).

3. ANÁLISIS CONCEPTUAL DEL PESO W2:
   El peso W2 corresponde a la variable X2 (Deudas). Tiene sentido empresarial que 
   W2 sea negativo (-0.5) porque las deudas representan un riesgo financiero para 
   la entidad. En la combinación lineal, un valor negativo de peso penaliza o 
   reduce la suma ponderada Z a medida que el cliente acumula más deudas, 
   dificultando que alcance el umbral de activación (Z >= 0) para aprobar el crédito.
"""

import numpy as np

# ==========================================
# FUNCIONES BÁSICAS DEL PERCEPTRÓN
# ==========================================
def funcion_escalon(z):
    """Función de activación escalón (Step Function)."""
    return 1 if z >= 0 else 0

def perceptron(X, W, b):
    """
    Calcula la propagación hacia adelante (Forward Pass) de una sola neurona.
    Z = X · W + b
    """
    Z = np.dot(X, W) + b
    return funcion_escalon(Z)


# ==========================================
# SECCIÓN 2: VERIFICACIÓN ANALÍTICA EN PYTHON
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO: VERIFICACIÓN DEL CRÉDITO BANCARIO")
print("=========================================================")

X_cliente = np.array([50, 20])
W_credito = np.array([0.8, -0.5])
b_credito = -10

Z_calculado = np.dot(X_cliente, W_credito) + b_credito
salida_credito = perceptron(X_cliente, W_credito, b_credito)

print(f"Valor Z obtenido: {Z_calculado}")
print(f"Resultado del Perceptrón: {salida_credito} ({'APROBADO' if salida_credito == 1 else 'RECHAZADO'})\n")


# ==========================================
# SECCIÓN 3: COMPUERTA LÓGICA AND (CÓDIGO BASE)
# ==========================================
print("=========================================================")
print("EVALUACIÓN DE LA COMPUERTA LÓGICA AND (CÓDIGO INICIAL)")
print("=========================================================")

W_and = np.array([0.5, 0.5])
b_and = -0.8

casos_and = [np.array([1, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 0])]

print("Entradas\tSalida AND")
print("-" * 25)
for x in casos_and:
    res = perceptron(x, W_and, b_and)
    print(f"{x.tolist()}\t\t{res}")


# ==========================================
# SECCIÓN 4: TALLER DE LABORATORIO (COMPUERTA LÓGICA OR)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO: COMPUERTA LÓGICA OR (PESOS AJUSTADOS)")
print("=========================================================")

"""
Solución del Reto (Compuerta OR):
Para lograr la lógica OR:
  - [1, 1] -> 1
  - [1, 0] -> 1
  - [0, 1] -> 1
  - [0, 0] -> 0

Ajuste matemático de parámetros:
- Definimos W = [0.5, 0.5] y b = -0.2:
  * x = [1, 1] => Z = (1*0.5) + (1*0.5) - 0.2 =  0.8 >= 0 -> Salida 1
  * x = [1, 0] => Z = (1*0.5) + (0*0.5) - 0.2 =  0.3 >= 0 -> Salida 1
  * x = [0, 1] => Z = (0*0.5) + (1*0.5) - 0.2 =  0.3 >= 0 -> Salida 1
  * x = [0, 0] => Z = (0*0.5) + (0*0.5) - 0.2 = -0.2 <  0 -> Salida 0
"""

W_or = np.array([0.5, 0.5])
b_or = -0.2

print(f"Pesos seleccionados (W): {W_or.tolist()}")
print(f"Sesgo seleccionado (b)  : {b_or}\n")

casos_or = [np.array([1, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 0])]

print("Entradas\tSalida OR")
print("-" * 25)
for x in casos_or:
    res = perceptron(x, W_or, b_or)
    print(f"{x.tolist()}\t\t{res}")