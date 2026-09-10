"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 2: TENSOR DE COLOR Y ANÁLISIS ESTADÍSTICO
===============================================================================

===============================================================================
SECCIÓN 1: RESPUESTAS TEÓRICAS Y ANÁLISIS CONCEPTUAL
===============================================================================

-------------------------------------------------------------------------------
TALLER ANALÍTICO 1: OPERACIONES CON TENSORES Y SLICING
-------------------------------------------------------------------------------
Pregunta 1: 
Si ejecutamos la instrucción 'recorte = imagen[100:200, 300:400, 1]' sobre 
una imagen de 1920x1080 (Ancho x Alto) en OpenCV, ¿cuáles son las dimensiones 
exactas (shape) de la variable recorte resultante y qué información contiene?

  - Respuesta: 
    Las dimensiones exactas (shape) son (100, 100).

  - Justificación Técnica y Matemática:
    * Slicing de Filas (Alto): [100:200] extrae exactamente 200 - 100 = 100 píxeles verticales.
    * Slicing de Columnas (Ancho): [300:400] extrae exactamente 400 - 300 = 100 píxeles horizontales.
    * Slicing de Profundidad (Canal): Se selecciona el índice entero escalar 1 (que en OpenCV 
      corresponde al Canal Verde / Green). Al pasar un número entero en lugar de un rango (1:2), 
      NumPy colapsa la tercera dimensión (colapso de rango tensorial).
    * Contenido: Contiene una matriz 2D con las intensidades de luminancia del canal verde (G) 
      de la región rectangular delimitada de 100x100 píxeles.

Pregunta 2:
¿Por qué es computacionalmente más eficiente aislar un canal usando Slicing 
(ej. img[:,:,0]) en lugar de crear dos ciclos 'for' anidados para iterar?

  - Respuesta:
    El slicing aprovecha la contigüidad de la memoria física y la vectorización en C.

  - Justificación Arquitectónica:
    1. Vectorización SIMD: NumPy ejecuta las operaciones de slicing directamente en código C 
       optimizado a nivel de hardware mediante instrucciones SIMD (Single Instruction, Multiple Data).
    2. Localidad de Referencia y Jerarquía de Caché: Iterar con bucles 'for' nativos en Python 
       genera un sobrecosto (overhead) masivo debido a la interpretación dinámica línea a línea,
       e interrumpe la carga eficiente de bloques de memoria contiguos en la memoria Caché L1/L2/L3.
    3. Operación mediante Vistas (Views): El slicing en NumPy no copia datos en memoria RAM, 
       sino que genera un puntero o 'vista' sobre la matriz original en O(1), mientras que dos 
       ciclos 'for' realizan M x N asignaciones elementales lentas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SECCIÓN 2: TALLER DE LABORATORIO 1 (TRANSFORMACIÓN DE ESPACIOS DE COLOR)
# ==========================================
print("=========================================================")
print("TALLER DE LABORATORIO 1: ESCALA DE GRISES Y PRODUCTO PUNTO")
print("=========================================================")

# 1. Definición del píxel de prueba BGR (Amarillo Intenso: B=0, G=255, R=255)
pixel_bgr = np.array([0, 255, 255], dtype=np.float32) # [Bloque 1: Entradas] - Declaración del vector escalar BGR de 3 elementos | Formato: [B, G, R]

# 2. Vector de pesos ponderados de la luminosidad humana ajustado a orden BGR
pesos_luminancia = np.array([0.114, 0.587, 0.299], dtype=np.float32) # [Bloque 2: Pesos] - Coeficientes del estándar ITU-R BT.601 | Y = 0.114*B + 0.587*G + 0.299*R

# 3. Cálculo del valor en escala de grises mediante Producto Punto (Dot Product)
gris_calculado = np.dot(pixel_bgr, pesos_luminancia) # [Bloque 3: Producto Punto] - Proyección del vector BGR sobre el espacio escalar de gris | Fórmula: Y = B \cdot W_B + G \cdot W_G + R \cdot W_R
gris_final_uint8 = np.clip(gris_calculado, 0, 255).astype(np.uint8) # [Bloque 3: Cuantización] - Truncamiento y conversión a entero de 8 bits

print(f"Píxel BGR de prueba (Amarillo puro): {pixel_bgr.astype(int)}") # [Bloque 3: Salida] - Despliega los componentes de color iniciales
print(f"Valor matemático calculado en escala de grises: {gris_calculado:.2f}") # [Bloque 3: Salida] - Muestra el valor en coma flotante
print(f"Valor cuantizado (uint8): {gris_final_uint8}") # [Bloque 3: Salida] - Muestra la intensidad final de 8 bits

# 4. Verificación con OpenCV sobre imagen real sintética
imagen_muestra = np.zeros((100, 100, 3), dtype=np.uint8) # [Bloque 4: Generación] - Creación de matriz 100x100
imagen_muestra[:] = [0, 255, 255] # [Bloque 4: Asignación] - Llenado completo con color amarillo puro BGR

imagen_gris_cv2 = cv2.cvtColor(imagen_muestra, cv2.COLOR_BGR2GRAY) # [Bloque 4: Transformación] - Conversión oficial optimizada mediante OpenCV
print(f"Valor obtenido con cv2.cvtColor(): {imagen_gris_cv2[0, 0]}") # [Bloque 4: Validación] - Muestra la coincidencia exacta con el modelo matemático


# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO 2 (ANÁLISIS ESTADÍSTICO DE HISTOGRAMAS)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO 2: DESCOMPOSICIÓN DE CANALES Y HISTOGRAMAS")
print("=========================================================")

# 1. Generación sintética de una imagen RGB/BGR con predominancia de luz
np.random.seed(42) # [Bloque 5: Semilla] - Inicialización para reproducibilidad de muestra
imagen_rgb = np.zeros((200, 200, 3), dtype=np.uint8) # [Bloque 5: Tensor] - Matriz tridimensional sintética 200x200x3

# Creación de patrón de colores (Fondo azulado con altas intensidades)
imagen_rgb[:, :, 0] = np.random.randint(150, 256, (200, 200)) # [Bloque 5: Canal Azul] - Matriz dominada por intensidades altas [150, 255]
imagen_rgb[:, :, 1] = np.random.randint(50, 150, (200, 200))   # [Bloque 5: Canal Verde] - Matriz con intensidades medias [50, 150]
imagen_rgb[:, :, 2] = np.random.randint(0, 100, (200, 200))     # [Bloque 5: Canal Rojo] - Matriz con intensidades bajas [0, 100]

# 2. Descomposición del Tensor mediante Slicing
canal_azul = imagen_rgb[:, :, 0]  # [Bloque 6: Slicing BGR] - Extracción bidimensional del canal B (Índice 0)
canal_verde = imagen_rgb[:, :, 1] # [Bloque 6: Slicing BGR] - Extracción bidimensional del canal G (Índice 1)
canal_rojo = imagen_rgb[:, :, 2]  # [Bloque 6: Slicing BGR] - Extracción bidimensional del canal R (Índice 2)

# 3. Cálculo de Histogramas para cada canal
hist_azul = cv2.calcHist([imagen_rgb], [0], None, [256], [0, 256])  # [Bloque 7: Histograma B] - Cálculo de frecuencias del canal Azul
hist_verde = cv2.calcHist([imagen_rgb], [1], None, [256], [0, 256]) # [Bloque 7: Histograma G] - Cálculo de frecuencias del canal Verde
hist_rojo = cv2.calcHist([imagen_rgb], [2], None, [256], [0, 256])  # [Bloque 7: Histograma R] - Cálculo de frecuencias del canal Rojo

# 4. Graficación superpuesta utilizando Matplotlib
plt.figure(figsize=(10, 5)) # [Bloque 8: Visualización] - Configuración del lienzo de gráficos
plt.plot(hist_azul, color='blue', label='Canal Azul (B)')   # [Bloque 8: Curva B] - Trazado de la distribución de luminancia azul
plt.plot(hist_verde, color='green', label='Canal Verde (G)') # [Bloque 8: Curva G] - Trazado de la distribución de luminancia verde
plt.plot(hist_rojo, color='red', label='Canal Rojo (R)')     # [Bloque 8: Curva R] - Trazado de la distribución de luminancia roja

plt.title("Distribución Estadísticas de Intensidades por Canal de Color") # [Bloque 8: Formato] - Título del gráfico
plt.xlabel("Nivel de Intensidad del Píxel (0-255)") # [Bloque 8: Formato] - Etiqueta del eje X
plt.ylabel("Frecuencia Absoluta (Cantidad de Píxeles)") # [Bloque 8: Formato] - Etiqueta del eje Y
plt.legend() # [Bloque 8: Formato] - Despliegue de leyenda identificativa
plt.grid(True, linestyle='--', alpha=0.6) # [Bloque 8: Formato] - Rejilla para lectura de datos

# Guardado automático de la figura para respaldo en el repositorio
plt.savefig("histograma_canales.png") # [Bloque 8: Persistencia] - Exporta la gráfica a imagen PNG
print("Gráfica de histogramas generada y guardada exitosamente como 'histograma_canales.png'.")

"""
===============================================================================
CONCLUSIÓN DEL ANÁLISIS ESTADÍSTICO (RESPUESTA AL PUNTO 5 DEL LABORATORIO 2)
===============================================================================
Al observar las curvas del histograma superpuesto:
1. El Canal Azul se agrupa fuertemente hacia la derecha del eje X (rango 150 a 255),
   presentando las frecuencias de intensidad más elevadas.
2. El Canal Verde ocupa la zona central del espectro (rango 50 a 150).
3. El Canal Rojo se concentra a la izquierda (rango 0 a 100), representando valores oscuros.

CONCLUSIÓN FINAL: 
El color dominante en la iluminación general de la imagen analizada es el AZUL. 
En tareas de Visión por Computadora e Inteligencia Artificial, esta asimetría cromática
indica que la información relevante de características dinámicas reside principalmente
en la banda espectral de ondas cortas (Canal B), permitiendo filtrar los canales R y G 
para reducir la dimensionalidad del modelo sin perder varianza explicada.
"""