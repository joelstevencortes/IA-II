"""
===============================================================================
INSTITUCIÓN UNIVERSITARIA DE COLOMBIA
ASIGNATURA: INTELIGENCIA ARTIFICIAL II
DOCENTE: AMAURY GIOVANNI MÉNDEZ AGUIRRE
SESIÓN 6: EXTRACCIÓN DE CARACTERÍSTICAS Y CONTORNOS
===============================================================================

===============================================================================
SECCIÓN 1: RESOLUCIÓN Y CÁLCULOS DEL TALLER ANALÍTICO
===============================================================================

-------------------------------------------------------------------------------
TALLER ANALÍTICO: BOUNDING BOX (CAJA DELIMITADORA)
-------------------------------------------------------------------------------
Dado un contorno definido por las coordenadas de 4 esquinas de un objeto irregular:
A = (2, 4)
B = (8, 2)
C = (10, 7)
D = (3, 9)

1. Determinación de Coordenadas Extremas (X_min, Y_min, X_max, Y_max):
   - X_min = min(2, 8, 10, 3) = 2
   - Y_min = min(4, 2, 7, 9)  = 2
   - X_max = max(2, 8, 10, 3) = 10
   - Y_max = max(4, 2, 7, 9)  = 9

   RESPUESTA 1:
   - Coordenadas extremas: (X_min=2, Y_min=2, X_max=10, Y_max=9)
   - Esquina superior izquierda de la caja: (2, 2)
   - Esquina inferior derecha de la caja: (10, 9)

2. Cálculo del Ancho (W) y Alto (H) del Bounding Box:
   - Ancho (W) = X_max - X_min = 10 - 2 = 8 unidades
   - Alto (H)  = Y_max - Y_min = 9 - 2  = 7 unidades

   RESPUESTA 2:
   - Ancho (W) = 8
   - Alto (H) = 7
   - El área del Bounding Box envolvente es W * H = 8 * 7 = 56 unidades cuadradas.
"""

import cv2
import numpy as np

# ==========================================
# SECCIÓN 2: DEMOSTRACIÓN ALGORÍTMICA DEL BOUNDING BOX (TALLER ANALÍTICO)
# ==========================================
print("=========================================================")
print("TALLER ANALÍTICO: VERIFICACIÓN DE BOUNDING BOX")
print("=========================================================")

# 1. Matriz de coordenadas del polígono
puntos_contorno = np.array([[2, 4], [8, 2], [10, 7], [3, 9]], dtype=np.int32) # [Bloque 1: Coordenadas]

# 2. Obtención de extremos y dimensiones mediante OpenCV
x_min, y_min, w_calc, h_calc = cv2.boundingRect(puntos_contorno) # [Bloque 2: Bounding Box]

print(f"Puntos del Contorno: {puntos_contorno.tolist()}")
print(f"X_min: {x_min}, Y_min: {y_min}")
print(f"X_max: {x_min + w_calc}, Y_max: {y_min + h_calc}")
print(f"Ancho (W) Calculado: {w_calc}")
print(f"Alto (H) Calculado:  {h_calc}")


# ==========================================
# SECCIÓN 3: TALLER DE LABORATORIO (CLASIFICADOR DE OBJETOS POR ÁREA)
# ==========================================
print("\n=========================================================")
print("TALLER DE LABORATORIO: PIPELINE INTEGRADOR Y CLASIFICADOR")
print("=========================================================")

# 1. Generación de Imagen Sintética con Objetos de Diferentes Tamaños (Monedas/Herramientas)
np.random.seed(42) # [Bloque 3: Semilla]
imagen_color = np.full((400, 500, 3), 220, dtype=np.uint8) # [Bloque 4: Fondo Claro] - Lienzo a color (BGR)

# Dibujar Objeto 1: Círculo Pequeño (Moneda chica / Arandela)
cv2.circle(imagen_color, (100, 150), 25, (40, 40, 40), -1)

# Dibujar Objeto 2: Círculo Grande (Moneda grande / Tuerca)
cv2.circle(imagen_color, (350, 150), 55, (40, 40, 40), -1)

# Dibujar Objeto 3: Rectángulo Pequeño
cv2.rectangle(imagen_color, (80, 280), (140, 330), (40, 40, 40), -1)

# Dibujar Objeto 4: Rectángulo Grande
cv2.rectangle(imagen_color, (280, 260), (420, 360), (40, 40, 40), -1)

# 2. PIPELINE PASO 1: Conversión a escala de grises
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY) # [Bloque 5: Grises]

# 3. PIPELINE PASO 2: Umbralización (Binarización para aislar objetos oscuros)
# Al ser el fondo claro y los objetos oscuros, se usa THRESH_BINARY_INV
_, imagen_binaria = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY_INV) # [Bloque 6: Binarización]

# 4. PIPELINE PASO 3: Limpieza Morfológica (Apertura)
kernel_3x3 = np.ones((3, 3), dtype=np.uint8) # [Bloque 7: Kernel]
binaria_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel_3x3) # [Bloque 7: Morfología]

# 5. PIPELINE PASO 4: Detección de Contornos
contornos, _ = cv2.findContours(binaria_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # [Bloque 8: Contornos]

# Umbral de decisión lógica de negocio (X píxeles)
UMBRAL_AREA_GRANDE = 3500 # [Bloque 9: Umbral de Área]

print(f"Total de objetos detectados: {len(contornos)}\n")

# 6. PIPELINE PASO 5: Extracción de Métricas y Clasificación
for idx, cnt in enumerate(contornos, start=1):
    # A) Área
    area = cv2.contourArea(cnt) # [Bloque 10: Área]
    
    # B) Bounding Box
    x, y, w, h = cv2.boundingRect(cnt) # [Bloque 10: Bounding Box]
    
    # C) Centroide (Momentos)
    M = cv2.moments(cnt) # [Bloque 10: Momentos]
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
        
    # Lógica de Clasificación
    if area > UMBRAL_AREA_GRANDE:
        color_caja = (255, 0, 0) # BGR: Azul para Objeto GRANDE
        categoria = "GRANDE"
    else:
        color_caja = (0, 0, 255) # BGR: Rojo para Objeto PEQUEÑO
        categoria = "PEQUEÑO"
        
    # Dibujar Bounding Box
    cv2.rectangle(imagen_color, (x, y), (x + w, y + h), color_caja, 2)
    
    # Dibujar Centroide
    cv2.circle(imagen_color, (cx, cy), 4, (0, 255, 0), -1)
    
    # Etiquetar en la imagen
    cv2.putText(imagen_color, f"{categoria}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja, 2)
                
    print(f"Objeto #{idx}: Área = {area:.1f} px² | Clasificación = {categoria} | Ubicación = ({cx}, {cy})")

# 7. Exportación de Resultados
cv2.imwrite("1_escena_binarizada.png", binaria_limpia)
cv2.imwrite("2_clasificacion_objetos_final.png", imagen_color)

print("\nProcesamiento completado con éxito.")
print("Archivos exportados: '1_escena_binarizada.png' y '2_clasificacion_objetos_final.png'.")

"""
===============================================================================
CONCLUSIONES TÉCNICAS DEL TALLER 6
===============================================================================
1. Eficiencia de los Descriptores Geométricos:
   La extracción de métricas como el Área y el Bounding Box permite reducir una 
   imagen de miles de píxeles a un conjunto estructurado de características numéricas 
   vectoriales (Features), óptimo para alimentar clasificadores de Machine Learning.

2. Robustez de la Separación Estructural:
   El pipeline completo (Grises -> Umbralización -> Morfología -> Contornos) garantiza 
   que los objetos se aislen correctamente antes del cálculo de momentos, evitando que 
   el ruido modifique erróneamente la ubicación del centroide (Cx, Cy) o la escala del área.
"""