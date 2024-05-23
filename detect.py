# Importar librerías
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO('best.pt')

# Leer la imagen
cap = cv2.imread('chess-0004-1698147135687.png')

# Realizar la predicción
resultados = model.predict(cap, imgsz=640)

coordenadas = []
# Obtener las coordenadas de las cajas detectadas

cajas = resultados[0].boxes.cpu().numpy()
coordenadas = cajas.xyxy

masks = resultados[0].masks
coord = masks.xy
print(coord)

x1, y1, x2, y2 = coordenadas[0]
img = cap[int(y1):int(y2), int(x1):int(x2)]

# Mostrar la imagen con las anotaciones
anotaciones = resultados[0].plot()
cv2.imshow("Detección y segmentación", anotaciones)
cv2.waitKey(0)
cv2.imshow("Imagen cortada", img)
cv2.waitKey(0)
