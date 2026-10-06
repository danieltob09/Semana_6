import os
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

import keras_hub


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "imagenes" / "San_F.jpg"
OUTPUT_DIR = BASE_DIR / "resultados"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def guardar_imagen(nombre, img):
    """
    Guarda una imagen en la carpeta resultados.
    """
    ruta = OUTPUT_DIR / nombre
    cv2.imwrite(str(ruta), img)
    return ruta


def medir_error(imagen_original, imagen_procesada):
    """
    Calcula MSE y MAE entre dos imágenes.
    """
    original = imagen_original.astype(np.float32)
    procesada = imagen_procesada.astype(np.float32)

    mse = np.mean((original - procesada) ** 2)
    mae = np.mean(np.abs(original - procesada))

    return mse, mae


# ============================================================
# ENCABEZADO
# ============================================================

print("=" * 65)
print("   PROYECTO: VEHÍCULOS AUTÓNOMOS")
print("   Procesamiento y segmentación de imágenes")
print("=" * 65)


# ============================================================
# 1. HARDWARE
# ============================================================

print("\n[1/10] Comprobando hardware...")

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detectada:")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("No se detectó GPU.")
    print("TensorFlow utilizará CPU.")

print(f"TensorFlow: {tf.__version__}")


# ============================================================
# 2. VERIFICAR IMAGEN
# ============================================================

print("\n[2/10] Verificando imagen...")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró la imagen:\n{IMAGE_PATH}"
    )

print(f"Imagen encontrada: {IMAGE_PATH}")


# ============================================================
# 3. CARGAR IMAGEN
# ============================================================

print("\n[3/10] Cargando imagen...")

img_bgr = cv2.imread(str(IMAGE_PATH))

if img_bgr is None:
    raise ValueError("OpenCV no pudo cargar la imagen.")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Redimensionamiento para MobileNetV2
img_rgb_224 = cv2.resize(
    img_rgb,
    (224, 224),
    interpolation=cv2.INTER_AREA
)

print(f"Dimensiones: {img_rgb_224.shape}")
print(f"Tipo de datos: {img_rgb_224.dtype}")

guardar_imagen(
    "01_imagen_original.jpg",
    cv2.cvtColor(img_rgb_224, cv2.COLOR_RGB2BGR)
)

print("Imagen original guardada.")
# ============================================================
# 4. REDUCCIÓN DE RUIDO
# ============================================================

print("\n[4/10] Aplicando filtrado de reducción de ruido...")

inicio = time.time()

imagen_filtrada = cv2.bilateralFilter(
    img_rgb_224,
    d=9,
    sigmaColor=75,
    sigmaSpace=75
)

tiempo_filtrado = time.time() - inicio

guardar_imagen(
    "02_imagen_filtrada.jpg",
    cv2.cvtColor(imagen_filtrada, cv2.COLOR_RGB2BGR)
)

print(f"Filtrado completado en {tiempo_filtrado:.2f} segundos.")


# ============================================================
# 5. CLAHE
# ============================================================

print("\n[5/10] Mejorando contraste e iluminación...")

inicio = time.time()

imagen_lab = cv2.cvtColor(
    imagen_filtrada,
    cv2.COLOR_RGB2LAB
)

l_channel, a_channel, b_channel = cv2.split(imagen_lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

l_mejorado = clahe.apply(l_channel)

imagen_mejorada_lab = cv2.merge(
    [l_mejorado, a_channel, b_channel]
)

imagen_mejorada = cv2.cvtColor(
    imagen_mejorada_lab,
    cv2.COLOR_LAB2RGB
)

tiempo_clahe = time.time() - inicio

guardar_imagen(
    "03_imagen_mejorada.jpg",
    cv2.cvtColor(imagen_mejorada, cv2.COLOR_RGB2BGR)
)

print(
    f"Mejoramiento completado en "
    f"{tiempo_clahe:.2f} segundos."
)


# ============================================================
# 6. MOBILENETV2
# ============================================================

print("\n[6/10] Cargando MobileNetV2...")

inicio = time.time()

mobilenet = MobileNetV2(
    weights="imagenet"
)

entrada_mobilenet = np.expand_dims(
    imagen_mejorada.astype(np.float32),
    axis=0
)

entrada_mobilenet = preprocess_input(
    entrada_mobilenet
)

predicciones = mobilenet.predict(
    entrada_mobilenet,
    verbose=0
)

tiempo_mobilenet = time.time() - inicio

resultados = decode_predictions(
    predicciones,
    top=5
)[0]

print(
    f"Clasificación completada en "
    f"{tiempo_mobilenet:.2f} segundos."
)

print("\nPrincipales predicciones de MobileNetV2:")

for _, etiqueta, probabilidad in resultados:
    print(
        f"  {etiqueta:<30} "
        f"{probabilidad * 100:6.2f}%"
    )


# ============================================================
# 7. DEEPLABV3+
# ============================================================

print("\n[7/10] Cargando modelo DeepLabV3+...")

inicio = time.time()

segmentation_model = (
    keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        "deeplab_v3_plus_resnet50_pascalvoc"
    )
)

tiempo_carga_modelo = time.time() - inicio

print(
    f"Modelo DeepLabV3+ cargado en "
    f"{tiempo_carga_modelo:.2f} segundos."
)
# ============================================================
# PREPARACIÓN PARA SEGMENTACIÓN
# ============================================================

print("Preparando imagen para segmentación...")

SEGMENT_SIZE = 512

imagen_segmentacion = cv2.resize(
    imagen_mejorada,
    (SEGMENT_SIZE, SEGMENT_SIZE),
    interpolation=cv2.INTER_AREA
)

entrada_segmentacion = (
    imagen_segmentacion.astype(np.float32)
)

# El modelo espera un lote
entrada_segmentacion = np.expand_dims(
    entrada_segmentacion,
    axis=0
)


# ============================================================
# INFERENCIA
# ============================================================

print("Ejecutando segmentación...")

inicio = time.time()

pred_segmentacion = segmentation_model.predict(
    entrada_segmentacion,
    verbose=0
)

tiempo_segmentacion = time.time() - inicio

print(
    f"Segmentación completada en "
    f"{tiempo_segmentacion:.2f} segundos."
)

print(
    f"Forma de salida del modelo: "
    f"{pred_segmentacion.shape}"
)


# ============================================================
# OBTENER MAPA DE CLASES
# ============================================================

if pred_segmentacion.ndim == 4:

    mapa_clases = np.argmax(
        pred_segmentacion,
        axis=-1
    )[0]

elif pred_segmentacion.ndim == 3:

    mapa_clases = np.argmax(
        pred_segmentacion,
        axis=-1
    )[0]

else:
    raise ValueError(
        "Formato inesperado en la salida de DeepLabV3+."
    )


# ============================================================
# REDIMENSIONAR MAPA
# ============================================================

mapa_clases = cv2.resize(
    mapa_clases.astype(np.uint8),
    (img_rgb_224.shape[1], img_rgb_224.shape[0]),
    interpolation=cv2.INTER_NEAREST
)


# ============================================================
# CREAR VISUALIZACIÓN DE SEGMENTACIÓN
# ============================================================

# Normalización visual del mapa
num_clases = max(
    int(np.max(mapa_clases)) + 1,
    1
)

mapa_visual = (
    mapa_clases.astype(np.float32)
    / max(num_clases - 1, 1)
    * 255
).astype(np.uint8)

mapa_color = cv2.applyColorMap(
    mapa_visual,
    cv2.COLORMAP_JET
)

mapa_color_rgb = cv2.cvtColor(
    mapa_color,
    cv2.COLOR_BGR2RGB
)

guardar_imagen(
    "04_mapa_segmentacion.jpg",
    mapa_color
)


# ============================================================
# SUPERPOSICIÓN
# ============================================================

overlay = cv2.addWeighted(
    img_rgb_224,
    0.60,
    mapa_color_rgb,
    0.40,
    0
)

guardar_imagen(
    "05_segmentacion_superpuesta.jpg",
    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
)

print("Mapa de segmentación guardado.")
print("Superposición guardada.")


# ============================================================
# 8. COMPARACIÓN VISUAL
# ============================================================

print("\n[8/10] Generando comparación visual...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb_224)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(imagen_mejorada)
plt.title("Imagen mejorada")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Segmentación DeepLabV3+")
plt.axis("off")

plt.tight_layout()

comparacion_path = OUTPUT_DIR / "06_comparacion.jpg"

plt.savefig(
    comparacion_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(f"Comparación guardada: {comparacion_path}")


# ============================================================
# 9. MÉTRICAS
# ============================================================

print("\n[9/10] Calculando métricas...")

mse, mae = medir_error(
    img_rgb_224,
    imagen_mejorada
)

print(f"MSE entre original y mejorada: {mse:.4f}")
print(f"MAE entre original y mejorada: {mae:.4f}")


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

valores, cantidades = np.unique(
    mapa_clases,
    return_counts=True
)

total_pixeles = mapa_clases.size

print("\nDistribución de clases detectadas:")

for valor, cantidad in zip(valores, cantidades):

    porcentaje = (
        cantidad / total_pixeles
    ) * 100

    print(
        f"  Clase {int(valor):2d}: "
        f"{porcentaje:6.2f}%"
    )

# ============================================================
# 10. RESUMEN
# ============================================================

print("\n[10/10] Proceso finalizado.")

print("\n" + "=" * 65)
print("RESULTADOS")
print("=" * 65)

print(f"Imagen original:")
print(f"  {OUTPUT_DIR / '01_imagen_original.jpg'}")

print(f"\nImagen filtrada:")
print(f"  {OUTPUT_DIR / '02_imagen_filtrada.jpg'}")

print(f"\nImagen mejorada:")
print(f"  {OUTPUT_DIR / '03_imagen_mejorada.jpg'}")

print(f"\nMapa de segmentación:")
print(f"  {OUTPUT_DIR / '04_mapa_segmentacion.jpg'}")

print(f"\nSegmentación superpuesta:")
print(f"  {OUTPUT_DIR / '05_segmentacion_superpuesta.jpg'}")

print(f"\nComparación:")
print(f"  {OUTPUT_DIR / '06_comparacion.jpg'}")

print("\n" + "=" * 65)
print("PROCESAMIENTO COMPLETADO CORRECTAMENTE")
print("=" * 65)
