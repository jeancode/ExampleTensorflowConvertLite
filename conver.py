import tensorflow as tf

# Cargar el modelo desde el archivo .h5
modelo = tf.keras.models.load_model('modelo_conversion.h5')

# Convertir el modelo a formato TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = converter.convert()

# Guardar el modelo TFLite en un archivo .tflite
with open('modelo_conversion.tflite', 'wb') as f:
    f.write(modelo_tflite)

print("Modelo TFLite guardado como 'modelo_conversion.tflite'.")