import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Adam

tf.config.threading.set_intra_op_parallelism_threads(tf.config.experimental.get_max_used_device().num_cores)

# Generar 1200 datos de temperatura en grados Celsius
n_datos = 500
celsius = np.linspace(-300, 1000, n_datos)

# Calcular los valores en grados Fahrenheit utilizando la fórmula de conversión
fahrenheit = (celsius * 1.8) + 32

# Crear el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, input_shape=[1], activation='relu'),  # Capa de entrada
    
    tf.keras.layers.Dense(units=500, activation='relu'),  # Capa oculta 1    
    

    tf.keras.layers.Dense(units=1)  # Capa de salida
])

# Compilar el modelo


# Especificar una tasa de aprendizaje
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
#modelo.compile(optimizer='adam', loss='mean_squared_error')

#modelo.compile(optimizer=optimizer, loss='mean_squared_error')

# Cargar el modelo desde el archivo .h5
#modelo = tf.keras.models.load_model('modelo_conversion.h5')

# Cargar el modelo previamente entrenado
modelo = tf.keras.models.load_model('modelo_conversion.h5')

# Entrenar el modelo
historial = modelo.fit(celsius, fahrenheit, epochs=10000, verbose=True)
print("Modelo entrenado.")

# Evaluar el modelo con datos de prueba
puntuacion = modelo.evaluate(celsius, fahrenheit)
print('Pérdida:', puntuacion)

# Realizar una predicción
resultado = modelo.predict([100.0])
print('Resultado de la predicción (100 grados Celsius):', resultado)

modelo.save('modelo_conversion.h5')



# Cargar el modelo desde el archivo .h5
modelo = tf.keras.models.load_model('modelo_conversion.h5')

# Convertir el modelo a formato TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = converter.convert()

# Guardar el modelo TFLite en un archivo .tflite
with open('modelo_conversion.tflite', 'wb') as f:
    f.write(modelo_tflite)

print("Modelo TFLite guardado como 'modelo_conversion.tflite'.")


import paramiko

# Configura los parámetros de conexión SSH
host = '192.168.137.130'
port = 22  # Puerto SSH por defecto
username = 'root'
password = '5an32sjr'

# Ruta local del archivo que deseas transferir
archivo_local = 'modelo_conversion.tflite'

# Ruta de destino en el dispositivo remoto
ruta_remota = '/root/tensor/example1/'

# Crea una instancia de cliente SSH
ssh_client = paramiko.SSHClient()

try:
    # Acepta automáticamente la clave SSH del dispositivo
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Conecta al dispositivo SSH
    ssh_client.connect(hostname=host, port=port, username=username, password=password)

    # Crea un canal SFTP para la transferencia de archivos
    sftp_client = ssh_client.open_sftp()

    # Transfiere el archivo al dispositivo remoto
    sftp_client.put(archivo_local, ruta_remota + archivo_local)

    print("Archivo transferido con éxito.")

except Exception as e:
    print(f"Error: {str(e)}")

finally:
    # Cierra la conexión SSH
    ssh_client.close()