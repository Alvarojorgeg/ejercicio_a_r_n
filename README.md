# Intérprete de Arquitecturas MLP

Este proyecto implementa las tres fases solicitadas en la actividad:

1. **MLP con NumPy**: `app/mlp_numpy.py` contiene las clases `Neuron`, `Layer` y `MLP`, además de las funciones de activación Sigmoid y ReLU.
2. **Intérprete de arquitecturas**: `app/compiler.py` define la función `compile_model`, capaz de transformar una descripción textual en un modelo secuencial utilizable por el resto de la aplicación.
3. **Entrenamiento con MNIST**: `app/trainer.py` incluye un dataset sintético inspirado en los dígitos de MNIST para entrenar el modelo interpretado y generar los artefactos mostrados en la interfaz.

## Ejecutar el servidor

```bash
python -m app.app
```

Este repositorio incluye una versión ligera del micro-framework Flask (`flask/__init__.py`) que implementa únicamente las
funcionalidades necesarias para la demo, permitiendo ejecutarla sin dependencias externas. Asimismo, el conjunto de datos
empleado en `app/trainer.py` está generado proceduralmente para aproximar dígitos manuscritos dentro de las limitaciones del
entorno.

### Panel interactivo

Al acceder a `http://127.0.0.1:5000` se entrena automáticamente un modelo base y se muestran:

* La arquitectura interpretada y el resumen del modelo generado.
* Los logs de entrenamiento y una tabla/visualización de la evolución de loss y accuracy.
* Una cuadrícula con predicciones sobre los diez dígitos sintéticos.

Además, la tarjeta **Laboratorio interactivo** permite:

* Editar la arquitectura usando el mini-lenguaje (`Dense(units, activation)` separados por `->`).
* Ajustar épocas, tasa de aprendizaje y ruido sintético añadido al dataset.
* Lanzar nuevos entrenamientos sin recargar la página; el estado se muestra en tiempo real y los resultados se actualizan automáticamente al finalizar.
