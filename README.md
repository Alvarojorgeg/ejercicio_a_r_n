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

La primera vez que se accede a `http://127.0.0.1:5000` se entrena el modelo. El sitio muestra la arquitectura interpretada,
los logs de entrenamiento, el resumen del modelo y ejemplos de predicciones con dígitos reales.
