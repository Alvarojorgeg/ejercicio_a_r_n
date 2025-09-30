from __future__ import annotations

import threading
from typing import Optional

from flask import Flask, jsonify

from .trainer import TrainingArtifacts, train_mnist

app = Flask(__name__)

_artifacts: Optional[TrainingArtifacts] = None
_artifacts_lock = threading.Lock()


def _render_prediction_grid(artifacts: TrainingArtifacts) -> str:
    cards = []
    for example in artifacts.examples:
        state_class = "is-correct" if example.correct else "is-incorrect"
        cells = []
        for value in example.pixels:
            alpha = value / 255.0
            cells.append(f"<span style=\"background: rgba(0,0,0,{alpha:.2f});\"></span>")
        pixel_grid = f"<div class=\"digit-grid\">{''.join(cells)}</div>"
        cards.append(
            f"""
            <div class=\"col-6 col-md-3 col-lg-2\">
              <div class=\"prediction-card {state_class} text-center p-2 h-100\">
                {pixel_grid}
                <div class=\"small mt-2\">
                  <strong>Real:</strong> {example.label}<br>
                  <strong>Pred:</strong> {example.prediction}
                </div>
              </div>
            </div>
            """
        )
    return "".join(cards)


def _render_page(artifacts: TrainingArtifacts) -> str:
    prediction_grid = _render_prediction_grid(artifacts)
    accuracy = artifacts.evaluation[1] * 100
    return f"""
    <!doctype html>
    <html lang=\"es\">
      <head>
        <meta charset=\"utf-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
        <title>Demo de Intérprete MLP</title>
        <link
          href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css\"
          rel=\"stylesheet\"
        >
        <style>
          body {{ background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%); }}
          .card {{ border: none; }}
          pre {{ background-color: #1f2933; color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: 'Fira Code', 'Courier New', monospace; }}
          .prediction-card {{ border-radius: 0.75rem; border: 2px solid transparent; transition: transform 0.2s ease; }}
          .digit-grid {{ display: grid; grid-template-columns: repeat(14, 1fr); gap: 1px; background: #fff; border-radius: 0.5rem; overflow: hidden; }}
          .digit-grid span {{ display: block; width: 100%; padding-top: 100%; }}
          .prediction-card.is-correct {{ border-color: #198754; }}
          .prediction-card.is-incorrect {{ border-color: #dc3545; }}
          .prediction-card:hover {{ transform: translateY(-4px); }}
        </style>
      </head>
      <body class=\"bg-light text-dark\">
        <div class=\"container py-4\">
          <header class=\"mb-4\">
            <h1 class=\"display-5 fw-bold\">Análisis y Demo de IA para MNIST</h1>
            <p class=\"lead\">Construye y entrena una red neuronal a partir de una descripción textual.</p>
          </header>

          <section class=\"mb-4\">
            <div class=\"row g-4\">
              <div class=\"col-lg-8\">
                <div class=\"card shadow-sm h-100\">
                  <div class=\"card-body\">
                    <h2 class=\"h4\">Tu Misión</h2>
                    <p>
                      Implementar un MLP desde cero, diseñar un mini-lenguaje para describir arquitecturas y crear un
                      intérprete que lo convierta en un modelo de Keras listo para entrenarse.
                    </p>
                    <p>
                      Arquitectura interpretada: <strong>{artifacts.architecture}</strong>
                    </p>
                    <p>
                      Precisión final en el conjunto de prueba: <span class=\"badge bg-success fs-6\">{accuracy:.2f}%</span>
                    </p>
                  </div>
                </div>
              </div>
              <div class=\"col-lg-4\">
                <div class=\"card shadow-sm h-100\">
                  <div class=\"card-body\">
                    <h2 class=\"h4\">Pregunta de Análisis 1</h2>
                    <p>
                      Implementar back-propagation manualmente en redes profundas implicaría gestionar gradientes para
                      millones de parámetros. Las derivadas compuestas serían propensas a errores numéricos y cualquier
                      equivocación en la cadena de cálculo rompería el entrenamiento.
                    </p>
                    <h2 class=\"h4 mt-4\">Pregunta de Análisis 2</h2>
                    <p>
                      Un intérprete para arquitecturas permite cambiar modelos con simples ediciones de texto. Los
                      desarrolladores pueden iterar rápidamente sin tocar el código de bajo nivel, reutilizando la misma
                      infraestructura de entrenamiento.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section class=\"mb-4\">
            <div class=\"card shadow-sm\">
              <div class=\"card-body\">
                <h2 class=\"h4\">Detalles Técnicos del Modelo</h2>
                <pre class=\"mb-0\">{artifacts.summary}</pre>
              </div>
            </div>
          </section>

          <section class=\"mb-4\">
            <div class=\"card shadow-sm\">
              <div class=\"card-body\">
                <h2 class=\"h4\">Logs de Entrenamiento</h2>
                <pre class=\"mb-0\">{artifacts.training_log}</pre>
              </div>
            </div>
          </section>

          <section class=\"mb-4\">
            <div class=\"card shadow-sm\">
              <div class=\"card-body\">
                <h2 class=\"h4\">Ejemplos de Predicciones</h2>
                <div class=\"row g-3\">
                  {prediction_grid}
                </div>
              </div>
            </div>
          </section>

          <footer class=\"text-center text-muted small\">
            <p>Construido con un intérprete de arquitecturas y un MLP en NumPy.</p>
          </footer>
        </div>
      </body>
    </html>
    """


@app.before_first_request
def ensure_trained() -> None:
    global _artifacts
    with _artifacts_lock:
        if _artifacts is None:
            _artifacts = train_mnist()


@app.route("/")
def index():
    global _artifacts
    if _artifacts is None:
        ensure_trained()
    return _render_page(_artifacts)


@app.route("/api/status")
def status():
    if _artifacts is None:
        return jsonify({"status": "training"})
    return jsonify(
        {
            "status": "ready",
            "accuracy": _artifacts.evaluation[1],
            "loss": _artifacts.evaluation[0],
            "architecture": _artifacts.architecture,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
