from __future__ import annotations

import json
import threading
from typing import Dict, Optional

from flask import Flask, jsonify, request

from .trainer import DEFAULT_ARCHITECTURE, TrainingArtifacts, train_mnist

app = Flask(__name__)

_artifacts: Optional[TrainingArtifacts] = None
_artifacts_lock = threading.Lock()
_training_thread: Optional[threading.Thread] = None
_training_status: Dict[str, object] = {
    "state": "idle",
    "message": "Modelo listo para entrenarse.",
}


def _serialize_artifacts(artifacts: TrainingArtifacts) -> Dict[str, object]:
    history = [
        {"epoch": idx + 1, "loss": loss, "accuracy": acc}
        for idx, (loss, acc) in enumerate(artifacts.history)
    ]
    return {
        "architecture": artifacts.architecture,
        "summary": artifacts.summary,
        "training_log": artifacts.training_log,
        "evaluation": {"loss": artifacts.evaluation[0], "accuracy": artifacts.evaluation[1]},
        "history": history,
        "examples": [
            {
                "pixels": example.pixels,
                "label": example.label,
                "prediction": example.prediction,
                "correct": example.correct,
            }
            for example in artifacts.examples
        ],
        "epochs": artifacts.epochs,
        "learning_rate": artifacts.learning_rate,
        "noise": artifacts.noise,
    }


def _start_training(architecture: str, epochs: int, learning_rate: float, noise: float) -> bool:
    global _training_thread, _training_status
    if _training_thread and _training_thread.is_alive():
        return False

    def worker() -> None:
        global _artifacts, _training_status
        _training_status = {
            "state": "running",
            "message": "Entrenando red neuronal...",
            "request": {
                "architecture": architecture,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "noise": noise,
            },
        }
        try:
            artifacts = train_mnist(
                architecture=architecture,
                epochs=epochs,
                learning_rate=learning_rate,
                noise=noise,
            )
            with _artifacts_lock:
                _artifacts = artifacts
            _training_status = {
                "state": "completed",
                "message": "Entrenamiento completado.",
                "result": {
                    "accuracy": artifacts.evaluation[1],
                    "loss": artifacts.evaluation[0],
                    "epochs": len(artifacts.history),
                },
            }
        except Exception as exc:  # pragma: no cover - debug helper
            _training_status = {
                "state": "error",
                "message": str(exc),
            }

    _training_thread = threading.Thread(target=worker, daemon=True)
    _training_thread.start()
    return True


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
    payload = json.dumps(_serialize_artifacts(artifacts), ensure_ascii=False)
    status_payload = json.dumps(_training_status, ensure_ascii=False)
    default_arch_json = json.dumps(DEFAULT_ARCHITECTURE)
    script_template = """
        <script>
          const DEFAULT_ARCHITECTURE = __DEFAULT_ARCH__;
          const INITIAL_ARTIFACTS = __PAYLOAD__;
          const INITIAL_STATUS = __STATUS__;

          const statusAlert = document.getElementById('status-alert');
          const statusChip = document.getElementById('status-chip');
          const trainForm = document.getElementById('training-form');
          const trainButton = document.getElementById('train-button');
          const resetButton = document.getElementById('reset-button');
          const architectureInput = document.getElementById('architecture-input');
          const epochsInput = document.getElementById('epochs-input');
          const learningRateInput = document.getElementById('learning-rate-input');
          const noiseInput = document.getElementById('noise-input');
          const noiseValue = document.getElementById('noise-value');
          const architectureDisplay = document.getElementById('architecture-display');
          const accuracyBadge = document.getElementById('accuracy-badge');
          const lossBadge = document.getElementById('loss-badge');
          const summaryBlock = document.getElementById('summary-block');
          const trainingLogBlock = document.getElementById('training-log');
          const predictionsGrid = document.getElementById('predictions-grid');
          const historyTable = document.getElementById('history-table');

          const defaultConfig = {
            architecture: INITIAL_ARTIFACTS.architecture || DEFAULT_ARCHITECTURE,
            epochs: INITIAL_ARTIFACTS.epochs || 20,
            learning_rate: INITIAL_ARTIFACTS.learning_rate || 0.05,
            noise: INITIAL_ARTIFACTS.noise || 0,
          };

          let lastState = 'idle';

          function setStatus(state, message) {
            statusAlert.className = 'alert mt-3';
            statusAlert.classList.add('alert-' + (state === 'error' ? 'danger' : state === 'running' ? 'warning' : state === 'completed' ? 'success' : 'secondary'));
            statusAlert.textContent = message;
            statusAlert.classList.remove('d-none');
            statusChip.className = 'status-chip status-' + state;
            statusChip.textContent = 'Estado: ' + state.charAt(0).toUpperCase() + state.slice(1);
          }

          function clearStatus() {
            statusAlert.classList.add('d-none');
            statusChip.className = 'status-chip status-idle';
            statusChip.textContent = 'Estado: Idle';
          }

          function setFormDisabled(disabled) {
            [architectureInput, epochsInput, learningRateInput, noiseInput, trainButton, resetButton].forEach((el) => {
              if (el) {
                el.disabled = disabled && el !== resetButton;
              }
            });
            if (resetButton) {
              resetButton.disabled = disabled;
            }
          }

          function buildPredictionCard(example) {
            const cells = example.pixels
              .map((value) => {
                const alpha = Math.max(0, Math.min(1, value / 255));
                return `<span style=\"background: rgba(0,0,0,${alpha.toFixed(2)})\"></span>`;
              })
              .join('');
            const stateClass = example.correct ? 'is-correct' : 'is-incorrect';
            return `
              <div class=\"col-6 col-md-3 col-lg-2\">
                <div class=\"prediction-card ${stateClass} text-center p-2 h-100\">
                  <div class=\"digit-grid\">${cells}</div>
                  <div class=\"small mt-2\">
                    <strong>Real:</strong> ${example.label}<br>
                    <strong>Pred:</strong> ${example.prediction}
                  </div>
                </div>
              </div>
            `;
          }

          function renderHistory(history) {
            historyTable.innerHTML = history
              .map((row) => `<tr><td>${row.epoch}</td><td>${row.loss.toFixed(4)}</td><td>${(row.accuracy * 100).toFixed(2)}%</td></tr>`)
              .join('');
          }

          function renderArtifacts(data) {
            if (!data) return;
            architectureDisplay.textContent = data.architecture;
            accuracyBadge.textContent = `${(data.evaluation.accuracy * 100).toFixed(2)}%`;
            lossBadge.textContent = data.evaluation.loss.toFixed(4);
            summaryBlock.textContent = data.summary;
            trainingLogBlock.textContent = data.training_log;
            if (Array.isArray(data.examples)) {
              predictionsGrid.innerHTML = data.examples.map(buildPredictionCard).join('');
            }
            if (Array.isArray(data.history)) {
              renderHistory(data.history);
              updateChart(data.history);
            }
          }

          const historyCtx = document.getElementById('history-chart');
          const initialHistory = Array.isArray(INITIAL_ARTIFACTS.history) ? INITIAL_ARTIFACTS.history : [];
          const historyChart = new Chart(historyCtx, {
            type: 'line',
            data: {
              labels: initialHistory.map((row) => `Ep ${row.epoch}`),
              datasets: [
                {
                  label: 'Precisión (%)',
                  data: initialHistory.map((row) => row.accuracy * 100),
                  borderColor: '#198754',
                  backgroundColor: 'rgba(25, 135, 84, 0.1)',
                  tension: 0.35,
                  fill: true,
                  yAxisID: 'accuracy',
                },
                {
                  label: 'Pérdida',
                  data: initialHistory.map((row) => row.loss),
                  borderColor: '#0d6efd',
                  backgroundColor: 'rgba(13, 110, 253, 0.1)',
                  tension: 0.35,
                  fill: true,
                  yAxisID: 'loss',
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              interaction: { intersect: false, mode: 'index' },
              scales: {
                accuracy: {
                  type: 'linear',
                  position: 'left',
                  beginAtZero: true,
                  suggestedMax: 100,
                  ticks: { callback: (value) => `${value}%` },
                },
                loss: {
                  type: 'linear',
                  position: 'right',
                  beginAtZero: true,
                },
              },
              plugins: {
                legend: { display: true },
              },
            },
          });

          function updateChart(history) {
            historyChart.data.labels = history.map((row) => `Ep ${row.epoch}`);
            historyChart.data.datasets[0].data = history.map((row) => row.accuracy * 100);
            historyChart.data.datasets[1].data = history.map((row) => row.loss);
            historyChart.update();
          }

          async function fetchArtifacts() {
            try {
              const response = await fetch('/api/artifacts');
              if (!response.ok) return;
              const data = await response.json();
              renderArtifacts(data);
            } catch (error) {
              console.error('No se pudo obtener el modelo actualizado', error);
            }
          }

          async function pollStatus() {
            try {
              const response = await fetch('/api/status');
              if (!response.ok) return;
              const status = await response.json();
              handleStatus(status);
            } catch (error) {
              console.error('Error consultando estado', error);
            }
          }

          function handleStatus(status) {
            if (!status || !status.state) {
              clearStatus();
              lastState = 'idle';
              return;
            }
            if (status.state === 'idle') {
              clearStatus();
              lastState = 'idle';
              setFormDisabled(false);
              return;
            }
            if (status.state === lastState && status.state !== 'running') {
              return;
            }
            lastState = status.state;
            setStatus(status.state, status.message || '');
            if (status.state === 'running') {
              setFormDisabled(true);
              return;
            }
            setFormDisabled(false);
            if (status.state === 'completed') {
              fetchArtifacts();
              setTimeout(() => clearStatus(), 4000);
            }
            if (status.state === 'error') {
              setTimeout(() => clearStatus(), 6000);
            }
          }

          trainForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const payload = {
              architecture: architectureInput.value.trim(),
              epochs: parseInt(epochsInput.value, 10) || defaultConfig.epochs,
              learning_rate: parseFloat(learningRateInput.value) || defaultConfig.learning_rate,
              noise: parseFloat(noiseInput.value) || 0,
            };
            try {
              const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
              });
              const data = await response.json();
              if (response.ok) {
                setStatus('running', data.message || 'Entrenando red neuronal...');
                setFormDisabled(true);
              } else {
                setStatus('error', data.message || 'No se pudo iniciar el entrenamiento');
              }
            } catch (error) {
              setStatus('error', 'Error al enviar la solicitud.');
            }
          });

          resetButton.addEventListener('click', () => {
            architectureInput.value = defaultConfig.architecture || DEFAULT_ARCHITECTURE;
            epochsInput.value = defaultConfig.epochs;
            learningRateInput.value = defaultConfig.learning_rate;
            noiseInput.value = defaultConfig.noise;
            noiseValue.textContent = Number(defaultConfig.noise).toFixed(2);
          });

          noiseInput.addEventListener('input', (event) => {
            noiseValue.textContent = Number(event.target.value).toFixed(2);
          });

          renderArtifacts(INITIAL_ARTIFACTS);
          handleStatus(INITIAL_STATUS);
          setInterval(pollStatus, 4000);
        </script>
    """
    script = (
        script_template
        .replace("__DEFAULT_ARCH__", default_arch_json)
        .replace("__PAYLOAD__", payload)
        .replace("__STATUS__", status_payload)
    )
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
        <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\" defer></script>
        <style>
          body {{ background: radial-gradient(circle at top, #eef2ff 0%, #ffffff 45%); }}
          .card {{ border: none; border-radius: 1rem; }}
          pre {{ background-color: #1f2933; color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: 'Fira Code', 'Courier New', monospace; }}
          .prediction-card {{ border-radius: 0.75rem; border: 2px solid transparent; transition: transform 0.2s ease; background: rgba(15, 23, 42, 0.03); }}
          .digit-grid {{ display: grid; grid-template-columns: repeat(14, 1fr); gap: 1px; background: #fff; border-radius: 0.5rem; overflow: hidden; box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.05); }}
          .digit-grid span {{ display: block; width: 100%; padding-top: 100%; }}
          .prediction-card.is-correct {{ border-color: #198754; }}
          .prediction-card.is-incorrect {{ border-color: #dc3545; }}
          .prediction-card:hover {{ transform: translateY(-4px); }}
          .metric-pill {{ display: inline-flex; flex-direction: column; align-items: flex-start; padding: 0.5rem 1rem; border-radius: 999px; background: rgba(13, 110, 253, 0.1); }}
          .metric-pill strong {{ font-size: 1.25rem; }}
          .status-chip {{ display: inline-flex; align-items: center; gap: 0.25rem; border-radius: 999px; padding: 0.35rem 0.75rem; font-weight: 600; }}
          .status-idle {{ background: rgba(108, 117, 125, 0.15); color: #6c757d; }}
          .status-running {{ background: rgba(255, 193, 7, 0.25); color: #b8860b; }}
          .status-completed {{ background: rgba(25, 135, 84, 0.2); color: #198754; }}
          .status-error {{ background: rgba(220, 53, 69, 0.2); color: #dc3545; }}
          #status-alert.d-none {{ display: none !important; }}
        </style>
      </head>
      <body class=\"bg-light text-dark\">
        <div class=\"container py-4\">
          <header class=\"mb-4\">
            <h1 class=\"display-5 fw-bold\">Análisis y Demo de IA para MNIST</h1>
            <p class=\"lead\">Construye y entrena una red neuronal a partir de una descripción textual.</p>
            <div class=\"d-flex flex-wrap gap-3 align-items-center\">
              <span class=\"metric-pill bg-white shadow-sm\">
                <small class=\"text-muted text-uppercase\">Precisión</small>
                <strong id=\"accuracy-badge\">{accuracy:.2f}%</strong>
              </span>
              <span class=\"metric-pill bg-white shadow-sm\">
                <small class=\"text-muted text-uppercase\">Pérdida</small>
                <strong id=\"loss-badge\">{artifacts.evaluation[0]:.4f}</strong>
              </span>
              <span id=\"status-chip\" class=\"status-chip status-{_training_status['state']}\">Estado: {_training_status['state'].capitalize()}</span>
            </div>
          </header>

          <section class=\"mb-4\">
            <div class=\"row g-4\">
              <div class=\"col-lg-7\">
                <div class=\"card shadow-sm h-100\">
                  <div class=\"card-body\">
                    <h2 class=\"h4 mb-3\">Laboratorio interactivo</h2>
                    <p class=\"mb-4\">
                      Ajusta la arquitectura del perceptrón multicapa, el número de épocas, la tasa de aprendizaje y la
                      cantidad de ruido sintético para experimentar cómo impactan en el entrenamiento.
                    </p>
                    <form id=\"training-form\" class=\"row gy-3\">
                      <div class=\"col-12\">
                        <label for=\"architecture-input\" class=\"form-label\">Arquitectura (mini-lenguaje)</label>
                        <textarea id=\"architecture-input\" class=\"form-control\" rows=\"3\">{artifacts.architecture}</textarea>
                        <div class=\"form-text\">Ejemplo: Dense(128, relu) -&gt; Dense(64, relu) -&gt; Dense(10, softmax)</div>
                      </div>
                      <div class=\"col-sm-4\">
                        <label for=\"epochs-input\" class=\"form-label\">Épocas</label>
                        <input id=\"epochs-input\" type=\"number\" class=\"form-control\" min=\"1\" max=\"100\" value=\"{artifacts.epochs}\">
                      </div>
                      <div class=\"col-sm-4\">
                        <label for=\"learning-rate-input\" class=\"form-label\">Learning rate</label>
                        <input id=\"learning-rate-input\" type=\"number\" class=\"form-control\" step=\"0.001\" min=\"0.001\" max=\"1\" value=\"{artifacts.learning_rate:.4f}\">
                      </div>
                      <div class=\"col-sm-4\">
                        <label for=\"noise-input\" class=\"form-label\">Ruido sintético</label>
                        <input id=\"noise-input\" type=\"range\" class=\"form-range\" min=\"0\" max=\"0.5\" step=\"0.05\" value=\"{artifacts.noise:.2f}\">
                        <div class=\"form-text\">Valor actual: <span id=\"noise-value\">{artifacts.noise:.2f}</span></div>
                      </div>
                      <div class=\"col-12 d-flex gap-2\">
                        <button type=\"submit\" class=\"btn btn-primary\" id=\"train-button\">Entrenar nuevo modelo</button>
                        <button type=\"button\" class=\"btn btn-outline-secondary\" id=\"reset-button\">Restablecer valores</button>
                      </div>
                    </form>
                    <div id=\"status-alert\" class=\"alert alert-secondary mt-3 d-none\" role=\"alert\"></div>
                    <div class=\"mt-4\">
                      <h3 class=\"h6 text-uppercase text-muted mb-1\">Arquitectura activa</h3>
                      <p class=\"mb-0\"><code id=\"architecture-display\">{artifacts.architecture}</code></p>
                    </div>
                  </div>
                </div>
              </div>
              <div class=\"col-lg-5\">
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
                <pre id=\"summary-block\" class=\"mb-0\">{artifacts.summary}</pre>
              </div>
            </div>
          </section>

          <section class=\"mb-4\">
            <div class=\"row g-4\">
              <div class=\"col-lg-6\">
                <div class=\"card shadow-sm h-100\">
                  <div class=\"card-body\">
                    <h2 class=\"h4\">Evolución de métricas</h2>
                    <canvas id=\"history-chart\" height=\"240\"></canvas>
                    <div class=\"table-responsive mt-3\">
                      <table class=\"table table-sm mb-0\">
                        <thead>
                          <tr>
                            <th>Época</th>
                            <th>Loss</th>
                            <th>Accuracy</th>
                          </tr>
                        </thead>
                        <tbody id=\"history-table\"></tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
              <div class=\"col-lg-6\">
                <div class=\"card shadow-sm h-100\">
                  <div class=\"card-body\">
                    <h2 class=\"h4\">Logs de Entrenamiento</h2>
                    <pre id=\"training-log\" class=\"mb-0\">{artifacts.training_log}</pre>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section class=\"mb-4\">
            <div class=\"card shadow-sm\">
              <div class=\"card-body\">
                <h2 class=\"h4\">Ejemplos de Predicciones</h2>
                <div class=\"row g-3\" id=\"predictions-grid\">
                  {prediction_grid}
                </div>
              </div>
            </div>
          </section>

          <footer class=\"text-center text-muted small\">
            <p>Construido con un intérprete de arquitecturas y un MLP en NumPy.</p>
          </footer>
        </div>
{script}
      </body>
    </html>
    """


@app.before_first_request
def ensure_trained() -> None:
    global _artifacts, _training_status
    with _artifacts_lock:
        if _artifacts is None:
            _training_status = {
                "state": "running",
                "message": "Entrenando modelo inicial...",
            }
            artifacts = train_mnist()
            _artifacts = artifacts
            _training_status = {
                "state": "completed",
                "message": "Modelo inicial entrenado.",
                "result": {
                    "accuracy": artifacts.evaluation[1],
                    "loss": artifacts.evaluation[0],
                    "epochs": len(artifacts.history),
                },
            }


@app.route("/")
def index():
    global _artifacts
    if _artifacts is None:
        ensure_trained()
    return _render_page(_artifacts)


@app.route("/api/status")
def status():
    payload = dict(_training_status)
    if _artifacts is not None:
        payload.update(
            {
                "evaluation": {
                    "accuracy": _artifacts.evaluation[1],
                    "loss": _artifacts.evaluation[0],
                },
                "architecture": _artifacts.architecture,
            }
        )
    return jsonify(payload)


@app.route("/api/artifacts")
def get_artifacts():
    if _artifacts is None:
        return jsonify({"message": "El modelo aún no está disponible."}, status=404)
    return jsonify(_serialize_artifacts(_artifacts))


@app.route("/api/train", methods=["POST"])
def train_endpoint():
    data = request.get_json(silent=True) or {}
    architecture = data.get("architecture") or DEFAULT_ARCHITECTURE
    try:
        epochs = int(data.get("epochs", 20))
    except (TypeError, ValueError):
        epochs = 20
    epochs = max(1, min(100, epochs))
    try:
        learning_rate = float(data.get("learning_rate", data.get("learningRate", 0.05)))
    except (TypeError, ValueError):
        learning_rate = 0.05
    learning_rate = max(0.0001, min(1.0, learning_rate))
    try:
        noise = float(data.get("noise", 0.0))
    except (TypeError, ValueError):
        noise = 0.0
    noise = max(0.0, min(0.5, noise))
    if not _start_training(architecture, epochs, learning_rate, noise):
        return jsonify({"message": "Ya hay un entrenamiento en curso."}, status=409)
    return jsonify(
        {
            "message": "Entrenamiento iniciado.",
            "request": {
                "architecture": architecture,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "noise": noise,
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
