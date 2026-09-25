# percepcion3d

Pipeline de percepción espacial 3D monocular en tiempo real (edge, GPU Ada 8 GB,
Linux/WSL2). Transforma píxeles en coordenadas métricas $X, Y, Z$, velocidad
relativa y alertas de colisión (TTC/CPA) con un diseño asíncrono *dual-rate*.

## Documentación

- [`docs/00_analisis_critico.md`](docs/00_analisis_critico.md) — riesgos,
  trade-offs y mitigaciones por etapa; hallazgos sobre el código actual.
- [`docs/01_plan_fases_mvp.md`](docs/01_plan_fases_mvp.md) — fases F0–F8,
  Definition of Done y estrategia de medición (P50/P95/P99, VRAM).

## Estado

Fase 0 + F0.1 (geometría con roll, intersección con el suelo vectorizada y
modelo de varianza, calibración, rectificación) implementadas y testeadas. El
resto de módulos (`detection/`, `depth/`, `tracking/`, `safety/`, `runtime/`)
son esqueletos pendientes de las fases F1–F7.

## Desarrollo

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"      # núcleo CPU + herramientas
ruff check src tests && ruff format --check src tests
mypy src tests
pytest tests -m "not slow and not gpu"
```

Grupos opcionales (`pyproject.toml`):

| Extra | Contenido | Cuándo |
|---|---|---|
| *(núcleo)* | numpy, opencv-headless, pyyaml, scipy | siempre; CI |
| `runtime` | tensorrt-cu12 10.x, cuda-python, rerun-sdk | inferencia en la GPU objetivo |
| `export` | torch, torchvision | exportar ONNX y `scripts/benchmark_depth.py` |
| `dev` | pytest, ruff, mypy | desarrollo y CI |
