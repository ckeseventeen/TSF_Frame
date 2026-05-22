# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable; installs both `tsf_frame` and `configs` as top-level packages)
pip install -e .

# Tests
pytest tests/                 # full suite (147 passing)
pytest tests/ -v              # verbose
pytest tests/test_improvements_tier_ab.py    # Tier A+B unit tests (16)
pytest tests/test_registry_integrity.py      # model registry integrity (21)
pytest tests/test_multi_horizon_monitor.py   # MultiHorizonMonitor lifecycle (13)
pytest tests/test_edge_cases.py              # adapter/feat_eng/mfh edge cases (13)
pytest tests/test_monitoring.py::test_model_monitor_basic   # single test

# Pipelines
python pipelines/examples/run_hpf_forecast.py   # end-to-end HPF (ML path, 4 PNG outputs)
python pipelines/run_monthly_controller.py      # monthly batch over all TASKS (csv/hive)
python pipelines/train_model.py --model ridge --dataset air_passengers
python pipelines/train_model.py --model lstm  --dataset air_passengers --epochs 20

# Single-module demos
python pipelines/examples/hpf_dl_example.py            # 6 DL models comparison (RevIN on)
python pipelines/examples/hpf_monitoring_example.py    # full monitoring loop → SQLite + alerts.log + PNG report
python pipelines/examples/inference_demo.py            # InferenceRunner train → save → load → predict
python pipelines/examples/feature_engineering_example.py
python pipelines/examples/probabilistic_example.py
python pipelines/examples/panel_income_spending_example.py  # panel data + DLinear multi-target
python pipelines/examples/hpf_moirai_zeroshot_example.py
```

CWD-independence: pipelines/examples insert project root + `src/` into `sys.path` at top of file, so direct `python file.py` works without `pip install -e .`.

## Architecture

### Layered design (1 business per `configs/<biz>/` + `BusinessAdapter`)

```
configs/<biz>/  + business/<biz>_adapter.py    ← swap business here, frame untouched
                          │
                          ▼
src/tsf_frame/  data → features → models → monitoring → visualization → deployment
```

Four hard rules baked into the codebase:

1. **Business / framework isolation** — `tsf_frame.*` is generic, `configs.*` + `business/*_adapter.py` is business-specific. New business = new `BusinessAdapter` subclass + new `configs/<biz>/` package, **never patch tsf_frame**.
2. **src-layout**: source under `src/tsf_frame/`; setup.py uses mixed `package_dir` (`{'': 'src', 'configs': 'configs'}`) so `pip install -e .` installs both top-level packages. Don't move `configs/` under `src/`.
3. **Config as code**: all hyperparams/thresholds are `@dataclass` (`configs/base_config.py`, `configs/hpf/hpf_config.py`). Runtime objects take `dict` — call `cfg.to_adapter_config()` / `cfg.to_dict()` at the seam.
4. **Logs anchor to project root, not CWD**: `HPFMonitoringConfig` uses `field(default_factory=lambda: _root_path(...))`. Don't replace with relative paths.

### Four core ABCs

| ABC | Location | Concrete impls |
|---|---|---|
| `BaseModel(ABC, nn.Module)` | `models/base_model.py` | 11 ML (`classical/ml_models.py`) + 6 DL (`transformer/transformer_models.py`) + MLP (`tabular/mlp_model.py`) + Moirai |
| `BaseBusinessAdapter` | `business/base_adapter.py` | `HPFAdapter` |
| `BaseMonitor` | `monitoring/base_monitor.py` | `ModelMonitor`, `PipelineMonitor` |
| 6 monitoring plugin ABCs | `monitoring/interfaces.py` | `MetricStore`, `AlertChannel`, `DriftDetector`, `QualityChecker`, `RuleChecker`, `ReportGenerator` |

All plugins use registry + factory: `@register_store('sqlite')` → `create_store('sqlite', path=...)`. New plugin = subclass ABC + decorator, no central edit. Listed in `monitoring/__init__.py`.

### `ModelMonitor` composition (read this before touching monitoring)

```
ModelMonitor(model_id)
├── store               MetricStore       (default InMemoryStore)
├── alert_manager       AlertManager      (Console/Logging/File/Callback/Store fan-out)
├── performance_monitor PerformanceMonitor   keys by target_ts (OrderedDict), not position
├── data_quality        DataQualityMonitor   (optional)
├── data_drift          DataDriftDetector    (only if reference passed)
├── concept_drift       ConceptDriftDetector
├── prediction_drift    PredictionDriftDetector  (optional)
├── rule_engine         RuleEngine
└── retraining_trigger  RetrainingTrigger
```

Two-call lifecycle for async actual arrival:
```python
mon.record_prediction(timestamp=target_ts, prediction=p, actual=None)   # at batch time
mon.settle_actual(target_ts=target_ts, y_actual=a)                       # when truth arrives
mon.check_status()  # aggregate metrics, drift, rules, retraining → MonitoringStatus
```

SQLite schema in `monitoring/stores.py`: `predictions`, `metrics_snapshot`, `alerts`. WAL mode for concurrent read/write.

### Production deployment scaffold

```
configs/hpf/task_registry.py        TASKS[] = [{task_id, task_name, sql_path, sql_infer_path?, config_builder}, ...]
configs/hpf/sql_templates/
    req_XX_<name>.sql               training/monitor SQL (full history, e.g. -60 months)
    req_XX_<name>_infer.sql         (optional) inference SQL (short window, e.g. -24 months)
configs/hpf/data_source_config.py   csv (default) | hive | mysql switch via env vars
pipelines/production_loader.py      fetch_data / save_predictions (uses DataSourceConfig)
pipelines/run_monthly_controller.py iterates TASKS → fetch train+infer → monitor → train → infer → sink
pipelines/job_train.py / job_inference.py  per-task runners
tsf_frame/deployment/inference_runner.py   end-to-end save/load/predict wrapper
```

Env switches: `TSF_HPF_SOURCE=hive`, `TSF_HPF_HIVE_HOST=...`, `TSF_HPF_SINK=mysql`, `TSF_HPF_MYSQL_URL=...`. **SQL files are the source of truth for data length/columns**; the framework consumes whatever the SQL returns and only does defensive checks (see `min_required_rows` below).

### Dual-SQL convention (training vs inference)

A task can declare two SQL templates so monthly batches avoid scanning the full training history each run:

```python
{
    "task_id": "REQ_01",
    "sql_path":       "configs/hpf/sql_templates/req_01_collection_amount.sql",        # train: -60 months
    "sql_infer_path": "configs/hpf/sql_templates/req_01_collection_amount_infer.sql",  # infer: -24 months
    ...
}
```

`run_monthly_controller.py` calls `fetch_data(sql_path)` once for monitor/retrain and `fetch_data(sql_infer_path)` once for `run_future_forecast` (autoregressive rolling). **`sql_infer_path` is optional** — when omitted, inference reuses `sql_path` (old behavior preserved). Both SQL files stay valid standalone SQL (no Jinja templating); DBAs can copy-paste either into HiveCLI to verify.

The inference window must satisfy `len(df_infer) >= seq_len + max(lags ∪ rolling_windows ∪ diff_periods) - 1`; the controller asserts this and skips the task if violated.

## Critical non-obvious behaviors

### Anti-data-leakage in `HPFAdapter.preprocess`

```python
adapter.preprocess(train_df, fit=True)   # learn μ/σ → self._scalers
adapter.preprocess(test_df,  fit=False)  # reuse — MUST pass fit=False
adapter.preprocess(df, fit=None)         # auto: first call fits, subsequent transforms
```
Passing `fit=False` before any fit raises. Don't bypass — the residual CI quality depends on this. `InferenceRunner.load` automatically forces `_is_fitted=True` to make this structurally safe.

### `target_ts` alignment in `PerformanceMonitor`

`_records: OrderedDict[datetime, _Record]` keyed by **target timestamp**, not position index. Async backfill via `fill_actual(target_ts, y_true)` works correctly even when window is full and truths arrive out of order. Don't reintroduce position-based access via `self.perf._records[i]` — use public `get_record(ts)` / `get_latest_record()` / `iter_records()`.

### RevIN default-on in all DL models

`_DLBaseModel._init_revin()` reads `config.get('use_revin', True)` — **default True** since the trend-data extrapolation work. All 6 DL models (LSTM/Transformer/Autoformer/iTransformer/TimesNet/DLinear) wrap `forward` with `_maybe_revin_norm` / `_maybe_revin_denorm_target` (or `denorm_multi_target` for DLinear). To disable: `config['use_revin'] = False`.

**RevIN × use_diff mutex guard** lives in `_DLBaseModel._init_revin` itself (not just in the example): set `config['_train_uses_diff_target']` to whatever the calling code passes as `use_diff` to surface a `UserWarning` when both are on. They produce unit-mismatched outputs (RevIN denorms with X-level stats, but Y is in delta units) and blow up MAPE to ~2000%.

### `DLinear` multi-target packing — use `pack_y`/`unpack_y`

DLinear's `forward` outputs `(B, num_targets * pred_len)` in **target-major flatten** order: `[t0_h0..t0_h(H-1), t1_h0..t1_h(H-1), ...]`. Use the helpers; do NOT hand-assemble y with `np.stack(axis=-1).reshape` (that's step-major, training silently misaligns):

```python
y_train = DLinear.pack_y([y_target0, y_target1, ...])   # (N, T*H) target-major
preds   = model.predict_structured(X_test)              # (N, T, H) structured
```

`DLinear.fit` validates `y.shape[-1] == num_targets * pred_len` and raises with a `pack_y` hint if misaligned.

### Residual CI sources (priority: cv > val > train)

`BaseMLModel.fit(train_data, val_data=None, cv_folds=0)`:
- `cv_folds > 0` → K-fold OOF residuals (`_residual_source='cv'`, most rigorous, uses all training samples). KFold runs with `shuffle=False` to preserve time order.
- `val_data` provided → validation residuals (`_residual_source='val'`, recommended default)
- Neither → train residuals + `logger.warning` (`_residual_source='train'`, overconfident on XGB/RF)

### `_dl_fit` early stopping + LR scheduling

Optional, default **off** for backward compat. Set in model config:
- `early_stop_patience: int` (0 = off; >0 tracks best val_loss, restores best state on stop)
- `early_stop_min_delta: float` (improvement threshold, default 0.0)
- `lr_scheduler: 'plateau' | 'cosine' | None` — `plateau` needs val_data; `cosine` uses `T_max=epochs`
- `lr_factor` / `lr_plateau_patience` — fine-tune scheduler

Returns `history['early_stopped_at']` + `history['best_val_loss']` when triggered.

### Inference protocol APIs

`MixedFeatureHandler` exposes two APIs for SQL/Kafka/API callers to write LIMIT clauses and defensive checks:

```python
handler.required_source_columns          # list of cols SQL must SELECT (target + time_varying + static)
handler.min_required_rows(feature_engineer=eng)  # min rows = seq_len + max(lag/window/period) - 1
```

`pipelines/run_monthly_controller.py` runs a defensive `assert len(df_raw) >= required_rows` after `fetch_data`, logging an error and skipping the task instead of letting downstream NaNs crash sklearn. **SQL is the source of truth for length**; these APIs are *consumer-side* checks, not framework-side dictation.

### `InferenceRunner` — end-to-end save/load/predict wrapper

`tsf_frame.deployment.InferenceRunner` bundles 5 training-time states (adapter, feat_eng, mfh, model, model_config) into a single pickle artifact, then offers a one-liner inference call:

```python
runner = InferenceRunner(adapter=..., feat_eng=..., mfh=..., model=..., model_config=cfg)
runner.save('artifact.pkl')
# Later, same or new process:
runner = InferenceRunner.load('artifact.pkl', model_cls=RidgeModel)
prob = runner.predict(latest_df, target_col='monthly_deposit')
```

Predict path: validate → `preprocess(fit=False)` → `feat_eng.transform` → `create_single_input` → `predict_probabilistic` → adapter denormalize. `load` forces `adapter._is_fitted=True`. ML vs DL input shape (2D vs 3D) is auto-detected via `hasattr(model, '_build_model')` duck-typing (don't use `isinstance(model, nn.Module)` — `BaseModel` already inherits `nn.Module`, both ML and DL hit it).

### Multi-output feature importance

`BaseMLModel.get_feature_importance(per_horizon=False, aggregate='mean')` — when wrapped in `MultiOutputRegressor`, default aggregates across all sub-estimators (mean). Use `per_horizon=True` for `(N_feat, n_horizons + 1)` DataFrame with per-horizon columns + `'mean'`. Don't reintroduce hardcoded `estimators_[0]`.

### `RetrainingDecision.errored_rules` vs `triggered`

Predicate exceptions land in `errored_rules` / `rule_errors`, **never** flip `should_retrain=True`. Don't mix the two — business hits and code errors must stay separated for ops.

### Project-root anchoring everywhere

Pipelines walk parents until they find a dir containing both `configs/` and `src/`:
```python
_HERE = Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        ...; break
```
New entry scripts should follow this pattern instead of hardcoding relative paths.

## Where things live

- **New ML model** → `models/classical/ml_models.py`, subclass `BaseMLModel`, register in `MODEL_REGISTRY`
- **New DL model** → `models/transformer/transformer_models.py`, subclass `_DLBaseModel` (reuses `_dl_fit` / `_dl_predict`); call `self._init_revin(num_features=self.input_size)` in `__init__` and wrap forward
- **New tabular / cross-section model** → `models/tabular/`, subclass `_DLBaseModel`, set `use_revin=False` default in config (instance-norm meaningless at L=1); see `MLPModel` for the template (supports 2D `(B,F)` and 3D `(B,L,C)` via `mlp_reduce` in `{'last','mean','flatten'}`)
- **New business** → `business/<name>_adapter.py` subclassing `BaseBusinessAdapter` + `configs/<name>/<name>_config.py`
- **New monitoring rule** → `@register_rule('R_BIZ_*')` decorated fn; HPF-specific rules live alongside `HPFAdapter` to avoid polluting generic `monitoring/rule_engine.py`
- **New alert channel / store / drift detector / report** → subclass ABC in `monitoring/interfaces.py` + `@register_*` decorator
- **New metric** → `@register_metric('name')` in `monitoring/performance_monitor.py` (signature `(y_true, y_pred, **kw) -> float`)
- **New deployment task** → add an entry to `configs/hpf/task_registry.py` `TASKS[]` with `sql_path` (training, full history) and optionally `sql_infer_path` (short window for monthly batches). Drop the corresponding `.sql` file(s) under `configs/hpf/sql_templates/`. `run_monthly_controller.py` picks it up automatically; tasks without `sql_infer_path` fall back to single-SQL behavior.
- **Custom InferenceRunner-like wrapper** → subclass `tsf_frame.deployment.InferenceRunner`, override `predict` / `save` / `load` if you need bespoke serialization or pre/post-processing.

## Reference docs

- `README.md` — quickstart, layer diagram, 30-second demo
- `docs/使用指南.md` — full user API tour (data prep / features / models / monitoring / viz / **deployment**)
- `docs/开发指南.md` — design principles, extension recipes, key technical decisions
- `docs/HPF生产部署蓝图.md` — 20-requirement HPF production blueprint (target deployment)
