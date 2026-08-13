# ServoVLA

ServoVLA is an end-to-end vision-language-action training and deployment stack for
LeRobot. It trains a flow-matching policy head from raw LeRobot datasets while
running frozen vision and language encoders online, then exports a LeRobot-compatible
pretrained policy for simulation or robot deployment.

## Requirements

- Linux
- Python 3.12
- CUDA-capable GPU for training and normal inference
- LeRobot 0.5.1

Create and activate the recommended Conda environment:

```bash
conda create -n ServoVLA python=3.12 -y
conda activate ServoVLA
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The default configuration downloads DINOv3, Qwen3.5, and the public ServoVLA
datasets from Hugging Face. To use an existing cache without network access, set
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

## Public Datasets

The default real-robot configuration uses these datasets:

- `ServoVLA/so101_clean_train`
- `ServoVLA/so101_pick_place_train`
- `ServoVLA/so101_stacking_train`
- `ServoVLA/so101_clean_val`
- `ServoVLA/so101_pick_place_val`
- `ServoVLA/so101_stacking_val`

Override `dataset.train_names`, `dataset.val_names`, camera keys, or action settings
with Hydra arguments when using another LeRobot dataset.

## Training

Start the default SO101 training run:

```bash
python scripts/train.py
```

Run a short startup validation without automatic export:

```bash
python scripts/train.py training.max_steps=1 deployment.auto_export=false
```

Training outputs are written under `runs/train/<action-mode>/`. Model and dataset
caches remain outside the repository and follow the standard Hugging Face variables,
including `HF_HOME` and `HF_LEROBOT_HOME`.

## Checkpoint Evaluation

```bash
python scripts/evaluate_checkpoints.py \
  --run-dir runs/train/delta/<run-name> \
  --device cuda
```

## Export and Deployment

Export the latest checkpoint and create a deployment bundle:

```bash
python scripts/deploy_from_run.py \
  --run-dir runs/train/delta/<run-name>
```

Exports are written under `artifacts/pretrained/<run-name>/`. The run's deployment
directory contains a manifest, policy-server configuration, Jetson configuration,
and Jetson runner.

Start a policy server with an exported policy path supplied by the LeRobot client:

```bash
python scripts/serve_servovla_policy.py --host=127.0.0.1 --port=8080
```

Edit `configs/deployment/jetson.yaml` for robot serial ports, cameras, tasks, and
the pretrained policy path. Relative paths in this file are resolved from the
repository root.

```bash
python scripts/jetson_runner.py list-tasks \
  --config configs/deployment/jetson.yaml
python scripts/jetson_runner.py async-client \
  --config configs/deployment/jetson.yaml \
  --task pick_place
```

## Simulation

Evaluate a training run on LIBERO:

```bash
python scripts/evaluate_sim_libero.py \
  --run-dir runs/train/abs/<run-name> \
  --max-tasks 1
```

This entry point requires a LIBERO checkpoint trained with 7 action dimensions,
8 state dimensions, and two cameras. Real-robot SO101 checkpoints are not
shape-compatible with LIBERO.

Evaluate an exported LIBERO policy:

```bash
python scripts/evaluate_sim_libero_pretrained.py \
  --pretrained-dir artifacts/pretrained/<run-name> \
  --max-tasks 1
```

Simulation environments require their upstream benchmark assets and dependencies.
Evaluation results are written under `runs/eval/` when an output path is configured.

## Tests

```bash
ruff check src scripts tests
python -m compileall -q src scripts tests
pytest -q
```

Full training and benchmark rollouts require local datasets, model weights, GPU
resources, and simulator installations; they are intentionally not run in CI.

## License

ServoVLA is released under the MIT License. See [LICENSE](LICENSE).
