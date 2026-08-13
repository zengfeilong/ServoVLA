# ServoVLA

ServoVLA trains and deploys vision-language-action policies with LeRobot. It reads
raw LeRobot datasets and runs the frozen vision and language encoders during
training. Only the flow-matching policy head is trained. The result can be exported
as a LeRobot-compatible policy for simulation or robot deployment.

## Requirements

- Linux
- Python 3.12
- CUDA-capable GPU for training and normal inference
- LeRobot 0.5.1

Set up the Conda environment:

```bash
conda create -n ServoVLA python=3.12 -y
conda activate ServoVLA
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The default configuration downloads DINOv3, Qwen3.5, and the ServoVLA datasets
from Hugging Face. For offline use with an existing cache, set
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

## Model architecture

By default, visual features come from
`facebook/dinov3-vitb16-pretrain-lvd1689m`, and language features come from
`Qwen/Qwen3.5-0.8B`. Both encoders stay frozen. The policy head is a 10-layer
transformer with a hidden dimension of 640, 10 attention heads, and 0.1 dropout.
It predicts 16-step action chunks and uses three flow-matching steps at inference
time.

## Published checkpoints

The released checkpoints use EMA weights:

- [ServoVLA-SO101](https://huggingface.co/ServoVLA/ServoVLA-SO101) - SO101
  real-robot manipulation policy from training step 150,000.
- [ServoVLA-LIBERO](https://huggingface.co/ServoVLA/ServoVLA-LIBERO) - LIBERO-40
  simulation policy from training step 1,190,000.

Each model repository has LeRobot-compatible `model.safetensors` weights,
processor configuration files, and a cleaned `checkpoint.pt`. The checkpoint can
be used for inference or to initialize a new training run. It does not contain an
optimizer, scheduler, W&B metadata, or other resume state.

SO101 and LIBERO have different observation and action contracts. Their
checkpoints are not interchangeable.

## Datasets

### SO101

The default robot configuration trains on:

- [`ServoVLA/so101_clean_train`](https://huggingface.co/datasets/ServoVLA/so101_clean_train)
- [`ServoVLA/so101_pick_place_train`](https://huggingface.co/datasets/ServoVLA/so101_pick_place_train)
- [`ServoVLA/so101_stacking_train`](https://huggingface.co/datasets/ServoVLA/so101_stacking_train)

Validation uses the matching splits:

- [`ServoVLA/so101_clean_val`](https://huggingface.co/datasets/ServoVLA/so101_clean_val)
- [`ServoVLA/so101_pick_place_val`](https://huggingface.co/datasets/ServoVLA/so101_pick_place_val)
- [`ServoVLA/so101_stacking_val`](https://huggingface.co/datasets/ServoVLA/so101_stacking_val)

Samples are drawn equally from the three training datasets. The policy uses
6-dimensional delta actions and 6-dimensional proprioception. The state order is
shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, then gripper. The
three camera keys are `observation.images.front`, `observation.images.wrist`, and
`observation.images.side`.

Action normalization has shape `16 x 6`. Its statistics come only from the three
training datasets; validation samples are not included.

### LIBERO

Simulation training uses
[`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero)
for LIBERO-40. It covers 40 tasks in `libero_10`, `libero_spatial`,
`libero_object`, and `libero_goal`.

The LIBERO policy sends 7-dimensional actions directly to the environment. Its
state has 8 dimensions, and it reads `observation.images.image` and
`observation.images.image2`. The action normalization shape is `16 x 7`.

Use the matching dataset configuration and checkpoint for each environment.
LIBERO and SO101 differ in action dimensions, state layout, camera keys, and
control semantics.

For another LeRobot dataset, use Hydra arguments to override
`dataset.train_names`, `dataset.val_names`, the camera keys, and the action
settings.

## Training

Start an SO101 training run with the default configuration:

```bash
python scripts/train.py
```

For a short startup check, run one step and disable automatic export:

```bash
python scripts/train.py training.max_steps=1 deployment.auto_export=false
```

Training writes its output to `runs/train/<action-mode>/`. Model and dataset
caches stay outside the repository. Their locations follow the standard Hugging
Face variables, including `HF_HOME` and `HF_LEROBOT_HOME`.

## Checkpoint evaluation

```bash
python scripts/evaluate_checkpoints.py \
  --run-dir runs/train/delta/<run-name> \
  --device cuda
```

## Export and deployment

Export the latest checkpoint and build its deployment bundle:

```bash
python scripts/deploy_from_run.py \
  --run-dir runs/train/delta/<run-name>
```

The exported policy goes to `artifacts/pretrained/<run-name>/`. Its deployment
directory has a manifest, policy server configuration, Jetson configuration, and
the Jetson runner.

Start the policy server below. The LeRobot client supplies the exported policy
path.

```bash
python scripts/serve_servovla_policy.py --host=127.0.0.1 --port=8080
```

Set the robot serial ports, cameras, tasks, and pretrained policy path in
`configs/deployment/jetson.yaml`. Relative paths in this file are resolved from
the repository root.

```bash
python scripts/jetson_runner.py list-tasks \
  --config configs/deployment/jetson.yaml
python scripts/jetson_runner.py async-client \
  --config configs/deployment/jetson.yaml \
  --task pick_place
```

## Simulation

Evaluate a LIBERO training run:

```bash
python scripts/evaluate_sim_libero.py \
  --run-dir runs/train/abs/<run-name> \
  --max-tasks 1
```

This command requires a LIBERO checkpoint with 7 action dimensions, 8 state
dimensions, and two cameras. An SO101 checkpoint has incompatible tensor shapes.

To evaluate an exported LIBERO policy:

```bash
python scripts/evaluate_sim_libero_pretrained.py \
  --pretrained-dir artifacts/pretrained/<run-name> \
  --max-tasks 1
```

Install the upstream benchmark assets and dependencies before running the
simulation. If an output path is configured, results go to `runs/eval/`.

## Tests

```bash
ruff check src scripts tests
python -m compileall -q src scripts tests
pytest -q
```

CI does not run full training or benchmark rollouts. Those jobs require local
datasets, model weights, GPU resources, and simulator installations.

## License

ServoVLA is released under the MIT License. See [LICENSE](LICENSE).
