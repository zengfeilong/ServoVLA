from pathlib import Path

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_world_default_dataset_uses_public_servovla_datasets():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/dataset/default.yaml")

    assert list(cfg.train_names) == [
        "ServoVLA/so101_clean_train",
        "ServoVLA/so101_pick_place_train",
        "ServoVLA/so101_stacking_train",
    ]
    assert list(cfg.val_names) == [
        "ServoVLA/so101_clean_val",
        "ServoVLA/so101_pick_place_val",
        "ServoVLA/so101_stacking_val",
    ]
    assert cfg.action_mode == "delta"


def test_default_real_dataset_uses_ordered_three_camera_contract():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/dataset/default.yaml")

    assert list(cfg.camera_keys) == [
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.side",
    ]
    assert cfg.num_cameras == 3
    assert OmegaConf.to_container(cfg.camera_key_aliases, resolve=True) == {
        "front": ["observation.images.front"],
        "wrist": ["observation.images.wrist"],
        "side": ["observation.images.side"],
    }


def test_real_world_default_dataset_uses_equal_explicit_weights():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/dataset/default.yaml")

    assert cfg.train_weight_strategy == "explicit_weights"
    assert list(cfg.train_weights) == [1.0, 1.0, 1.0]


def test_default_policy_head_uses_dropout_point_one():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/model/policy_head.yaml")

    assert cfg.hidden_dim == 640
    assert cfg.num_heads == 10
    assert cfg.num_layers == 10
    assert cfg.dropout == 0.1
    assert cfg.num_inference_steps == 3


def test_default_encoder_configs_use_sim_aligned_model_sizes():
    vision_cfg = OmegaConf.load(_REPO_ROOT / "configs/model/vision_encoder.yaml")
    vlm_cfg = OmegaConf.load(_REPO_ROOT / "configs/model/vlm_encoder.yaml")

    assert vision_cfg.model_id == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert vision_cfg.feature_dim == 768
    assert vision_cfg.image_size == 256
    assert vlm_cfg.model_id == "Qwen/Qwen3.5-0.8B"
    assert vlm_cfg.feature_dim == 1024
    assert vlm_cfg.image_size == 256


def test_default_training_config_is_end2end_only():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/training/default.yaml")

    forbidden = [
        "data_mode",
        "shards_" + "root_base",
        "shards_" + "root",
        "auto_pack_if_missing",
        "stage_" + "shards_to_fast_storage",
        "stage_" + "shards_fast_root_base",
        "stage_" + "shards_fast_root",
        "stage_" + "shards_cleanup_other_datasets",
        "shard_assignment",
        "active_chunks",
        "payload_cache",
        "collate_io",
    ]
    for key in forbidden:
        assert key not in cfg
    assert "resume_from" not in cfg

    assert cfg.batch_size == 128
    assert cfg.amp_dtype == "bfloat16"
    assert cfg.policy_head_param_dtype == "float32"
    assert cfg.compile is True
    assert cfg.compile_policy_head is True
    assert cfg.compile_vision_encoder is True
    assert cfg.compile_vlm_encoder is True
    assert cfg.prefetch_factor == 1
    assert cfg.vlm_processor_micro_batch_size == 64
    assert cfg.vlm_sequence_padding_multiple == 16
    assert cfg.eval_batch_size == 16
    assert cfg.eval_every == 0
    assert cfg.eval_use_ema is True
    assert cfg.ema.decay == 0.99609375
    assert cfg.ema.update_after_step == 5000
    assert cfg.ema.update_every == 4
    assert cfg.num_workers == 2
    assert cfg.prompt_cache_size == 128
    assert cfg.semantic_delay.max_delay_chunks == 2
    assert cfg.semantic_delay.chunk_size_threshold == 0.0
    assert cfg.semantic_delay.sync_weight == 1.0
    assert list(cfg.semantic_delay.async_bucket_weights) == [1.0, 1.0]
    assert cfg.action_normalization.dataset_weight_strategy == "equal_datasets"
    assert cfg.optimizer.lr == 1.0e-4
    assert cfg.scheduler.warmup_steps == 0


def test_default_training_config_exposes_end2end_optimization_namespaces():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/training/default.yaml")

    assert cfg.raw_shuffle.strategy == "epoch_chunk"
    assert cfg.raw_shuffle.chunk_steps == 256
    assert cfg.raw_shuffle.shuffle_chunks is True
    assert cfg.raw_shuffle.reshuffle_each_epoch is True
    assert cfg.raw_shuffle.epoch_boundary_offsets is True
    assert cfg.raw_shuffle.seed_stride == 100003
    assert cfg.raw_shuffle.batch_dataset_strategy == "quota"
    assert cfg.raw_shuffle.batch_dataset_burst_batches == 1
    assert cfg.raw_shuffle.quota_max_datasets_per_batch == 1
    assert cfg.raw_shuffle.max_episodes_per_batch == 0
    assert cfg.raw_shuffle.episode_burst_batches == 1
    assert cfg.batch_mixing.mode == "explicit_weights"
    assert cfg.batch_mixing.integerization == "largest_remainder"
    assert cfg.gpu_pipeline.enabled is True
    assert cfg.gpu_pipeline.feature_queue_depth == 4
    assert cfg.gpu_pipeline.encoder_parallel_streams is True
    assert cfg.gpu_pipeline.gpu_preprocess is True
    assert cfg.gpu_pipeline.async_producer is True
    assert cfg.gpu_pipeline.cpu_prefetch_depth == 4
    assert cfg.gpu_pipeline.vision_encoder_micro_batch_size == 8
    assert cfg.gpu_pipeline.vlm_encoder_micro_batch_size == 32
    assert cfg.gpu_pipeline.startup_trace_submit_limit == 0
    assert cfg.gpu_pipeline.startup_trace_step_limit == 0
    assert cfg.vlm_compile.overlap_train_reader is True
    assert cfg.dataloader_in_order is False
    assert cfg.decode.num_workers == 2
    assert cfg.decode.intra_batch_parallelism == 4
    assert cfg.decode.decoder_cache_size == 12
    assert cfg.decode.max_batch_decode_span_s == 2.0


def test_default_training_config_exposes_rollout_loss_and_epoch_chunk_shuffle():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/training/default.yaml")

    assert cfg.raw_shuffle.strategy == "epoch_chunk"
    assert cfg.raw_shuffle.chunk_steps == 256
    assert cfg.raw_shuffle.shuffle_chunks is True
    assert cfg.raw_shuffle.reshuffle_each_epoch is True
    assert cfg.raw_shuffle.epoch_boundary_offsets is True
    assert cfg.raw_shuffle.seed_stride == 100003
    assert cfg.raw_shuffle.batch_dataset_strategy == "quota"
    assert cfg.raw_shuffle.batch_dataset_burst_batches == 1
    assert cfg.raw_shuffle.quota_max_datasets_per_batch == 1
    assert cfg.raw_shuffle.max_episodes_per_batch == 0
    assert cfg.raw_shuffle.episode_burst_batches == 1

    assert cfg.rollout_action_loss.enabled is True
    assert cfg.rollout_action_loss.loss_type == "smooth_l1"
    assert cfg.rollout_action_loss.beta == 0.05
    assert cfg.rollout_action_loss.max_weight == 0.2
    assert cfg.rollout_action_loss.start_step == 20000
    assert cfg.rollout_action_loss.end_step == 60000
    assert cfg.rollout_action_loss.schedule == "cosine"
    assert cfg.rollout_action_loss.noise_source == "fm_noise"


def test_default_training_config_exposes_vlm_compile_warmup_limits():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/training/default.yaml")

    assert cfg.vlm_compile.warmup_enabled is True
    assert cfg.vlm_compile.warmup_max_batches == 2
    assert cfg.vlm_compile.warmup_batch_size is None
    assert cfg.vlm_compile.warmup_num_workers == 0
    assert cfg.vlm_compile.warmup_shuffle_episodes is False
    assert cfg.vlm_compile.synthetic_raw_batches is True
    assert cfg.vlm_compile.signature_max_entries == 16
    assert cfg.vlm_compile.on_new_signature == "warn_and_warmup"
    assert cfg.vlm_compile.log_signatures is True
    assert cfg.vlm_compile.eval_warmup_enabled is True
    assert cfg.vlm_compile.position_cache_enabled is False
    assert cfg.vlm_compile.flash_attention_graph_break is True
    assert cfg.vlm_compile.qwen_visual_position_graph_break is True
    assert cfg.vlm_compile.nested_fx_trace_fallback is True


def test_sim_libero_training_config_matches_real_world_default_training_config():
    default_cfg = OmegaConf.load(_REPO_ROOT / "configs/training/default.yaml")
    libero_cfg = OmegaConf.load(_REPO_ROOT / "configs/training/sim_libero.yaml")

    default_container = OmegaConf.to_container(default_cfg, resolve=True)
    libero_container = OmegaConf.to_container(libero_cfg, resolve=True)
    assert libero_container["semantic_delay"]["chunk_size_threshold"] == 0.0
    assert default_container["semantic_delay"]["chunk_size_threshold"] == 0.0
    assert libero_container == default_container


def test_default_deployment_config_keeps_sync_mode_default():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/deployment/default.yaml")

    assert cfg.robot_client.mode == "sync"
    assert cfg.robot_client.chunk_size_threshold == 0.2


def test_train_config_describes_end2end_training():
    content = (_REPO_ROOT / "configs/train.yaml").read_text(encoding="utf-8")

    assert "end-to-end" in content
    assert "chunk-" + "native" not in content


def test_train_config_uses_end2end_artifact_root():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/train.yaml")
    unresolved = OmegaConf.to_container(cfg, resolve=False)

    assert str(cfg.output_root) == "runs/train"
    assert (
        unresolved["output_dir"] == "${output_root}/${dataset.action_mode}/${now:%Y-%m-%d_%H-%M-%S}"
    )


def test_top_level_inference_seed_defaults_are_config_driven():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/train.yaml")

    assert cfg.inference.noise_seed is None
    assert cfg.inference.noise_seed_mode == "step"


def test_public_docs_and_configs_use_repository_relative_artifact_paths():
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    train_cfg = OmegaConf.load(_REPO_ROOT / "configs/train.yaml")
    libero_cfg = OmegaConf.load(_REPO_ROOT / "configs/eval/sim_libero.yaml")

    assert "runs/train/" in readme
    assert "artifacts/pretrained/" in readme
    assert train_cfg.output_root == "runs/train"
    assert libero_cfg.run_root == "runs/eval/libero_40"


def test_public_repository_contains_only_approved_python_entrypoints():
    assert {path.name for path in (_REPO_ROOT / "scripts").glob("*.py")} == {
        "async_eval_worker.py",
        "deploy_from_run.py",
        "evaluate_checkpoints.py",
        "evaluate_sim_libero.py",
        "evaluate_sim_libero_pretrained.py",
        "export_lerobot_pretrained.py",
        "jetson_runner.py",
        "serve_servovla_policy.py",
        "train.py",
    }


def test_training_and_eval_devices_are_config_driven():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/training/default.yaml")

    assert cfg.device == "cuda"
    assert cfg.async_eval.device == "cuda"
    assert "cuda_visible_devices" not in cfg.async_eval
    assert cfg.decode.device == "cuda"


def test_public_defaults_allow_downloads_and_do_not_pin_a_gpu_index():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/deployment/default.yaml")
    vision_cfg = OmegaConf.load(_REPO_ROOT / "configs/model/vision_encoder.yaml")
    vlm_cfg = OmegaConf.load(_REPO_ROOT / "configs/model/vlm_encoder.yaml")

    assert cfg.server.hf_offline is False
    assert cfg.server.cuda_visible_devices == ""
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.public_host == "127.0.0.1"
    assert vision_cfg.attn_implementation == "sdpa"
    assert vlm_cfg.attn_implementation == "sdpa"


def test_jetson_config_is_public_and_relative():
    cfg = OmegaConf.load(_REPO_ROOT / "configs/deployment/jetson.yaml")

    assert cfg.async_client.policy_type == "servovla"
    assert cfg.async_client.server_address == "127.0.0.1:8080"
    assert cfg.async_client.pretrained_name_or_path == "artifacts/pretrained/default"
    assert cfg.record.dataset_root == "data/recordings"


def test_dataset_configs_define_raw_action_dimensions():
    default_cfg = OmegaConf.load(_REPO_ROOT / "configs/dataset/default.yaml")
    libero_cfg = OmegaConf.load(_REPO_ROOT / "configs/dataset/sim_libero.yaml")

    assert default_cfg.action_dim == 6
    assert libero_cfg.action_dim == 7
    assert libero_cfg.proprio_dim == 8


def test_dependency_contract_pins_lerobot_051_runtime():
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.12"' in pyproject
    assert 'target-version = "py312"' in pyproject

    expected_pyproject_deps = [
        '"torch>=2.7"',
        '"torchvision>=0.22"',
        '"transformers==5.3.0"',
        '"numpy>=2.0,<2.3"',
        '"lerobot==0.5.1"',
    ]
    for dep in expected_pyproject_deps:
        assert dep in pyproject

    expected_requirements = [
        "torch>=2.7",
        "torchvision>=0.22",
        "transformers==5.3.0",
        "numpy>=2.0,<2.3",
        "lerobot==0.5.1",
    ]
    for dep in expected_requirements:
        assert dep in requirements

    assert "lerobot>=0.4.4" not in pyproject
    assert "lerobot>=0.4.4" not in requirements


def test_docs_reference_lerobot_051_python312_runtime():
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "LeRobot 0.5.1" in readme
    assert "v0.4.4" not in readme
    assert "Python 3.12" in readme
