from __future__ import annotations

from pathlib import Path

from scripts import evaluate_checkpoints


def test_checkpoint_eval_cli_accepts_partial_weight_and_metric_modes(tmp_path):
    args = evaluate_checkpoints._parse_args(
        [
            "--run-dir",
            str(tmp_path),
            "--weights",
            "raw",
            "--metrics",
            "fm",
        ]
    )

    assert args.weights == "raw"
    assert args.metrics == "fm"


def test_partial_checkpoint_eval_uses_distinct_output_path(tmp_path):
    output_path, error_path = evaluate_checkpoints._eval_artifact_paths(
        tmp_path,
        step=123,
        weights="raw",
        metrics="fm",
    )

    assert output_path == Path(tmp_path) / "eval" / "step0000123.raw.fm.json"
    assert error_path == Path(tmp_path) / "eval" / "step0000123.raw.fm.error.json"


def test_full_checkpoint_eval_keeps_existing_output_path(tmp_path):
    output_path, error_path = evaluate_checkpoints._eval_artifact_paths(
        tmp_path,
        step=123,
        weights="both",
        metrics="both",
    )

    assert output_path == Path(tmp_path) / "eval" / "step0000123.json"
    assert error_path == Path(tmp_path) / "eval" / "step0000123.error.json"


def test_collect_checkpoints_only_scans_requested_run(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "checkpoint_step0001000.pt").write_bytes(b"checkpoint")
    (run_b / "checkpoint_step0002000.pt").write_bytes(b"checkpoint")

    checkpoints = evaluate_checkpoints._collect_checkpoints(
        run_b,
        checkpoint_glob="checkpoint_step*.pt",
    )

    assert checkpoints == [run_b / "checkpoint_step0002000.pt"]
