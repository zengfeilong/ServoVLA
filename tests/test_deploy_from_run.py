from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from scripts import deploy_from_run


def test_deploy_from_run_use_raw_disables_ema_export(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text("deployment: {}\n", encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint_step000100.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    pretrained_dir = run_dir / "pretrained_raw"

    cfg = OmegaConf.create(
        {
            "deployment": {
                "prefer_ema": True,
                "pretrained_root": "artifacts/pretrained",
            }
        }
    )
    captured: dict[str, object] = {}

    def fake_export(checkpoint, output_dir, *, prefer_ema, config_overrides):
        captured["checkpoint"] = Path(checkpoint)
        captured["output_dir"] = Path(output_dir)
        captured["prefer_ema"] = prefer_ema
        captured["config_overrides"] = config_overrides
        return Path(output_dir)

    def fake_write_bundle(cfg_arg, run_dir_arg, checkpoint_arg, exported_dir_arg):
        captured["bundle"] = (
            cfg_arg,
            Path(run_dir_arg),
            Path(checkpoint_arg),
            Path(exported_dir_arg),
        )
        return Path(run_dir_arg) / "deployment"

    monkeypatch.setattr(deploy_from_run.OmegaConf, "load", lambda _: cfg)
    monkeypatch.setattr(
        deploy_from_run, "build_export_kwargs_from_training_cfg", lambda _: {"device": "cpu"}
    )
    monkeypatch.setattr(deploy_from_run, "export_checkpoint_to_pretrained", fake_export)
    monkeypatch.setattr(deploy_from_run, "write_deployment_bundle", fake_write_bundle)

    deploy_from_run.main(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--pretrained-dir",
            str(pretrained_dir),
            "--use-raw",
        ]
    )

    assert captured["checkpoint"] == checkpoint_path.resolve()
    assert captured["output_dir"] == pretrained_dir.resolve()
    assert captured["prefer_ema"] is False
    assert captured["config_overrides"] == {"device": "cpu"}
    bundle_cfg, *_ = captured["bundle"]
    assert bundle_cfg.deployment.prefer_ema is False
    assert "Export complete:" in capsys.readouterr().out


def test_deploy_from_run_use_ema_enables_ema_export(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text("deployment: {}\n", encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint_step000100.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    cfg = OmegaConf.create(
        {
            "deployment": {
                "prefer_ema": False,
                "pretrained_root": "artifacts/pretrained",
            }
        }
    )
    captured: dict[str, object] = {}

    def fake_export(checkpoint, output_dir, *, prefer_ema, config_overrides):
        captured["checkpoint"] = Path(checkpoint)
        captured["output_dir"] = Path(output_dir)
        captured["prefer_ema"] = prefer_ema
        captured["config_overrides"] = config_overrides
        return Path(output_dir)

    def fake_write_bundle(cfg_arg, *_args):
        captured["bundle_cfg"] = cfg_arg
        return run_dir / "deployment"

    monkeypatch.setattr(deploy_from_run.OmegaConf, "load", lambda _: cfg)
    monkeypatch.setattr(
        deploy_from_run, "build_export_kwargs_from_training_cfg", lambda _: {"device": "cpu"}
    )
    monkeypatch.setattr(deploy_from_run, "export_checkpoint_to_pretrained", fake_export)
    monkeypatch.setattr(deploy_from_run, "write_deployment_bundle", fake_write_bundle)

    deploy_from_run.main(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--use-ema",
        ]
    )

    assert captured["prefer_ema"] is True
    assert (
        captured["output_dir"]
        == (deploy_from_run._PROJECT_ROOT / "artifacts" / "pretrained" / run_dir.name).resolve()
    )
    assert captured["bundle_cfg"].deployment.prefer_ema is True


def test_deploy_from_run_skips_bundle_when_deployment_disabled(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text("deployment: {}\n", encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint_step000100.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    cfg = OmegaConf.create(
        {
            "deployment": {
                "enabled": False,
                "prefer_ema": False,
                "pretrained_root": "artifacts/pretrained",
            }
        }
    )
    captured: dict[str, object] = {}

    def fake_export(checkpoint, output_dir, *, prefer_ema, config_overrides):
        captured["checkpoint"] = Path(checkpoint)
        captured["output_dir"] = Path(output_dir)
        captured["prefer_ema"] = prefer_ema
        captured["config_overrides"] = config_overrides
        return Path(output_dir)

    def fail_write_bundle(*_args, **_kwargs):
        raise AssertionError("disabled deployment should not write a robot bundle")

    monkeypatch.setattr(deploy_from_run.OmegaConf, "load", lambda _: cfg)
    monkeypatch.setattr(
        deploy_from_run, "build_export_kwargs_from_training_cfg", lambda _: {"device": "cpu"}
    )
    monkeypatch.setattr(deploy_from_run, "export_checkpoint_to_pretrained", fake_export)
    monkeypatch.setattr(deploy_from_run, "write_deployment_bundle", fail_write_bundle)

    deploy_from_run.main(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    assert captured["checkpoint"] == checkpoint_path.resolve()
    assert (
        captured["output_dir"]
        == (deploy_from_run._PROJECT_ROOT / "artifacts" / "pretrained" / run_dir.name).resolve()
    )
    assert captured["prefer_ema"] is False
    output = capsys.readouterr().out
    assert "Export complete:" in output
    assert "Deployment bundle skipped" in output
