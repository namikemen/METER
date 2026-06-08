from pathlib import Path

from globals import RGB_img_res, augmentation_parameters, depth_unit, dts_type, max_depth


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nyu_cm_defaults():
    assert RGB_img_res == (3, 192, 256)
    assert dts_type == "nyu"
    assert depth_unit == "cm"
    assert max_depth == 1000.0
    assert augmentation_parameters == {
        "flip": 0.5,
        "mirror": 0.5,
        "c_swap": 0.5,
        "random_crop": 0.5,
        "shifting_strategy": 0.5,
    }


def test_upload_script_uses_resumable_rsync():
    script = (REPO_ROOT / "scripts" / "upload_nyu_data.sh").read_text()

    assert "rsync" in script
    assert "--partial" in script
    assert "--progress" in script
    assert "nyu_depth_v2" in script


def test_docker_run_script_mounts_server_directories():
    script = (REPO_ROOT / "scripts" / "run_server_docker.sh").read_text()

    assert "--gpus all" in script
    assert "/workspace/METER" in script
    assert "/workspace/data" in script
    assert "/workspace/outputs" in script
    assert "/workspace/checkpoints" in script
    assert '"$@"' in script
