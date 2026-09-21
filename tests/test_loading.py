import json
from unittest.mock import patch

import pytest

from smolbge_image_embedding import SmolBGEEmbedder


def test_hub_id_resolves_adapter(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "weights").mkdir()
    (tmp_path / "model/adapter_config.json").write_text(json.dumps({
        "projector_path": "../weights/projector.safetensors"
    }))
    (tmp_path / "weights/projector.safetensors").write_bytes(b"test fixture")
    with patch("smolbge_image_embedding.modeling.snapshot_download", return_value=str(tmp_path)) as download:
        with patch.object(SmolBGEEmbedder, "__init__", return_value=None) as constructor:
            SmolBGEEmbedder.from_pretrained("example/model", revision="fixed")
    assert download.call_args.kwargs["revision"] == "fixed"
    assert download.call_args.kwargs["allow_patterns"] == ["model/adapter_config.json", "weights/projector.safetensors"]
    assert constructor.call_args.args[0] == tmp_path / "model"


def test_lfs_pointer_has_actionable_error(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"projector_path": "model.safetensors"}))
    (tmp_path / "model.safetensors").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        SmolBGEEmbedder.from_pretrained(tmp_path)


def test_missing_local_path_does_not_request_hub(tmp_path):
    with patch("smolbge_image_embedding.modeling.snapshot_download") as download:
        with pytest.raises(FileNotFoundError):
            SmolBGEEmbedder.from_pretrained(tmp_path / "missing")
    download.assert_not_called()
