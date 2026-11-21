"""A Git LF checkout and original Windows CRLF fixture retain the same seal."""

import importlib.util


from pathlib import Path


import pytest


ROOT = Path(__file__).resolve().parents[2]


spec = importlib.util.spec_from_file_location(
    "frozen_discovery_fixture", ROOT / "scripts/discovery_fixture.py"
)


module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_fixture_newline_portability_and_label_integrity(tmp_path):
    path = ROOT / "demo/semantic-discovery-v1.json"
    original = path.read_bytes()
    expected = module.load_fixture(path)
    for data in [
        original.replace(b"\r\n", b"\n"),
        original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"),
    ]:
        copy = tmp_path / "fixture.json"
        copy.write_bytes(data)
        assert module.load_fixture(copy) == expected
    copy.write_bytes(original.replace(b"paraphrase", b"wronglabel", 1))
    with pytest.raises(ValueError):
        module.load_fixture(copy)
    assert path.read_bytes() == original
