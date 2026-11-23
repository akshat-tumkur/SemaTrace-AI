import pytest
from app.storage import docker_seed


def test_seed_preserves_original_and_never_overwrites_live_volume(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "corpus.sqlite3").write_bytes(b"original immutable corpus")
    (source / "blobs").mkdir()
    (source / "blobs" / "fingerprint").write_bytes(b"reference")
    volume = tmp_path / "volume"
    volume.mkdir()
    monkeypatch.setattr(docker_seed.os, "chown", lambda *args: None, raising=False)
    docker_seed.main(source, volume)
    assert (volume / "data" / "blobs" / "fingerprint").read_bytes() == b"reference"
    (volume / "data" / "corpus.sqlite3").write_bytes(b"new Docker reference state")
    docker_seed.main(source, volume)
    assert (
        volume / "data" / "corpus.sqlite3"
    ).read_bytes() == b"new Docker reference state"
    assert (source / "corpus.sqlite3").read_bytes() == b"original immutable corpus"


def test_failed_seed_is_not_published_and_can_resume(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "corpus.sqlite3").write_bytes(b"complete")
    volume = tmp_path / "volume"
    volume.mkdir()
    monkeypatch.setattr(docker_seed.os, "chown", lambda *args: None, raising=False)
    original = docker_seed.shutil.copytree

    def interrupted(source, target, **options):
        target.mkdir()
        (target / "corpus.sqlite3").write_bytes(b"partial")
        raise OSError("interrupted copy")

    monkeypatch.setattr(docker_seed.shutil, "copytree", interrupted)
    with pytest.raises(OSError):
        docker_seed.main(source, volume)
    assert not (volume / "data").exists()
    monkeypatch.setattr(docker_seed.shutil, "copytree", original)
    docker_seed.main(source, volume)
    assert (volume / "data" / "corpus.sqlite3").read_bytes() == b"complete"
