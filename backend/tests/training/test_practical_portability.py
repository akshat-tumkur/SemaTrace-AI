"""Byte-level portability tests; no model or real evaluation predictions."""

import hashlib, json, tempfile, unittest, sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training_practical import data


class Portability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / "original"
        self.target = self.base / "practical"
        (self.source / "prepared").mkdir(parents=True)
        self.target.mkdir()
        self.canonical = {
            s: ('{"id":"' + s + '","label":1}\r\n{"label":0}\r\n').encode()
            for s in data.FROZEN
        }
        self.hashes = {
            s: hashlib.sha256(b).hexdigest() for s, b in self.canonical.items()
        }
        self.patcher = patch.object(data, "FROZEN", self.hashes)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def write(self, newline):
        manifest = {
            "campaign": "original",
            "splits": {},
            "archives": {"train": "unchanged"},
            "exposure": "retained",
        }
        for s, b in self.canonical.items():
            raw = b.replace(b"\r\n", newline)
            (self.source / "prepared" / f"{s}.jsonl").write_bytes(raw)
            manifest["splits"][s] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "pairs": 2,
            }
        (self.source / "prepared/manifest.json").write_bytes(
            json.dumps(manifest).encode()
        )
        (self.source / "test-opened.json").write_bytes(b'{"original_seal":true}')
        return {p.name: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}

    def test_lf_and_crlf_produce_identical_frozen_bytes_without_changing_source(self):
        for separator in (b"\n", b"\r\n"):
            before = self.write(separator)
            result = data.stage_frozen(self.source, self.target)
            self.assertEqual(
                {p.name: p.read_bytes() for p in self.source.rglob("*") if p.is_file()},
                before,
            )
            for s, b in self.canonical.items():
                self.assertEqual(
                    (self.target / "prepared" / f"{s}.jsonl").read_bytes(), b
                )
                self.assertEqual(result["splits"][s]["sha256"], self.hashes[s])
            self.assertEqual(result["archives"], {"train": "unchanged"})
            self.assertEqual(result["exposure"], "retained")

    def test_manifest_mismatch_and_non_newline_content_changes_fail_before_publication(
        self,
    ):
        self.write(b"\n")
        p = self.source / "prepared/train.jsonl"
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            data.stage_frozen(self.source, self.target)
        self.assertFalse((self.target / "prepared").exists())
        manifest = json.loads((self.source / "prepared/manifest.json").read_text())
        manifest["splits"]["train"]["sha256"] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        (self.source / "prepared/manifest.json").write_bytes(
            json.dumps(manifest).encode()
        )
        with self.assertRaises(ValueError):
            data.stage_frozen(self.source, self.target)
        self.assertFalse((self.target / "prepared").exists())

    def test_existing_partial_practical_lf_setup_repaired_but_sealed_setup_blocked(
        self,
    ):
        self.write(b"\n")
        import shutil

        shutil.copytree(self.source / "prepared", self.target / "prepared")
        (self.target / "subset").mkdir()
        (self.target / "subset/selection.json").write_bytes(b"unchanged subset")
        data.ensure_frozen(self.target)
        self.assertEqual(
            (self.target / "subset/selection.json").read_bytes(), b"unchanged subset"
        )
        self.write(b"\n")
        shutil.copytree(
            self.source / "prepared", self.target / "prepared", dirs_exist_ok=True
        )
        (self.target / "test-opened.json").write_bytes(b"preserve")
        with self.assertRaises(RuntimeError):
            data.ensure_frozen(self.target)
        self.assertEqual((self.target / "test-opened.json").read_bytes(), b"preserve")

    def test_standalone_colab_cell_repairs_lf_and_is_idempotent(self):
        import ast

        self.write(b"\n")
        (
            self.source / "test-opened.json"
        ).unlink()  # Own fixture seal; production cell never removes seals.
        before = {p.name: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        path = (
            Path(__file__).resolve().parents[3]
            / "notebooks/colab_fix_prepared_newlines.py"
        )
        # Substitute only test fixture frozen hashes; exercise the actual cell logic.
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "expected" for t in node.targets
            ):
                node.value = ast.parse(repr(self.hashes), mode="eval").body
        compiled = compile(ast.fix_missing_locations(tree), str(path), "exec")
        scope = {"ROOT": self.target, "PRIOR": self.source}
        exec(compiled, scope)
        exec(compiled, scope)
        self.assertEqual(
            {p.name: p.read_bytes() for p in self.source.rglob("*") if p.is_file()},
            before,
        )
        for s, b in self.canonical.items():
            self.assertEqual((self.target / "prepared" / f"{s}.jsonl").read_bytes(), b)


if __name__ == "__main__":
    unittest.main()
