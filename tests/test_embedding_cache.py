import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from embedding_cache import prepare_parts


class EmbeddingCacheTests(unittest.TestCase):
    def test_resume_requires_same_text_order_and_settings(self):
        items = [({"id": "a"}, "first"), ({"id": "b"}, "second")]
        settings = {"model": "test", "dims": 2}
        with tempfile.TemporaryDirectory() as folder:
            prepare_parts(folder, items, settings)
            prepare_parts(folder, items, settings)
            for changed_items, changed_settings in [
                (items[:1], settings), (items[::-1], settings),
                ([({"id": "a"}, "edited"), items[1]], settings), (items, {"model": "new", "dims": 2}),
            ]:
                with self.assertRaisesRegex(ValueError, "changed"):
                    prepare_parts(folder, changed_items, changed_settings)

    def test_unverified_existing_parts_are_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            part = Path(folder) / "part-0000.npy"
            part.write_bytes(b"existing part")
            with self.assertRaisesRegex(ValueError, "no input manifest"):
                prepare_parts(folder, [({"id": "a"}, "text")], {"model": "test"})
            self.assertEqual(part.read_bytes(), b"existing part")


if __name__ == "__main__":
    unittest.main()
