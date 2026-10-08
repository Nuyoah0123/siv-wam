import json
import tempfile
import unittest
from pathlib import Path

from c2s.seeds import load_seed_map, seed_for_episode


class SeedMappingTests(unittest.TestCase):
    def test_text_seed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.txt"
            path.write_text("3 7 12", encoding="utf-8")
            self.assertEqual(load_seed_map(str(path)), {0: 3, 1: 7, 2: 12})

    def test_json_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.json"
            path.write_text(json.dumps({"episodes": {"0": 11, "4": 29}}), encoding="utf-8")
            mapping = load_seed_map(str(path))
            self.assertEqual(seed_for_episode(mapping, 4), 29)

    def test_missing_episode_fails(self):
        with self.assertRaises(KeyError):
            seed_for_episode({0: 3}, 1)


if __name__ == "__main__":
    unittest.main()
