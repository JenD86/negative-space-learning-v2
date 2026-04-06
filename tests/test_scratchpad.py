import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.scratchpad import CrossEpisodeScratchpad


class ScratchpadTests(unittest.TestCase):
    def test_snapshot_storage_wraps_saved_payload_with_episode_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            storage_path = base_dir / "checkpoint.json"
            snapshot_path = base_dir / "snapshots" / "episode_0000.json"
            storage_path.write_text(
                json.dumps(
                    {
                        "content": "remember this",
                        "char_count": 13,
                        "max_chars": 1000,
                        "last_updated": "2026-04-06T00:00:00",
                    }
                ),
                encoding="utf-8",
            )

            created_path = CrossEpisodeScratchpad.snapshot_storage(
                storage_path,
                snapshot_path,
                episode_id="ep-0",
                episode_index=0,
                generation_id=7,
                snapshot_reason="episode_complete",
            )

            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

            self.assertEqual(created_path, snapshot_path)
            self.assertEqual(payload["episode_id"], "ep-0")
            self.assertEqual(payload["episode_index"], 0)
            self.assertEqual(payload["generation_id"], 7)
            self.assertEqual(payload["snapshot_reason"], "episode_complete")
            self.assertEqual(payload["scratchpad"]["content"], "remember this")


if __name__ == "__main__":
    unittest.main()
