from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional


class CrossEpisodeScratchpad:
    def __init__(self, max_chars: int = 10000, storage_path: Optional[Path] = None):
        self.max_chars = max_chars
        self.storage_path = storage_path or Path("./data/scratchpad.json")
        self.content: str = ""
        self.load()

    def append(self, text: str, episode_id: Optional[str] = None) -> None:
        """Add new content with optional episode marker."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if episode_id:
            entry = f"\n[{timestamp} | Episode {episode_id}] {text}"
        else:
            entry = f"\n[{timestamp}] {text}"

        self.content += entry

        # Trim from start if over limit (FIFO)
        if len(self.content) > self.max_chars:
            # Find a good cut point (try to cut at episode boundaries)
            excess = len(self.content) - self.max_chars
            cut_point = excess

            # Look for episode boundary within reasonable range
            search_start = max(0, excess - 200)
            search_end = min(len(self.content), excess + 200)
            episode_marker = self.content.find("\n[", search_start)

            if episode_marker != -1 and episode_marker < search_end:
                cut_point = episode_marker

            self.content = self.content[cut_point:].lstrip()

        self.save()

    def get_content(self) -> str:
        """Get current scratchpad content."""
        return self.content

    def snapshot(self, snapshot_path: Path) -> None:
        """Save a copy of the current scratchpad state to snapshot_path."""
        try:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "content": self.content,
                        "char_count": len(self.content),
                        "max_chars": self.max_chars,
                        "snapshot_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: Failed to snapshot scratchpad: {e}")

    def clear(self) -> None:
        """Clear all content (for testing/reset)."""
        self.content = ""
        self.save()

    @staticmethod
    def load_storage_payload(storage_path: Path) -> Optional[dict[str, Any]]:
        """Load a persisted scratchpad payload from disk."""
        if not storage_path.exists():
            return None
        with open(storage_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Scratchpad payload must be a JSON object")
        return payload

    @classmethod
    def snapshot_storage(
        cls,
        storage_path: Path,
        snapshot_path: Path,
        *,
        episode_id: Optional[str],
        episode_index: int,
        generation_id: int,
        snapshot_reason: str,
    ) -> Optional[Path]:
        """Archive the current persisted scratchpad for later episode analysis."""
        try:
            scratchpad_payload = cls.load_storage_payload(storage_path)
            if scratchpad_payload is None:
                return None

            snapshot_payload = {
                "generation_id": generation_id,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "snapshot_reason": snapshot_reason,
                "source_path": str(storage_path),
                "snapshotted_at": datetime.now().isoformat(),
                "scratchpad": scratchpad_payload,
            }
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(snapshot_payload, handle, indent=2)
            return snapshot_path
        except Exception as e:
            # Don't fail episode on snapshot save error
            print(f"Warning: Failed to snapshot scratchpad: {e}")
            return None

    def save(self) -> None:
        """Persist scratchpad to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "content": self.content,
                        "char_count": len(self.content),
                        "max_chars": self.max_chars,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            # Don't fail episode on scratchpad save error
            print(f"Warning: Failed to save scratchpad: {e}")

    def load(self) -> None:
        """Load scratchpad from storage."""
        try:
            payload = self.load_storage_payload(self.storage_path)
            if payload is not None:
                self.content = str(payload.get("content", ""))
                # Update max_chars if changed in config
                if len(self.content) > self.max_chars:
                    self.append("", None)  # Trigger trim
        except Exception as e:
            # Start fresh if load fails
            self.content = ""
            print(f"Warning: Failed to load scratchpad, starting fresh: {e}")
