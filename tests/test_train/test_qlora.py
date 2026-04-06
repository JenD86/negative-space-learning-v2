import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestQloraExports(unittest.TestCase):
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_export_lora_adapter_saves_adapter_config(
        self,
        mock_load_base_model: MagicMock,
        mock_attach_lora_adapter: MagicMock,
    ) -> None:
        from src.train.qlora import export_lora_adapter

        model = MagicMock(spec=["save_pretrained"])
        tokenizer = MagicMock()

        def save_pretrained(output_dir: str) -> None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_path / "adapter_model.safetensors").write_text(
                "weights", encoding="utf-8"
            )

        model.save_pretrained.side_effect = save_pretrained
        mock_load_base_model.return_value = (model, tokenizer, object())
        mock_attach_lora_adapter.return_value = model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "adapter"
            exported_dir = export_lora_adapter(
                "unsloth/Qwen2.5-7B",
                str(output_dir),
            )
            self.assertEqual(exported_dir, output_dir.resolve())
            self.assertTrue((exported_dir / "adapter_config.json").exists())
            self.assertTrue((exported_dir / "adapter_model.safetensors").exists())
            metadata = json.loads((exported_dir / "adapter_info.json").read_text())
            self.assertEqual(metadata["base_model"], "unsloth/Qwen2.5-7B")
            self.assertEqual(metadata["export_method"], "lora")

    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_export_merged_model_writes_model_info_json(
        self,
        mock_load_base_model: MagicMock,
        mock_attach_lora_adapter: MagicMock,
    ) -> None:
        from src.train.qlora import export_merged_model

        model = MagicMock(spec=["save_pretrained_merged"])
        tokenizer = MagicMock()

        def save_pretrained_merged(
            output_dir: str,
            tokenizer_arg: MagicMock,
            save_method: str,
        ) -> None:
            self.assertIs(tokenizer_arg, tokenizer)
            self.assertEqual(save_method, "merged_16bit")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "model.safetensors").write_text("weights", encoding="utf-8")

        model.save_pretrained_merged.side_effect = save_pretrained_merged
        mock_load_base_model.return_value = (model, tokenizer, object())
        mock_attach_lora_adapter.return_value = model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "merged"
            exported_dir = export_merged_model(
                "unsloth/Qwen2.5-7B",
                str(output_dir),
            )
            self.assertEqual(exported_dir, output_dir.resolve())
            metadata = json.loads((exported_dir / "model_info.json").read_text())
            self.assertEqual(metadata["base_model"], "unsloth/Qwen2.5-7B")
            self.assertEqual(metadata["export_method"], "merged_16bit")

    @patch("src.train.qlora.subprocess.run")
    @patch("src.train.qlora._resolve_convert_lora_script")
    def test_convert_adapter_to_gguf_calls_subprocess(
        self,
        mock_resolve_script: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        from src.train.qlora import convert_adapter_to_gguf

        mock_resolve_script.return_value = Path("/tmp/convert_lora_to_gguf.py")
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_path = Path(tmpdir) / "adapter.gguf"

            converted_path = convert_adapter_to_gguf(
                str(adapter_dir),
                str(output_path),
                quantize="f16",
            )

        self.assertEqual(converted_path, output_path.resolve())
        command = mock_run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("--outfile", command)
        self.assertIn(str(output_path.resolve()), command)
        self.assertIn("--outtype", command)
        self.assertIn("f16", command)
        self.assertIn(str(adapter_dir.resolve()), command)


if __name__ == "__main__":
    unittest.main()
