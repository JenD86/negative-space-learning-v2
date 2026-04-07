import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestQloraExports(unittest.TestCase):
    def test_resolve_training_export_format_auto_for_vllm(self) -> None:
        from src.train.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format("vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"),
            "peft",
        )

    def test_resolve_training_export_format_auto_for_llama(self) -> None:
        from src.train.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format(
                "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
            ),
            "gguf",
        )

    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_export_lora_adapter_saves_adapter_config(
        self,
        mock_load_base_model: MagicMock,
        mock_attach_lora_adapter: MagicMock,
    ) -> None:
        from src.train.qlora import export_lora_adapter

        model = MagicMock(spec=["save_pretrained_merged"])
        tokenizer = MagicMock()

        def save_pretrained_merged(
            output_dir: Path,
            tokenizer_arg: MagicMock,
            save_method: str,
        ) -> None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_path / "adapter_model.safetensors").write_text(
                "weights", encoding="utf-8"
            )

        model.save_pretrained_merged.side_effect = save_pretrained_merged
        mock_load_base_model.return_value = (model, tokenizer)
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

        model.save_pretrained_merged.assert_called_once()
        self.assertEqual(
            model.save_pretrained_merged.call_args.kwargs["save_method"], "lora"
        )

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
        mock_load_base_model.return_value = (model, tokenizer)
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


class TestLoadSftDataset(unittest.TestCase):
    def _make_jsonl(self, rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _make_tokenizer(self):
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            user_content = messages[0]["content"]
            asst_content = messages[1]["content"]
            return f"<user>{user_content}</user><asst>{asst_content}</asst>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        return tokenizer

    def _make_row(self, *, success: bool = True, **kwargs) -> dict:
        base = {
            "prompt": "Do something",
            "raw_response": "Done",
            "timestamp": "2026-04-06T00:00:00",
            "interaction_type": "orchestrator",
            "success": success,
            "error_message": None,
            "episode_id": "ep_gen0_0001_123",
            "generation_id": 0,
            "episode_space_freed_kb": 100.0,
        }
        base.update(kwargs)
        return base

    def test_load_sft_dataset_from_jsonl(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(), self._make_row()], path)
            dataset = _load_sft_dataset([path], tokenizer)
        self.assertGreater(len(dataset), 0)
        self.assertIn("text", dataset.column_names)

    def test_load_sft_dataset_multiple_files(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path0 = Path(tmpdir) / "gen0" / "sft_training_rows.jsonl"
            path1 = Path(tmpdir) / "gen1" / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(generation_id=0)], path0)
            self._make_jsonl([self._make_row(generation_id=1)], path1)
            dataset = _load_sft_dataset([path0, path1], tokenizer)
        self.assertEqual(len(dataset), 2)

    def test_load_sft_dataset_empty_input(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([], path)
            with self.assertRaises(ValueError, msg="No training rows found"):
                _load_sft_dataset([path], tokenizer)

    def test_load_sft_dataset_filters_unsuccessful(self):
        from src.train.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [self._make_row(success=True), self._make_row(success=False)], path
            )
            dataset = _load_sft_dataset([path], tokenizer)
        self.assertEqual(len(dataset), 1)


class TestTrainSft(unittest.TestCase):
    def _make_jsonl(self, rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _make_row(self) -> dict:
        return {
            "prompt": "Do something",
            "raw_response": "Done",
            "timestamp": "2026-04-06T00:00:00",
            "interaction_type": "orchestrator",
            "success": True,
            "error_message": None,
            "episode_id": "ep_gen0_0001_123",
            "generation_id": 0,
            "episode_space_freed_kb": 100.0,
        }

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_runs_trainer(
        self, mock_load_base_model, mock_attach_lora_adapter, mock_sft_trainer_cls
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        tokenizer.eos_token = "<|endoftext|>"
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model

        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            result = train_sft(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
            )

        trainer_instance.train.assert_called_once()
        self.assertEqual(result, output_dir.resolve())

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._save_training_artifact")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_passes_export_format_to_artifact_saver(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "artifact"
            mock_save_training_artifact.return_value = output_dir.resolve()

            train_sft(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                export_format="merged_16bit",
            )

        mock_save_training_artifact.assert_called_once()
        self.assertEqual(
            mock_save_training_artifact.call_args.kwargs["export_format"],
            "merged_16bit",
        )

    @patch("src.train.qlora.SFTTrainer")
    @patch("src.train.qlora._attach_lora_adapter")
    @patch("src.train.qlora._load_base_model")
    def test_train_sft_writes_metadata(
        self, mock_load_base_model, mock_attach_lora_adapter, mock_sft_trainer_cls
    ):
        from src.train.qlora import train_sft

        model = MagicMock()
        tokenizer = MagicMock()

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<text>{messages[0]['content']}</text>"

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        tokenizer.eos_token = "<|endoftext|>"
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(), self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            train_sft(
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                learning_rate=1e-4,
            )

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        self.assertEqual(metadata["base_model"], "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(metadata["learning_rate"], 1e-4)
        self.assertEqual(metadata["row_count"], 2)
        self.assertIn("training_data_paths", metadata)
        self.assertIn("exported_at", metadata)
        self.assertIn("max_steps", metadata)

    def test_train_sft_cli_multiple_training_data(self):
        """CLI --train mode should accept multiple --training-data arguments."""
        from src.train.qlora import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--train",
                "--base-model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--training-data",
                "./data/gen0/sft.jsonl",
                "--training-data",
                "./data/gen1/sft.jsonl",
                "--output",
                "./models/adapters/test",
            ]
        )
        self.assertTrue(args.train)
        self.assertEqual(len(args.training_data), 2)
        self.assertIn("./data/gen0/sft.jsonl", args.training_data)
        self.assertIn("./data/gen1/sft.jsonl", args.training_data)


if __name__ == "__main__":
    unittest.main()
