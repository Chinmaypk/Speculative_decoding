import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_gemma4_draft import example_to_text, load_texts, read_jsonl


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last_args = (tokenize, add_generation_prompt)
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class DraftDataTests(unittest.TestCase):
    def test_text_example(self):
        self.assertEqual(example_to_text({"text": "hello"}), "hello")

    def test_prompt_completion_example(self):
        self.assertEqual(
            example_to_text({"prompt": "Question: ", "completion": "Answer"}),
            "Question: Answer",
        )

    def test_messages_example_uses_chat_template(self):
        processor = FakeProcessor()
        text = example_to_text(
            {"messages": [{"role": "user", "content": "Hi"}]},
            processor,
        )
        self.assertEqual(text, "user: Hi")
        self.assertEqual(processor.last_args, (False, False))

    def test_load_texts_limits_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(
                "\n".join(json.dumps({"text": f"row {idx}"}) for idx in range(3)),
                encoding="utf-8",
            )
            examples = load_texts(path, processor=None, max_examples=2)
        self.assertEqual([example.text for example in examples], ["row 0", "row 1"])

    def test_read_jsonl_rejects_non_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("[1, 2, 3]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read_jsonl(path))


if __name__ == "__main__":
    unittest.main()
