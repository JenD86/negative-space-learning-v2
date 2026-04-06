import unittest

from result import Err

from src.genner.Base import Genner
from src.observability.types import UsageInfo


class DummyGenner(Genner):
    def __init__(self) -> None:
        super().__init__("dummy")

    def plist_completion(self, messages):
        return Err("unused")

    def generate_code(self, messages):
        return Err(("unused", None))

    def generate_list(self, messages):
        return Err(("unused", None))

    @staticmethod
    def extract_code(response):
        return Err("unused")

    @staticmethod
    def extract_list(response):
        return Err("unused")

    @staticmethod
    def get_usage_info(response: object) -> UsageInfo:
        return UsageInfo()


class GennerBaseTests(unittest.TestCase):
    def test_genner_exposes_client_and_collector_interface(self) -> None:
        genner = DummyGenner()

        self.assertIsNone(genner.client)
        self.assertIsNone(genner.collector)


if __name__ == "__main__":
    unittest.main()
