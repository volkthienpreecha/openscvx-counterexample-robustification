from pathlib import Path


def pytest_configure(config):
    Path(".cache").mkdir(exist_ok=True)
