"""Pytest configuration for pinocchio tests."""



def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU (skipped without CUDA)")
