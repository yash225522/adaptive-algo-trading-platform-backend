"""Minimal test verifying that the adaptive_trading package can be imported."""


def test_package_import() -> None:
    import adaptive_trading

    assert hasattr(adaptive_trading, "__version__")
    assert adaptive_trading.__version__ == "0.1.0"
