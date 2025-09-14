def test_public_imports():
    from itpu.sdk import ITPU
    from itpu.utils.windowed import windowed_mi
    assert ITPU is not None and callable(windowed_mi)
