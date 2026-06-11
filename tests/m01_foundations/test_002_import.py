"""验证 zllm 包可导入。"""


def test_zllm_importable():
    import zllm

    assert zllm.__version__ == "0.0.1"


def test_config_importable():
    from zllm import ZLLMConfig

    assert ZLLMConfig is not None
