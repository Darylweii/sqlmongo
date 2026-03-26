from src.cache.cache_manager import CacheManager


def test_generate_key_differs_by_query_shape() -> None:
    manager = CacheManager("redis://unused")

    base = manager._generate_key(["a2_b1"], "2024-01-01", "2024-01-01", data_type="u_line", tags=["ua"], output_format="json")
    different_tag = manager._generate_key(["a2_b1"], "2024-01-01", "2024-01-01", data_type="u_line", tags=["ub"], output_format="json")
    different_format = manager._generate_key(["a2_b1"], "2024-01-01", "2024-01-01", data_type="u_line", tags=["ua"], output_format="minimal")

    assert base != different_tag
    assert base != different_format
