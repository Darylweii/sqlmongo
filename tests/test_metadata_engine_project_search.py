from src.metadata.metadata_engine import MetadataEngine


def test_search_projects_matches_name_plus_code_compound() -> None:
    engine = MetadataEngine("sqlite:///unused.db")
    engine.list_projects = lambda: [
        {"id": 15, "project_name": "测试项目", "project_code_name": "123456", "code_name": "123456"},
        {"id": 51, "project_name": "测试项目", "project_code_name": "111", "code_name": "111"},
        {"id": 70, "project_name": "测试项目", "project_code_name": "123", "code_name": "123"},
    ]

    results = engine.search_projects("测试项目111", limit=5)

    assert results
    assert results[0]["id"] == 51
    assert results[0]["matched_fields"] == ["project_compound"]
