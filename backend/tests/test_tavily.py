from app.retrieval.tavily import TavilySearch


def test_tavily_without_key_degrades_to_empty_results() -> None:
    client = TavilySearch(api_key="", enabled=True)

    assert client.search("voice cloning") == []
    assert client.last_error is None
