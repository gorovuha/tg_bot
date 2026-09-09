from bot.handlers import parse_analyze_args


def test_parse_analyze_args():
    assert parse_analyze_args(None) == (10, None)
    assert parse_analyze_args([]) == (10, None)
    assert parse_analyze_args(["20"]) == (20, None)
    assert parse_analyze_args(["999"]) == (50, None)
    assert parse_analyze_args(["2022", "was", "hard"]) == (None, "2022 was hard")
    assert parse_analyze_args(["nato", "lies"]) == (None, "nato lies")
