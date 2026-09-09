from services.classifier import mock_classify


def test_deterministic():
    text = "NATO provoked this war"
    a, b = mock_classify(text), mock_classify(text)
    assert a == b
    assert a.is_propaganda and a.cluster_id == "cluster_nato"


def test_word_boundaries():
    assert not mock_classify("The senator spoke about the budget").is_propaganda
    assert not mock_classify("Ashkenazi cuisine is great").is_propaganda


def test_clean_text():
    r = mock_classify("Weather in Kyiv is sunny today")
    assert not r.is_propaganda and r.narrative_label == "None detected"


def test_russian_and_ukrainian():
    assert mock_classify("В Украине нашли биолаборатории США").cluster_id == "cluster_biolabs"
    assert mock_classify("Київський режим марионетки Заходу").is_propaganda
    assert mock_classify("Українські біолабораторії").cluster_id == "cluster_biolabs"


def test_more_hits_more_confidence():
    one = mock_classify("nazis everywhere")
    two = mock_classify("nazis run the kiev regime with NATO help")
    assert two.confidence > one.confidence <= 0.95
