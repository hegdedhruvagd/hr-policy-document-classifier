from app.core import CATEGORIES, HR_EXAMPLES, LABEL_TOKENS, milestone_1


def test_milestone_1():
    result = milestone_1()
    assert len(result["categories"]) == 5
    assert len(result["examples"]) == 25
    assert all(item["category"] in CATEGORIES for item in result["examples"])


def test_milestone_2_token_definitions():
    assert len(LABEL_TOKENS) == 5
    assert all(token.startswith("<|") and token.endswith("|>") for token in LABEL_TOKENS)


def test_training_data_categories():
    assert {item["category"] for item in HR_EXAMPLES} == set(CATEGORIES)
