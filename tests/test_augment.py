from toxicity.augment import neutral_sentences
from toxicity.fairness import load_terms, template_sentences


def test_augmentation_differs_from_evaluation_templates():
    terms = load_terms()
    evaluation = {s for _, s in template_sentences(terms["categories"], terms["templates"])}
    augmented = set(neutral_sentences(terms, ["{Term} — обычный человек."]))
    assert augmented and not augmented & evaluation


def test_pronoun_agrees_with_gender():
    terms = {"categories": {"gender": {"terms": ["женщина", "мужчина"]}}}
    sentences = neutral_sentences(terms, ["{pron} {term}"])
    assert sentences == ["она женщина", "он мужчина"]
