from oriel.grammar import JSONSchemaLogitMasker


def test_json_schema_logit_masker():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    vocab = {"{": 0, "}": 1, '"': 2, "name": 3}
    masker = JSONSchemaLogitMasker(schema, vocab)

    mask = masker.compute_mask("")
    assert len(mask) == len(vocab)
    assert mask[0] == 0.0  # '{' token allowed
    assert mask[1] == -1e9  # '}' token masked
    assert mask[2] == -1e9  # '"' token masked
    assert mask[3] == -1e9  # 'name' token masked
