import json
import pytest
from src.extractor import (
    extract,
    ExtractionError,
    ParseError,
    SchemaValidationError,
    classify_document,
)


class FakeResponse:
    def __init__(self, text: str):
        self.output_text = text

class FakeClient:
    """Fake OpenAI client to mimic responses.create(...)"""
    def __init__(self, responses):
        if isinstance(responses, list):
            self._responses = responses
        else:
            self._responses = [responses]
        self.calls = 0

    @property
    def responses(self):
        return self

    def create(self, *args, **kwargs):
        if self.calls >= len(self._responses):
            raise RuntimeError("No more fake responses available")
        out = self._responses[self.calls]
        self.calls += 1
        return FakeResponse(out)


@pytest.fixture
def registry():
    return {
        "w2": {
            "$id": "w2",
            "title": "W2 Form",
            "version": "1.0",
            "type": "object",
            "properties": {
                "employer_name": {"type": "string"},
                "employee_name": {"type": "string"},
                "wages": {"type": "number"},
            },
            "required": ["employer_name", "employee_name", "wages"],
            "additionalProperties": False,
        },
        "1040": {
            "$id": "1040",
            "title": "1040 Form",
            "version": "1.0",
            "type": "object",
            "properties": {
                "taxpayer_name": {"type": "string"},
                "income": {"type": "number"},
            },
            "required": ["taxpayer_name", "income"],
            "additionalProperties": False,
        },
    }


def test_extract_success_plain_json(registry):
    text = "This looks like a W2 form"
    llm_client = FakeClient(json.dumps({
        "employer_name": "Acme Corp",
        "employee_name": "Jane Doe",
        "wages": 50000
    }))

    result = extract(text, registry, llm_client, schema=registry["w2"], model="fake")
    assert result["doc_type"] == "w2"
    assert result["data"]["employee_name"] == "Jane Doe"
    assert result["schema_version"] == "1.0"

def test_extract_success_fenced_json(registry):
    text = "This looks like a W2 form"
    fenced_json = """```json
{
  "employer_name": "Acme Corp",
  "employee_name": "Jane Doe",
  "wages": 50000
}
```"""
    llm_client = FakeClient(fenced_json)

    result = extract(text, registry, llm_client, schema=registry["w2"], model="fake")
    assert result["doc_type"] == "w2"
    assert result["data"]["wages"] == 50000

def test_extract_retry_then_success(registry):
    text = "This looks like a W2 form"
    bad_json = json.dumps({"employer_name": "Acme Corp"})
    good_json = json.dumps({
        "employer_name": "Acme Corp",
        "employee_name": "Jane Doe",
        "wages": 50000
    })
    llm_client = FakeClient([bad_json, good_json])

    result = extract(text, registry, llm_client, schema=registry["w2"], model="fake")
    assert result["doc_type"] == "w2"
    assert llm_client.calls == 2

def test_extract_classification_failure(registry):
    text = "Completely unrelated document"
    llm_client = FakeClient("{}")

    with pytest.raises(ExtractionError) as exc:
        extract(text, registry, llm_client, model="fake")
    assert "classify" in str(exc.value)

def test_extract_parse_failure(registry):
    text = "This looks like a 1040 form"
    llm_client = FakeClient("NOT JSON")

    with pytest.raises(ExtractionError) as exc:
        extract(text, registry, llm_client, schema=registry["1040"], model="fake")
    assert "parse" in str(exc.value)

def test_extract_schema_validation_failure(registry):
    text = "This looks like a 1040 form"
    bad_json = json.dumps({"taxpayer_name": "John"})
    llm_client = FakeClient([bad_json, bad_json])

    with pytest.raises(ExtractionError) as exc:
        extract(text, registry, llm_client, schema=registry["1040"], model="fake")
    assert "validate" in str(exc.value)

def test_classify_success_w2(registry):
    """Classification returns w2 schema_id."""
    client = FakeClient("w2")
    text = "This is a W2 document"
    schema_id = classify_document(text, registry, client, model="fake")
    assert schema_id == "w2"

def test_classify_success_1040(registry):
    """Classification returns 1040 schema_id."""
    client = FakeClient("1040")
    text = "This is a 1040 tax form"
    schema_id = classify_document(text, registry, client, model="fake")
    assert schema_id == "1040"

def test_classify_unknown_schema(registry):
    """Raises error if LLM returns unknown schema_id."""
    client = FakeClient("unknown_schema")
    text = "Random text"
    with pytest.raises(ExtractionError) as exc:
        classify_document(text, registry, client, model="fake")
    assert "Unknown schema_id" in str(exc.value)