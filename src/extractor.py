import json
import logging
import re
import tiktoken
from typing import Any, Dict, Optional
from jsonschema import validate, ValidationError
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ExtractionError(Exception): pass
class ClassificationError(ExtractionError): pass
class ModelCallError(ExtractionError): pass
class ParseError(ExtractionError): pass
class SchemaValidationError(ExtractionError): pass


def count_tokens(text: str, model: str) -> int:
    """
    Return number of tokens for a given text and model using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def classify_document(text: str, registry: Dict[str, dict], llm: OpenAI, model: str, metrics: dict | None = None) -> str:
    """
    Use OpenAI API to classify which schema applies, based on document text.
    Returns the schema_id.
    """
    prompt = (
        "You are given document text and a set of available schema IDs. "
        f"Schema IDs: {list(registry.keys())}\n"
        f"Document text: {text[:200] + text[-200:]}...\n"
        "Which schema_id best matches this document? Respond with only the schema_id."
    )

    logger.info(f"Classification started.")
    prompt_tokens = count_tokens(prompt, model)
    if metrics is not None:
        metrics["classification_prompt_tokens"] = prompt_tokens

    response = llm.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )

    raw_output = getattr(response, "output_text", "")
    response_tokens = count_tokens(raw_output, model)
    if metrics is not None:
        metrics["classification_response_tokens"] = response_tokens
        metrics["classification_total_tokens"] = prompt_tokens + response_tokens

    schema_id = raw_output.strip()

    if schema_id not in registry:
        raise ExtractionError(f"Classification failed. Unknown schema_id: {schema_id}")
    
    logger.info(f"Classification successful. Schema chosen: {schema_id}")
    return schema_id


def _clean_json_output(raw: str) -> str:
    """
    Remove common Markdown fences (```json ... ``` or ``` ... ```) 
    and trim whitespace.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    return cleaned.strip()


def call_llm_extract(text: str, schema: dict, llm: OpenAI, model: str = "gpt-4o-mini", metrics: dict | None = None) -> dict:
    """
    Ask LLM to extract structured JSON given a schema.
    """
    prompt = f"""
        You are an information extraction engine.
        Extract the required fields from this document text.

        Document text: {text}

        Output JSON MUST strictly conform to this JSON Schema:
        {json.dumps(schema, indent=2)}
        """.strip()

    logger.info(f"Extraction started.")
    prompt_tokens = count_tokens(prompt, model)
    if metrics is not None:
        metrics["extraction_prompt_tokens"] = prompt_tokens

    try:
        response = llm.responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )
    except Exception as e:
        raise ModelCallError(f"LLM call failed: {e}")

    try:
        raw_output = getattr(response, "output_text", "")
        response_tokens = count_tokens(raw_output, model)
        if metrics is not None:
            metrics["extraction_response_tokens"] = response_tokens
            metrics["extraction_total_tokens"] = prompt_tokens + response_tokens

        cleaned_output = _clean_json_output(raw_output.strip())
        data = json.loads(cleaned_output)
    except Exception as e:
        raise ParseError(f"Could not parse LLM output as JSON: {e}\nRaw output: {raw_output}")
    
    logger.info(f"Extraction successful.")

    return data


def validate_against_schema(data: dict, schema: dict) -> None:
    """
    Validate JSON against given schema.
    """
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        raise SchemaValidationError(f"Schema validation failed: {e.message}")
    
    logger.info(f"Validation successful.")


def extract(
    text: str,
    registry: Dict[str, dict],
    llm: OpenAI,
    schema: Optional[dict] = None,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    End-to-end pipeline.
    """
    attempts = []
    last_error: Optional[Exception] = None
    metrics = {}

    for attempt in range(2):
        logger.info(f"Attempt {attempt+1}")

        # Step 1: select schema
        try:
            if schema is not None:
                schema_id = schema.get("$id", "unknown")
                selected_schema = schema
            else:
                schema_id = classify_document(text, registry, llm, model, metrics)
                selected_schema = registry[schema_id]
        except Exception as e:
            last_error = e
            attempts.append({"stage": "classify", "schema": None, "error": str(e)})
            continue

        schema_version = selected_schema.get("version")

        # Step 2: extract
        try:
            extracted = call_llm_extract(text, selected_schema, llm, model=model, metrics=metrics)
        except ExtractionError as e:
            last_error = e
            attempts.append({"stage": "extract", "schema": schema_id, "error": str(e)})
            continue

        # Step 3: validate
        try:
            validate_against_schema(extracted, selected_schema)
        except SchemaValidationError as e:
            last_error = e
            attempts.append({"stage": "validate", "schema": schema_id, "error": str(e)})
            if attempt == 0:
                continue
            else:
                break

        # Success
        return {
            "doc_type": schema_id,
            "schema_version": schema_version,
            "data": extracted,
            "metrics": metrics,
        }

    # Failure
    raise ExtractionError({
        "message": "Extraction failed after 2 attempts.",
        "attempts": attempts,
        "last_error": str(last_error),
    })
