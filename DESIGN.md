# Document Processing System Flow

## Flow

1. **Upload**: Clients upload documents via API (We could use FastAPI)
2. **OCR**: Perform OCR to extract raw text
3. **Schema Selection**: Analyze text content to determine a document type and select the appropriate schema. This could be done using LLM to classify the document based on keywords, structure, etc.
4. **Storage**: Store raw text and metadata in a database (PostgreSQL) before processing.
5. **LLM Extraction**: Use OpenAI API with structured prompts to extract fields according to selected schema
6. **Validation**: Parse LLM output to JSON, then validate strictly against schema. Enforce no extra fields, types, required props, etc.
7. **Persistence**: On success, store validated data and metadata in DB. On failure, log and retry once before final error.

## Schema Registry

- Schemas stored as JSON files in `schemas/` folder, loaded into dict {schema_id: schema_dict} ('w2': {...}).

## Error Handling

- **LLM Failure**: Catch API errors (e.g., rate limits), retry once after delay.
- **Invalid JSON**: On parse failure, retry with stricter prompt (emphasize JSON format).
- **Validation Failure**: Retry once with refined prompt (e.g., "strictly no extra fields").
- **Classification Failure**: If no match, raise an error.
- **Retry**: Exactly one retry on validation/parse failure. On final failure, raise error with summary.

## Metrics/Logs for Observability

- Attempt number (1 or retry).
- Selected schema_id + version.
- Prompt size (tokens).
- Response size (tokens).
- Failure stage.