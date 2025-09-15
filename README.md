# Document Extraction Feature

This project provides a **schema-driven document extraction pipeline** using OpenAI LLMs.  
It supports classification, JSON extraction, schema validation, and observability (token metrics).

## How to run

### Requirements

-   Python 3.12+
-   Poetry for dependency management (see https://python-poetry.org/docs/ for installation instructions)
-   OpenAI API key

### Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/OlhaKashyrina/lira-test-task.git
    cd <repo-directory>
    ```
3. **Install dependencies via Poetry**
    ```bash
   poetry install
    ```
5. **Activate virtual environment**
    ```bash
   poetry shell
    ```
7. **Set your OpenAI key**
 Create `.env` file in your repo directory, and define `OPENAI_API_KEY` variable (see `.env.dist` for example)

### Running example

1. (Optional) Change `text` in `src/example.py` to your desired input text.
2. From your repo directory, run `poetry run python src/example.py`

### Running mock tests

From your repo directory, run `poetry run pytest`
