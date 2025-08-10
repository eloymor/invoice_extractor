# Invoice Data Extractor

This project provides a robust solution for extracting structured data from PDF invoices using various Large Language 
Models (LLMs). It features both a command-line interface for direct processing and a Streamlit web application for an 
interactive user experience, allowing users to upload invoices, review, edit extracted information, and generate 
reports.

It uses the [Docling](https://github.com/docling-project/docling) library to parse unstructured documents and extract text and 
tabular data. Then this data is prepared and sent to the LLM for extraction, using langchain. The LLM's output is parsed and formatted 
into a structured JSON format.

### Notes: 
* This project is still in development. 
* Many features will be canceled or changed in the future.
* The computer which would run this project does not have a compatible GPU, so all models have been moved to use ollama API.
* Ollama uses a GGUP format, it is executed using C++ and will run a little bit faster on the CPU (compared to python - Transformers).
* Frontier models like Gemini and GPT-4o Mini are just for testing purposes. The idea is to just use local models.
* The PydanticOutputParser is not validated after receiving the prompt, it would cause to re-run the prompt again (on CPU, extract a single PDF can last 30 seconds).
* Recommended model for inference time and accuracy: `llama3.2:3B` (via Ollama)


## Features

*   **PDF Data Extraction**: Automatically parses PDF documents to extract text and tabular data.
*   **LLM Integration**: Leverages different LLMs (Qwen3, Llama3.2, Deepseek, GPT-4o Mini, Gemini) for intelligent invoice data extraction.
*   **Structured Output**: Extracts key invoice details such as invoice number, date, issuer information, and total amounts into a structured JSON format.
*   **Editable Extracted Data**: The Streamlit application allows users to review and modify the extracted data for accuracy.
*   **Configurable Context**: Easily customize context words and items to find for improved extraction accuracy via the Streamlit interface.
*   **Report Generation**: Generates a formatted report from the extracted invoice data.

## Installation

This project requires Python 3.12.6 or higher. It uses `uv` as the package manager.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/eloymor/invoice_extractor.git
    cd invoice-extractor
    ```

2.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```
    *(Note: The project's `pyproject.toml` lists dependencies. If `uv` is configured to read `pyproject.toml` directly, `uv install` might be sufficient. Given the provided environment context, `uv pip install -r requirements.txt` is a safer bet assuming `requirements.txt` is generated or maintained.)*

## Usage

### Command Line Interface

For quick processing of a single PDF and direct output to the console:
This will process the PDF and model specified in `main.py` and print the extracted data in a Holded-like 
format (Spanish popular invoice manager for companies).


```bash
python main.py <path_to_pdf> <model_name>
uv run main.py <path_to_pdf> <model_name>
```




### Streamlit Web Application

For an interactive experience with PDF upload, data editing, and report generation:
This will open the application in your web browser. You can then:
1.  Upload a PDF invoice.
2.  Select the desired LLM for extraction.
3.  "Extract Data" to process the invoice.
4.  Review and edit the extracted data.
5.  "Create Report" to generate a formatted table of the extracted invoice details.
6.  Manage "Context words" and "Items to find" for custom extraction rules.

## Currently supported LLM Models

The application supports data extraction using the following LLMs:

*   `qwen3:0.6B` (via Ollama) (thinking model, longer inference time)
*   `llama3.2:3B` (via Ollama)
*   `deepseek-r1:7b` (via Ollama) (thinking model, longer inference time)
*   `gpt-4o-mini` (via OpenAI API)
*   `gemini-2.5-flash-lite` (via Google Generative AI API)

Ensure that you have the necessary API keys configured as environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`) if using OpenAI or Google models, or that Ollama is running locally for the Ollama-based models.
