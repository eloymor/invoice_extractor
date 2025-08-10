import os
import logging
import re
from typing import Optional
from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel
import pandas as pd
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from facturas.config import item_table_words, total_table_words, key_words_path, items_to_find_path
from facturas.utils import (context_words, convert_to_json, create_main_items_df, load_key_words, load_items_to_find,
                            get_holded_format)

load_dotenv(override=True)


# _log = logging.getLogger(__name__)


class InvoiceData(BaseModel):
    """Fields to extract from the invoice data -> Structured output (JSON format)."""

    invoice_number: Optional[str] = Field(..., description="Invoice number")
    invoice_date: Optional[str] = Field(..., description="Invoice date")
    issuer_name: Optional[float] = Field(..., description="Name of the issuer")
    issuer_NIF: Optional[str] = Field(..., description="Issuer NIF or issuer document code")
    issuer_address: Optional[str] = Field(..., description="Address or street of the issuer")
    issuer_city: Optional[str] = Field(..., description="City of the issuer")
    issuer_postal_code: Optional[int] = Field(..., description="Postal code of the issuer")
    invoice_total: Optional[float] = Field(..., description="Total amount of the invoice, including possible taxes")
    invoice_subtotal: Optional[float] = Field(..., description="Total amount of the invoice, excluding possible taxes")
    invoice_iva_21: Optional[float] = Field(..., description="Amount of IVA 21%, 21% tax value")
    invoice_iva_10: Optional[float] = Field(..., description="Amount of IVA 10%, 10% tax value")
    invoice_iva_4: Optional[float] = Field(..., description="Amount of IVA 4%, 4% tax value")


class Invoice:
    def __init__(self, input_doc_path: str):
        self.doc_converter = DocumentConverter()
        self.input_doc_path = input_doc_path
        self.document = None
        self.doc_text = None
        self.items_table = None
        self.total_table = None
        self.main_table = None
        self.thinking_content = None

    def print_document(self):
        """
        Convert a .pdf file into a PIL image
        :return: list of PIL images
        """
        return convert_from_path(self.input_doc_path)

    def parse_document(self):
        """
        Extract tables and text from a .pdf file
        Updates self.document and self.item_table
        :return: None
        """

        logging.basicConfig(level=logging.INFO)

        conv_res = self.doc_converter.convert(source=self.input_doc_path)

        for table_idx, table in enumerate(conv_res.document.tables):
            table_df: pd.DataFrame = table.export_to_dataframe()
            col_list: list[str] = [col.lower() for col in table_df.columns]
            if 'Unnamed: 0' in col_list:
                col_list.remove('Unnamed: 0')
                table_df = table_df.drop(columns=['Unnamed: 0'])
            for count, word in enumerate(item_table_words):
                if word in col_list:
                    table_df = table_df.loc[:, ~table_df.columns.duplicated()]
                    self.items_table = table_df
                    break
                elif (count == len(item_table_words) - 1) and (word not in col_list) and (
                        len(conv_res.document.tables) == 2):
                    self.total_table = table_df
            for word in total_table_words:
                if word in col_list:
                    self.total_table = table_df
                    break

        self.document = conv_res.document.export_to_markdown()
        table_nodes = [
            node for node, _level in conv_res.document.iterate_items()
            if node.label == DocItemLabel.TABLE
        ]
        if table_nodes:
            conv_res.document.delete_items(node_items=table_nodes)
        if isinstance(self.total_table, pd.DataFrame):
            conv_res.document.add_text(label=DocItemLabel.PARAGRAPH, text=self.total_table.to_json(orient='records'))
        doc_text = conv_res.document.export_to_markdown().lower()
        # Translation is managed directly in the LLM via context
        # key_words: dict[str, str] = load_key_words(key_words_path)
        # for key, value in key_words.items():
        #     self.doc_text = doc_text.replace(key, value)
        self.doc_text = doc_text  # comment line if manual translation is applied
        pass


def decode_invoice_qwen3(text: str) -> tuple[str, str]:
    """
    Extract invoice data using the Qwen3 model. Transformers API.

    Args:
        text: The text to extract data from

    Returns:
        A tuple containing (content, thinking_content) where:
        - content: The extracted data in JSON format
        - thinking_content: The model's thinking process
    """
    model_id = "Qwen/Qwen3-0.6B"
    # local_model_path = "/root/.cache/huggingface/hub/Qwen--Qwen3-0.6B/" # only for docker deployment
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    items: str = context_words(load_key_words(key_words_path))
    items_to_find: list[str] = load_items_to_find(items_to_find_path)

    messages = [
        {"role": "system", "content": """You are an invoice data extractor.
                                        If an item is not found, just return an empty JSON object.
                                        Answer must be in JSON format."""},
        {"role": "user", "content": f"Please try to extract the following items (some of them may not be present): " +
                                    f"{items_to_find} \n\n" +
                                    f"Context information for these words: {items} \n\n" +
                                    f"This is the invoice data: {text} \n\n"
         },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    filtered_content = re.search(r'(\{.*\})', content, re.DOTALL).group(1)

    return filtered_content, thinking_content


def decode_invoice_langchain(model: str = "qwen3:0.6B", *, text: str) -> str:
    OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

    if model == "qwen3:0.6B":
        llm = OllamaLLM(model="qwen3:0.6b",
                        base_url=OLLAMA_API_URL,
                        temperature=0)
    elif model == "llama3.2:3B":
        llm = OllamaLLM(model="llama3.2:latest",
                        base_url=OLLAMA_API_URL,
                        temperature=0)
    elif model == "deepseek-r1:7b":
        llm = OllamaLLM(model="deepseek-r1:7b",
                        base_url=OLLAMA_API_URL,
                        temperature=0)
    elif model == "gpt-4o-mini":
        llm = ChatOpenAI(model="gpt-4o-mini",
                         temperature=0,
                         api_key=os.environ.get("OPENAI_API_KEY"))
    elif model == "gemini-2.5-flash-lite":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
                                     temperature=0,
                                     api_key=os.environ.get("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Invalid model name\nSelected model: {model}")

    parser = PydanticOutputParser(pydantic_object=InvoiceData)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an invoice data extractor. If an item is not found, just return an empty JSON object. "
                "Answer must be in JSON format.",
            ),
            (
                "user",
                "Please extract the following items: {items_to_find}\n\n"
                "Context info: {items}\n\n"
                "This is the invoice data: {invoice_text}\n"
                "Only return the requested items\n\n"
                "{format_instructions}"
            ),
        ]
    )

    items: str = context_words(load_key_words(key_words_path))
    items_to_find: list[str] = load_items_to_find(items_to_find_path)

    prompt_text = prompt.format(
        items_to_find=items_to_find,
        items=items,
        invoice_text=text,
        format_instructions=parser.get_format_instructions()
    )

    if model in ["gpt-4o-mini", "gemini-2.5-flash-lite"]:
        response_text = llm.invoke(prompt_text).content
    else:
        response_text = llm.invoke(prompt_text)
    response = re.search(r'(\{.*\})', response_text, re.DOTALL)
    if not response:
        response = ""
    else:
        response = response.group(1)

    return response


def run_models(pdf_path: str, model: str = "qwen3:0.6B") -> Invoice:
    """
    Run the invoice processing models on a PDF file.

    Args:
        pdf_path: Path to the PDF file
        model: LLM model name

    Returns:
        The processed Invoice object
    """
    invoice = Invoice(pdf_path)
    invoice.parse_document()
    output_text = decode_invoice_langchain(model, text=invoice.doc_text)

    # Transformers
    # output_text, thinking_content = decode_invoice_qwen3(invoice.doc_text)

    invoice.main_table = create_main_items_df(convert_to_json(output_text))

    return invoice
