import argparse
from facturas.model import run_models
from facturas.utils import get_holded_format


def main():
    parser = argparse.ArgumentParser(description="Process PDF invoices using various LLMs.")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF invoice file.')
    parser.add_argument('model', type=str, help='Name of the LLM model to use (e.g., llama3.2:3B).')

    args = parser.parse_args()

    invoice = run_models(args.pdf_path, args.model)
    df = get_holded_format(invoice.main_table)
    print(df)

if __name__ == "__main__":
    main()
