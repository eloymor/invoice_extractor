import streamlit as st
import pandas as pd
from facturas.model import run_models, Invoice
from facturas.config import key_words_path, items_to_find_path
from facturas.utils import load_key_words, save_key_words, load_items_to_find, save_items_to_find, get_holded_format
import tempfile
import os

st.set_page_config(layout="wide", page_title="Invoice Data Extractor")

st.title("Invoice Data Extractor")


# Context words
context_words: dict[str, str] = load_key_words(key_words_path)

if "context_words" not in st.session_state:
    st.session_state.context_words = context_words

if "show_form_context" not in st.session_state:
    st.session_state.show_form_context = False

if "row_deleted" not in st.session_state:
    st.session_state.row_deleted = None


# Items to find
items_to_find: list[str] = load_items_to_find(items_to_find_path)

if "items_to_find" not in st.session_state:
    st.session_state.items_to_find = items_to_find

if "show_form_items" not in st.session_state:
    st.session_state.show_form_items = False

if "row_deleted_items" not in st.session_state:
    st.session_state.row_deleted_items = None
    
# Report table
if "show_report" not in st.session_state:
    st.session_state.show_report = False
    
if "report_table" not in st.session_state:
    st.session_state.report_table = None


# Selected model
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Qwen3:0.6B"

button_1, button_2, button_3 = st.columns(3)
with button_1:
    if st.button("Edit context words"):
        st.session_state.show_form_context = not st.session_state.show_form_context
with button_2:
    if st.button("Items to find"):
        st.session_state.show_form_items = not st.session_state.show_form_items
with button_3:
    selected_model = st.selectbox(
        "Select model",
        options=["qwen3:0.6B", "llama3.2:3B", "deepseek-r1:7b", "gpt-4o-mini", "gemini-2.5-flash-lite"]
    )
    st.session_state.selected_model = selected_model

# Context words
if st.session_state.show_form_context:
    with st.form("context_form"):
        st.markdown("### 📝 Edit context words")
        col1, col2, col3 = st.columns([1, 1, 0.5])
        col1.markdown("**Original**")
        col2.markdown("**Translated**")
        col3.markdown("**Delete**")

        new_items = []
        keys_seen = set()
        for i, (key, value) in enumerate(st.session_state.context_words.items()):
            c1, c2, c3 = st.columns([1, 1, 0.5])
            new_key = c1.text_input(f"Key {i+1}", value=key, label_visibility="collapsed", key=f"key_{i}")
            new_value = c2.text_input(f"Value {i+1}", value=value, label_visibility="collapsed", key=f"value_{i}")
            delete = c3.checkbox("🗑️", key=f"delete_{i}")
            if delete:
                continue
            if new_key.strip() != "" or new_value.strip() != "":
                new_items.append((new_key, new_value))
                keys_seen.add(new_key)

        c1, c2, _ = st.columns([1, 1, 0.5])
        extra_key = c1.text_input("New key", "", label_visibility="collapsed", key="extra_key")
        extra_val = c2.text_input("New value", "", label_visibility="collapsed", key="extra_val")
        if extra_key.strip() != "" and extra_val.strip() != "" and extra_key not in keys_seen:
            new_items.append((extra_key.strip(), extra_val))

        submitted = st.form_submit_button("Submit")

        if submitted:
            updated_context_words = {key: value for key, value in new_items if key.strip() != ""}
            st.session_state.context_words = updated_context_words
            st.success(f"✅ Context words updated successfully!")
            save_key_words(key_words_path, updated_context_words)

# Items to find
if st.session_state.show_form_items:
    with st.form("items_form"):
        st.markdown("### 📝 Edit items to find")
        col1, col2 = st.columns([1, 0.5])
        col1.markdown("**Item to find**")
        col2.markdown("**Delete**")

        new_items = []
        keys_seen = set()
        for i, value in enumerate(st.session_state.items_to_find):
            c1, c2, = st.columns([1, 0.5])
            new_value = c1.text_input(f"Value {i + 1}", value=value, label_visibility="collapsed", key=f"value_{i}")
            delete = c2.checkbox("🗑️", key=f"delete_{i}")
            if delete:
                continue
            if new_value.strip() != "":
                new_items.append(new_value)
                keys_seen.add(new_value)

        c1, _ = st.columns([1, 0.5])
        extra_val = c1.text_input("New value", "", label_visibility="collapsed", key="extra_val")
        if extra_val.strip() != "" and extra_key not in keys_seen:
            new_items.append((extra_key.strip(), extra_val))

        submitted = st.form_submit_button("Submit")

        if submitted:
            updated_items = [value for value in new_items if value.strip() != ""]
            st.session_state.items_to_find = items_to_find
            st.success(f"✅ Items to find updated successfully!")
            save_items_to_find(items_to_find_path, updated_items)

# Create two columns
left_col, right_col = st.columns(2)

# Global variable to store the invoice object
if 'invoice' not in st.session_state:
    st.session_state.invoice = None

with left_col:
    st.header("Upload Invoice PDF")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Preview the PDF
        st.subheader("PDF Preview")

        # Create an Invoice object to use its print_document method
        invoice_preview = Invoice(pdf_path)
        pdf_images = invoice_preview.print_document()

        # Display the first page of the PDF
        if pdf_images:
            st.image(pdf_images[0], caption="Page 1", use_container_width=True)

            # Add a button to process the PDF
            if st.button("Extract Data"):
                with st.spinner("Processing PDF..."):
                    # Process the invoice using run_models function
                    from facturas.model import run_models

                    # run_models handles creating the Invoice object, parsing the document,
                    # extracting data, and storing the thinking_content
                    invoice_obj = run_models(pdf_path, st.session_state.selected_model)

                    # Store the invoice object in session state
                    st.session_state.invoice = invoice_obj
                st.success("Data extracted successfully!")

with right_col:
    st.header("Extracted Data")

    if st.session_state.invoice is not None and hasattr(st.session_state.invoice, 'main_table'):
        # Get the main table from the invoice object
        main_table = st.session_state.invoice.main_table

        if main_table is not None and not main_table.empty:
            # Create a form for editing the extracted data
            with st.form("edit_form"):
                st.subheader("Edit Extracted Data")

                # Convert DataFrame to dictionary for easier editing
                data_dict = main_table.to_dict('records')[0] if len(main_table) > 0 else {}

                # Create input fields for each item in the data
                edited_data = {}
                for key, value in data_dict.items():
                    edited_data[key] = st.text_input(key, value)

                # Submit button
                submit_button = st.form_submit_button("Save Changes")

                if submit_button:
                    # Update the main_table with the edited data
                    st.session_state.invoice.main_table = pd.DataFrame([edited_data])
                    st.success("Changes saved successfully!")

                    # Display the updated data
                    st.subheader("Updated Data")
                    st.dataframe(st.session_state.invoice.main_table)

            # Add a form for editing items_table
            if hasattr(st.session_state.invoice, 'items_table') and st.session_state.invoice.items_table is not None and not st.session_state.invoice.items_table.empty:
                with st.form("items_table_form"):
                    st.subheader("Edit Items Table")

                    # Get the items table
                    items_table = st.session_state.invoice.items_table

                    # Create an editable dataframe
                    edited_items_data = st.data_editor(items_table, use_container_width=True)

                    # Submit button
                    items_submit_button = st.form_submit_button("Save Items Table Changes")

                    if items_submit_button:
                        # Update the items_table with the edited data
                        st.session_state.invoice.items_table = edited_items_data
                        st.success("Items table changes saved successfully!")
            
            # Add "Create Report" button that calculates and shows the report table
            if hasattr(st.session_state.invoice, 'main_table') and st.session_state.invoice.main_table is not None:
                if st.button("Create Report"):
                    # Calculate the table using get_holded_format when button is pressed
                    new_report_data = get_holded_format(st.session_state.invoice.main_table)
                    
                    # Concatenate with existing data if it exists
                    if st.session_state.report_table is not None:
                        st.session_state.report_table = pd.concat([st.session_state.report_table, new_report_data], ignore_index=True)
                    else:
                        st.session_state.report_table = new_report_data
                        
                    st.session_state.show_report = True
        else:
            st.info("No data extracted. Please try another PDF or check if the extraction process completed successfully.")
    else:
        st.info("Please upload a PDF and extract data first.")

# Display thinking content below the two columns
# if st.session_state.invoice is not None and hasattr(st.session_state.invoice, 'thinking_content') and st.session_state.invoice.thinking_content:
#     st.header("Model Thinking Process")
#     st.text_area("AI Reasoning", st.session_state.invoice.thinking_content, height=300)

# Display the calculated report table when show_report is True
if st.session_state.show_report and st.session_state.invoice is not None and "report_table" in st.session_state:
    st.header("Invoice Report")
    st.dataframe(st.session_state.report_table, use_container_width=True)
    
    # Add a button to hide the report
    if st.button("Hide Report"):
        st.session_state.show_report = False

# Clean up temporary files
if 'pdf_path' in locals():
    try:
        os.unlink(pdf_path)
    except:
        pass
