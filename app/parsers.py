# app/parsers.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_table_of_benefits(full_text: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Uses an LLM to find and extract the 'Table of Benefits' from the full document text
    and format it as a clean markdown table.
    """
    print("Extracting Table of Benefits...")
    
    # This prompt is highly specific to the task of table extraction
    prompt_template = """
    From the following full text of an insurance policy document, find the section titled "Table of Benefits".
    Extract all the information from this table, including all plans (Plan A, Plan B, Plan C), features, and their corresponding limits or values.
    Format the extracted information as a clean, readable Markdown table.
    If the table is not found, respond with "No Table of Benefits found in the provided text."

    FULL DOCUMENT TEXT:
    {text}

    MARKDOWN TABLE:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    table_extraction_chain = prompt | llm | StrOutputParser()
    
    try:
        markdown_table = table_extraction_chain.invoke({"text": full_text})
        print("✅ Table of Benefits extracted successfully.")
        return markdown_table
    except Exception as e:
        print(f"❌ Error extracting table: {e}")
        return "Error extracting table data."
