import pandas as pd
import os

EXCEL_FILE_PATH = "uploads/gst_invoices.xlsx"

# The exact columns the user requested
COLUMNS = [
    "Type",
    "Place of Supply(POS)",
    "Applicable % of Tax Rate",
    "Rate",
    "Taxable Value",
    "Total Amount",
    "Cess Amount",
    "GSTIN"
]

def append_to_excel(record: dict) -> str:
    """
    Appends a new GST record row to the central Excel file.
    Creates the file with headers if it doesn't exist.
    """
    # Map the dict from Pydantic schema to the Excel columns
    new_row = {
        "Type": record.get("type", "OE"),
        "Place of Supply(POS)": record.get("place_of_supply", ""),
        "Applicable % of Tax Rate": record.get("applicable_tax_rate_percent", ""),
        "Rate": record.get("rate", ""),
        "Taxable Value": record.get("taxable_value", 0.0),
        "Total Amount": record.get("total_amount", 0.0),
        "Cess Amount": record.get("cess_amount", 0.0),
        "GSTIN": record.get("gstin", "")
    }

    df_new = pd.DataFrame([new_row])

    if os.path.exists(EXCEL_FILE_PATH):
        # File exists, append without headers
        try:
            with pd.ExcelWriter(EXCEL_FILE_PATH, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                # Find the current max row to append to
                startrow = writer.sheets['Sheet1'].max_row
                df_new.to_excel(writer, index=False, header=False, startrow=startrow)
        except Exception as e:
            # Fallback if the file is corrupted or sheet name issues
            df_existing = pd.read_excel(EXCEL_FILE_PATH)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(EXCEL_FILE_PATH, index=False)
    else:
        # File doesn't exist, create it with headers
        os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)
        df_new.to_excel(EXCEL_FILE_PATH, index=False)
    
    return EXCEL_FILE_PATH

def reset_excel() -> bool:
    """
    Deletes the Excel file if it exists.
    """
    if os.path.exists(EXCEL_FILE_PATH):
        os.remove(EXCEL_FILE_PATH)
        return True
    return False
