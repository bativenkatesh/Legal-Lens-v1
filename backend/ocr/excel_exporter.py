import pandas as pd
import os

EXCEL_FILE_PATH = "uploads/gst_invoices.xlsx"

# =========================
# SHEET DEFINITIONS
# =========================

B2B_COLUMNS = [
    "Invoice Date",
    "Invoice Value",
    "Place Of Supply",
    "Reverse Charge",
    "Applicable % of Tax Rate",
    "Invoice Type",
    "E-Commerce GSTIN",
    "Rate",
    "Taxable Value",
    "Cess Amount"
]

HSN_COLUMNS = [
    "HSN",
    "Description",
    "UQC",
    "Total Quantity",
    "Total Value",
    "Rate",
    "Taxable Value",
    "Integrated Tax Amount",
    "Central Tax Amount",
    "State/UT Tax Amount",
    "Cess Amount"
]

DOC_COLUMNS = [
    "Nature of Document",
    "Sr. No. From",
    "Sr. No. To",
    "Total Number",
    "Cancelled"
]


# =========================
# MAIN FUNCTION
# =========================

def append_to_excel(data: dict) -> str:
    """
    Writes GST structured data into 3 sheets:
    B2B, HSN, Documents
    """

    os.makedirs(os.path.dirname(EXCEL_FILE_PATH), exist_ok=True)

    b2b_df = pd.DataFrame(data.get("b2b", []))
    hsn_df = pd.DataFrame(data.get("hsn", []))
    doc_df = pd.DataFrame(data.get("documents", []))

    # enforce schema
    b2b_df = b2b_df.reindex(columns=B2B_COLUMNS)
    hsn_df = hsn_df.reindex(columns=HSN_COLUMNS)
    doc_df = doc_df.reindex(columns=DOC_COLUMNS)

    if not os.path.exists(EXCEL_FILE_PATH):
        with pd.ExcelWriter(EXCEL_FILE_PATH, engine="openpyxl") as writer:
            b2b_df.to_excel(writer, sheet_name="B2B", index=False)
            hsn_df.to_excel(writer, sheet_name="HSN", index=False)
            doc_df.to_excel(writer, sheet_name="Documents", index=False)
    else:
        with pd.ExcelWriter(
            EXCEL_FILE_PATH,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="overlay"
        ) as writer:

            # --- B2B ---
            if "B2B" in writer.book.sheetnames:
                startrow = writer.book["B2B"].max_row
                b2b_df.to_excel(writer, sheet_name="B2B", index=False, header=False, startrow=startrow)
            else:
                b2b_df.to_excel(writer, sheet_name="B2B", index=False)

            # --- HSN ---
            if "HSN" in writer.book.sheetnames:
                startrow = writer.book["HSN"].max_row
                hsn_df.to_excel(writer, sheet_name="HSN", index=False, header=False, startrow=startrow)
            else:
                hsn_df.to_excel(writer, sheet_name="HSN", index=False)

            # --- DOCS ---
            if "Documents" in writer.book.sheetnames:
                startrow = writer.book["Documents"].max_row
                doc_df.to_excel(writer, sheet_name="Documents", index=False, header=False, startrow=startrow)
            else:
                doc_df.to_excel(writer, sheet_name="Documents", index=False)

    return EXCEL_FILE_PATH


# =========================
# RESET
# =========================

def reset_excel() -> bool:
    if os.path.exists(EXCEL_FILE_PATH):
        os.remove(EXCEL_FILE_PATH)
        return True
    return False