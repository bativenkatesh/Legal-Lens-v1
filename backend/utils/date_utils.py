from datetime import datetime

def get_financial_year(date_str: str | None) -> str:
    """
    Indian financial year:
    Apr 1 – Mar 31
    """
    if not date_str:
        return "unknown"

    dt = datetime.fromisoformat(date_str)
    year = dt.year

    if dt.month >= 4:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"
