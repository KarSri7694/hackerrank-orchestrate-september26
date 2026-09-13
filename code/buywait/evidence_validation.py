from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def _numeric_candidates(token: str) -> set[Decimal]:
    """Parse common thousands/decimal separators without locale assumptions."""
    value = token.strip().rstrip(".,")
    if not value:
        return set()
    candidates = set()
    forms = {value}
    if "," in value and "." in value:
        # Whichever separator occurs last is conventionally the decimal mark.
        decimal_mark = "," if value.rfind(",") > value.rfind(".") else "."
        thousands_mark = "." if decimal_mark == "," else ","
        forms.add(value.replace(thousands_mark, "").replace(decimal_mark, "."))
    elif "," in value:
        forms.add(value.replace(",", ""))
        forms.add(value.replace(",", "."))
    elif "." in value:
        forms.add(value.replace(".", ""))
    for form in forms:
        try:
            number = Decimal(form)
        except (InvalidOperation, ValueError):
            continue
        if number.is_finite():
            candidates.add(number)
    return candidates
from datetime import date


def text_supports_amount(text: str | None, amount, currency: str | None) -> bool:
    """Return whether a cached amount is explicitly present in source text."""
    if amount is None:
        return True
    source = text or ""
    expected = Decimal(str(amount))
    tokens = re.findall(
        r"(?<![A-Za-z0-9])\d[\d,.]*(?![A-Za-z0-9])", source
    )
    if not any(expected in _numeric_candidates(token) for token in tokens):
        return False
    if not currency:
        return True
    currency = str(currency).casefold()
    symbols = {"usd": "$", "eur": "\u20ac", "gbp": "\u00a3", "inr": "\u20b9", "idr": "rp", "zar": "r"}
    lowered = source.casefold()
    return currency in lowered or symbols.get(currency, "\0") in source


def text_supports_date(text: str | None, value: date | None) -> bool:
    """Return whether a date is explicitly represented in source text.

    This intentionally checks numeric date forms rather than English words.
    The supplied datasets use ISO and numeric day/month/year forms across
    languages, and a model's relative interpretation (for example, "next
    payroll") must not become an invented calendar date.
    """
    if value is None:
        return True
    source = text or ""
    year, month, day = value.year, value.month, value.day
    numbers = [str(day), f"{day:02d}", str(month), f"{month:02d}", str(year)]
    forms = (
        rf"{year}[-/.]{numbers[3]}[-/.]{numbers[1]}",
        rf"{year}[-/.]{numbers[2]}[-/.]{numbers[0]}",
        rf"{numbers[1]}[-/.]{numbers[3]}[-/.]{year}",
        rf"{numbers[0]}[-/.]{numbers[2]}[-/.]{year}",
        rf"{numbers[3]}[-/.]{numbers[1]}[-/.]{year}",
        rf"{numbers[2]}[-/.]{numbers[0]}[-/.]{year}",
        rf"{year}\s*年\s*(?:{numbers[2]}|{numbers[3]})\s*月\s*(?:{numbers[0]}|{numbers[1]})\s*日?",
    )
    return any(re.search(rf"(?<!\d){form}(?!\d)", source) for form in forms)
