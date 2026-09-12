from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path


class CsvExchangeRates:
    def __init__(self, path: str | Path):
        self.rates: dict[tuple[date, str, str], Decimal] = {}
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                self.rates[(date.fromisoformat(row["rate_date"]), row["from_currency"], row["to_currency"])] = Decimal(row["rate"])

    def convert(self, amount: Decimal, source: str, target: str, on_date: date) -> Decimal:
        if source == target:
            return amount
        direct = self.rates.get((on_date, source, target))
        if direct is not None:
            return amount * direct
        inverse = self.rates.get((on_date, target, source))
        if inverse is not None:
            return amount / inverse
        # Forecasted recurring events can fall after the final dated rate. Use
        # the latest supplied rate for that pair, keeping the run deterministic
        # while never consulting a live market source.
        earlier = [(d, rate) for (d, src, dst), rate in self.rates.items() if src == source and dst == target and d <= on_date]
        if earlier:
            return amount * max(earlier, key=lambda x: x[0])[1]
        inverse_earlier = [(d, rate) for (d, src, dst), rate in self.rates.items() if src == target and dst == source and d <= on_date]
        if inverse_earlier:
            return amount / max(inverse_earlier, key=lambda x: x[0])[1]
        raise ValueError(f"missing exchange rate for {on_date}: {source}->{target}")
