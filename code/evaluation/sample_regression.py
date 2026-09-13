from __future__ import annotations

import csv
import argparse
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buywait.application import Application  # noqa: E402
from buywait.domain import FinancialRequest  # noqa: E402

FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)


@dataclass(frozen=True)
class FieldDiff:
    field: str
    expected: str
    actual: str
    category: str = ""


@dataclass(frozen=True)
class SampleResult:
    request_id: str
    diffs: tuple[FieldDiff, ...] = ()
    error: str | None = None

    @property
    def passed(self) -> bool:
        return not self.diffs and self.error is None


def _decimal(value: str | Decimal | None) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))


def _plain_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format(value, "f")


def _parse_plan(value: str | None) -> tuple[tuple[date, Decimal], ...]:
    value = (value or "").strip()
    if not value or value.lower() == "none":
        return ()
    result = []
    for entry in value.split("|"):
        day, amount = entry.rsplit(":", 1)
        result.append((date.fromisoformat(day), Decimal(amount)))
    return tuple(result)


def _format_plan(payments) -> str:
    if not payments:
        return "none"
    return "|".join(f"{payment.date.isoformat()}:{_plain_decimal(payment.amount)}" for payment in payments)


def _parse_changes(value: str | None) -> tuple[tuple[str, str, Decimal | None], ...]:
    value = (value or "").strip()
    if not value or value.lower() == "none":
        return ()
    result = []
    for action in value.split("|"):
        parts = action.split(":")
        if parts[0] == "stop" and len(parts) == 2:
            result.append((parts[0], parts[1], None))
        elif parts[0] == "reduce_to" and len(parts) == 3:
            result.append((parts[0], parts[1], Decimal(parts[2])))
        else:
            raise ValueError(f"invalid spending change: {action}")
    return tuple(result)


def _format_changes(changes) -> str:
    if not changes:
        return "none"
    values = []
    for change in changes:
        if change.action == "stop":
            values.append(f"stop:{change.event_id}")
        else:
            values.append(f"reduce_to:{change.event_id}:{_plain_decimal(change.new_amount)}")
    return "|".join(values)


def _actual_values(decision) -> dict[str, object]:
    return {
        "amount_safe_to_pay": decision.amount_safe_to_pay,
        "affordability_status": decision.affordability_status.value,
        "recommended_payment_method": decision.recommended_payment_method.value,
        "payment_plan": tuple((p.date, p.amount) for p in decision.payment_plan),
        "earliest_date_for_full_payment": decision.earliest_date_for_full_payment,
        "spending_changes_needed": tuple((c.action, c.event_id, c.new_amount) for c in decision.spending_changes),
    }


def _expected_values(row: dict[str, str]) -> dict[str, object]:
    return {
        "amount_safe_to_pay": _decimal(row["amount_safe_to_pay"]),
        "affordability_status": row["affordability_status"],
        "recommended_payment_method": row["recommended_payment_method"],
        "payment_plan": _parse_plan(row["payment_plan"]),
        "earliest_date_for_full_payment": date.fromisoformat(row["earliest_date_for_full_payment"]) if row["earliest_date_for_full_payment"] else None,
        "spending_changes_needed": _parse_changes(row["spending_changes_needed"]),
    }


def _display(field: str, value: object) -> str:
    if field == "amount_safe_to_pay":
        return _plain_decimal(value)
    if field == "payment_plan":
        return "none" if not value else "|".join(f"{day.isoformat()}:{_plain_decimal(amount)}" for day, amount in value)
    if field == "spending_changes_needed":
        if not value:
            return "none"
        return "|".join(f"{action}:{event_id}" if amount is None else f"{action}:{event_id}:{_plain_decimal(amount)}" for action, event_id, amount in value)
    if value is None:
        return ""
    return str(value)


def _equal(field: str, expected: object, actual: object) -> bool:
    if field == "amount_safe_to_pay":
        return expected == actual
    if field in {"payment_plan", "spending_changes_needed"}:
        return expected == actual
    return expected == actual


def _field_category(field: str, expected: str, actual: str) -> str:
    if field == "amount_safe_to_pay":
        return "SAFE_AMOUNT"
    if field == "earliest_date_for_full_payment":
        return "EARLIEST_DATE"
    if field == "spending_changes_needed":
        return "SPENDING_CHANGE"
    if field in {"recommended_payment_method", "payment_plan"} and "installments" in f"{expected} {actual}":
        return "INSTALLMENT"
    return "PLAN_RANKING"


def evaluate_samples(dataset_dir: str | Path, evidence_extractor=None, progress=None, decision_provider=None) -> tuple[SampleResult, ...]:
    dataset_dir = Path(dataset_dir)
    app = Application(dataset_dir, evidence_extractor=evidence_extractor)
    with (dataset_dir / "sample_requests.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    # The public samples have their own request IDs and are intentionally not
    # in requests.csv. Overlay only these parsed request rows in memory so the
    # production solver remains unchanged and no sample labels enter it.
    for row in rows:
        app.repository.requests[row["request_id"]] = FinancialRequest(
            request_id=row["request_id"], user_id=row["user_id"], request_date=date.fromisoformat(row["request_date"]),
            request_type=row["request_type"], requested_amount=Decimal(row["requested_amount"]),
            desired_completion_date=date.fromisoformat(row["desired_completion_date"]),
            allows_partial_payment=row["allows_partial_payment"].strip().lower() in {"true", "1", "yes", "y"},
            request_text=row["request_text"],
        )
    results = []
    for row in rows:
        request_id = row["request_id"]
        try:
            expected = _expected_values(row)
            decision = decision_provider(app, request_id) if decision_provider else app.decision(request_id)
            decision = app.finalized_decision(request_id, decision)
            actual = _actual_values(decision)
            diffs = tuple(FieldDiff(field, _display(field, expected[field]), _display(field, actual[field]), _field_category(field, _display(field, expected[field]), _display(field, actual[field]))) for field in FIELDS if not _equal(field, expected[field], actual[field]))
            if not decision.decision_explanation.strip():
                diffs += (FieldDiff("decision_explanation", "non-empty", "", "EXPLANATION"),)
            result = SampleResult(request_id, diffs)
            results.append(result)
        except Exception as exc:
            result = SampleResult(request_id, error=f"{type(exc).__name__}: {exc}")
            results.append(result)
        if progress:
            progress(result, len(results), len(rows))
    return tuple(results)


def print_progress(result: SampleResult, index: int, total: int) -> None:
    """Emit a completed row immediately so long local-model runs are inspectable."""
    if result.error:
        print(f"[{index}/{total}] {result.request_id} ERROR {result.error}", flush=True)
        return
    if result.passed:
        print(f"[{index}/{total}] {result.request_id} PASS", flush=True)
        return
    fields = ",".join(diff.field for diff in result.diffs)
    print(f"[{index}/{total}] {result.request_id} FAIL fields={fields}", flush=True)
    for diff in result.diffs:
        print(f"  {diff.field}: expected={diff.expected!r} actual={diff.actual!r} category={diff.category}", flush=True)


def print_report(results: tuple[SampleResult, ...]) -> None:
    total = len(results)
    passed = sum(result.passed for result in results)
    field_totals = {field: 0 for field in FIELDS}
    field_passes = {field: 0 for field in FIELDS}
    categories: dict[str, int] = {}
    for result in results:
        failed_fields = {diff.field for diff in result.diffs}
        if result.error:
            categories["SOLVER_ERROR"] = categories.get("SOLVER_ERROR", 0) + 1
        else:
            for diff in result.diffs:
                categories[diff.category] = categories.get(diff.category, 0) + 1
        for field in FIELDS:
            field_totals[field] += 1
            if field not in failed_fields and not result.error:
                field_passes[field] += 1
    print("Sample regression summary")
    print(f"Requests: {total}")
    print(f"Fully matching: {passed}/{total} ({(passed / total * 100) if total else 0:.1f}%)")
    print("Field accuracy:")
    for field in FIELDS:
        print(f"  {field}: {field_passes[field]}/{field_totals[field]} ({(field_passes[field] / field_totals[field] * 100) if field_totals[field] else 0:.1f}%)")
    print("Failure categories:")
    if categories:
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")
    else:
        print("  none")
    failures = [result for result in results if not result.passed]
    if not failures:
        return
    print("\nDetailed diffs")
    for result in failures:
        request_categories = sorted({diff.category for diff in result.diffs}) if not result.error else ["SOLVER_ERROR"]
        print(f"\n{result.request_id} [{', '.join(request_categories)}]")
        if result.error:
            print(f"  error: {result.error}")
        for diff in result.diffs:
            print(f"  {diff.field} [{diff.category}]")
            print(f"    expected: {diff.expected}")
            print(f"    actual:   {diff.actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public sample regression evaluator")
    parser.add_argument("--no-ai", action="store_true", help="run deterministic evaluation without evidence extraction")
    parser.add_argument("--require-ai", action="store_true", help="fail if the configured AI provider is unavailable")
    parser.add_argument("--trace-path", default=str(Path("evaluation") / "model_turns.jsonl"),
                        help="JSONL file receiving every model turn (default: evaluation/model_turns.jsonl)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    extractor = None
    from agent.config import AgentConfig
    config = AgentConfig.from_dotenv(root / ".env")
    if args.require_ai and (args.no_ai or config.ai_mode == "disabled" or not config.api_key):
        raise SystemExit("--require-ai needs configured AI_MODE and OPENAI_API_KEY in .env")
    if not args.no_ai and config.ai_mode != "disabled" and config.api_key:
        from agent.evidence_extractor import OpenAIEvidenceExtractor
        from agent.trace import ModelTurnRecorder
        recorder = ModelTurnRecorder(root / args.trace_path,
                                     run_id=datetime.now(timezone.utc).isoformat())
        extractor = OpenAIEvidenceExtractor(config, trace_writer=recorder)
    else:
        recorder = None
    try:
        print_report(evaluate_samples(root / "dataset", evidence_extractor=extractor, progress=print_progress))
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
