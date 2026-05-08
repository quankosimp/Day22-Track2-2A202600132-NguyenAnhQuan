"""Step 4: Custom Guardrails validators for PII and JSON formatting."""

from __future__ import annotations

import json
import re

from guardrails import Guard, OnFailAction
from guardrails.validators import FailResult, PassResult, Validator, register_validator


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detect and redact common PII with regular expressions."""

    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_types = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_types.append(pii_type)

        if found_types:
            return FailResult(
                error_message=f"Detected PII: {', '.join(sorted(set(found_types)))}",
                fix_value=redacted_text,
            )
        return PassResult(value_override=value)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validate JSON strings and repair common LLM formatting issues."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    @staticmethod
    def _format(text: str) -> str:
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)

    def validate(self, value: str, metadata: dict):
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        try:
            repaired = self._repair(value)
            formatted = self._format(repaired)
            return FailResult(
                error_message="JSON repaired successfully",
                fix_value=formatted,
            )
        except json.JSONDecodeError as exc:
            fallback = json.dumps(
                {
                    "error": f"Invalid JSON after repair attempt: {exc.msg}",
                    "raw": value,
                },
                ensure_ascii=False,
            )
            return FailResult(
                error_message="Invalid JSON after repair attempt",
                fix_value=fallback,
            )


def demo_pii_guard() -> None:
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Passed: {result.validation_passed}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


def demo_json_guard() -> None:
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Passed: {result.validation_passed}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


def main() -> None:
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)
    demo_pii_guard()
    demo_json_guard()
    print("\nStep 4 complete")


if __name__ == "__main__":
    main()
