"""Condition checker utility for recording and validating conditions

This module provides a reusable condition checker that automatically extracts
condition expressions from code and records their evaluation results.
"""

import inspect
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ConditionChecker:
    """Condition checker that automatically records each condition check result"""

    def __init__(self, name: str) -> None:
        """Initialize the condition checker

        Args:
            name: Name of the component being checked (e.g., strategy name, class name)
        """
        self.name: str = name
        self.conditions: List[Tuple[int, str, bool, Dict[str, Any]]] = []
        self.counter: int = 0

    def check(self, result: bool, **details: Any) -> bool:
        """Check and record a condition

        Automatically extracts the condition expression from the calling code.

        Args:
            result: Condition evaluation result
            **details: Optional context information (e.g., quant_method="FP8")

        Returns:
            The condition result (passthrough)
        """
        condition_str = self._extract_condition_string()

        self.counter += 1
        self.conditions.append((self.counter, condition_str, result, details))
        return result

    @staticmethod
    def _parse_check_argument(text: str) -> str:
        """Parse the argument from checker.check() call using bracket matching.

        Args:
            text: Text starting after ".check("

        Returns:
            The extracted condition expression, or empty string if not found
        """
        match = re.search(r"\.check\s*\(", text)
        if not match:
            return ""

        start_pos = match.end()
        paren_count = 0
        bracket_count = 0

        for i in range(start_pos, len(text)):
            char = text[i]
            if char == "(":
                paren_count += 1
            elif char == ")":
                if paren_count == 0:
                    # Found closing parenthesis of .check()
                    result = text[start_pos:i].strip()
                    return re.sub(r"\s+", " ", result)  # Normalize whitespace
                paren_count -= 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            elif char == "," and paren_count == 0 and bracket_count == 0:
                # Found comma at same level as .check()
                result = text[start_pos:i].strip()
                return re.sub(r"\s+", " ", result)
        return ""

    @staticmethod
    def _read_source_lines(filename: str, lineno: int, max_lines: int = 10) -> str:
        """Read multiple lines from source file for multi-line call support.

        Args:
            filename: Source file path
            lineno: Starting line number
            max_lines: Maximum number of lines to read

        Returns:
            Concatenated lines with single spaces, or empty string if failed
        """
        try:
            import linecache

            lines: List[str] = []
            for i in range(max_lines):
                line_content = linecache.getline(filename, lineno + i)
                if not line_content:
                    break
                lines.append(line_content.rstrip())
            return " ".join(lines)
        except Exception:
            return ""

    def _extract_condition_string(self) -> str:
        """Extract the condition expression string from the calling code

        Returns:
            The condition expression as a string, or "unknown" if extraction fails
        """
        try:
            frame = inspect.currentframe()
            if not frame or not frame.f_back or not frame.f_back.f_back:
                return "unknown"

            caller_frame = frame.f_back.f_back
            info = inspect.getframeinfo(caller_frame)

            # Try multi-line reading from source file first
            if info.filename and info.lineno:
                full_text = self._read_source_lines(info.filename, info.lineno)
                if full_text:
                    result = self._parse_check_argument(full_text)
                    if result:
                        return result

            # Fallback to single line from code_context
            if info.code_context:
                line = info.code_context[0].strip()
                result = self._parse_check_argument(line)
                if result:
                    return result

        except Exception:
            pass
        return "unknown"

    def all_passed(self) -> bool:
        """Check if all conditions passed

        Returns:
            True if all conditions passed, False otherwise
        """
        result = all(cond[2] for cond in self.conditions)

        lines = [f"[ConditionChecker] Checking {self.name}:"]
        for counter, condition_str, cond_result, details in self.conditions:
            symbol = "✓" if cond_result else "✗"
            msg = f"  {symbol} condition_{counter}: {condition_str} = {cond_result}"
            if details:
                detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
                msg += f" ({detail_str})"
            lines.append(msg)
        lines.append(f"  → Final result: {result}")

        logger.debug("\n".join(lines))
        return result
