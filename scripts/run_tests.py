"""Colored unittest runner for the Grid-world RL project."""

from __future__ import annotations

import argparse
import os
import sys
import time
import unittest
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep test discovery and package imports anchored at the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


@dataclass(frozen=True)
class Colors:
    """ANSI color escape sequences used by the test runner."""

    green: str = "\033[32m"
    red: str = "\033[31m"
    yellow: str = "\033[33m"
    blue: str = "\033[34m"
    cyan: str = "\033[36m"
    bold: str = "\033[1m"
    dim: str = "\033[2m"
    reset: str = "\033[0m"


COLORS = Colors()


def colorize(text: str, color: str, enabled: bool) -> str:
    """Wrap text in ANSI color codes when color output is enabled."""
    if not enabled:
        return text
    return f"{color}{text}{COLORS.reset}"


class ColoredTextResult(unittest.TextTestResult):
    """Unittest result that prints compact colored per-test statuses."""

    def __init__(self, *args: Any, color: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.color = color
        self.warning_records: list[warnings.WarningMessage] = []

    def getDescription(self, test: unittest.case.TestCase) -> str:
        """Return a readable dotted test name."""
        return test.id()

    def startTest(self, test: unittest.case.TestCase) -> None:
        """Start a test without printing unittest's default prefix."""
        unittest.TestResult.startTest(self, test)

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        unittest.TestResult.addSuccess(self, test)
        self._write_status("PASS", test, COLORS.green)

    def addFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        unittest.TestResult.addFailure(self, test, err)
        self._write_status("FAIL", test, COLORS.red)

    def addError(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        unittest.TestResult.addError(self, test, err)
        self._write_status("ERROR", test, COLORS.red)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        unittest.TestResult.addSkip(self, test, reason)
        self._write_status("SKIP", test, COLORS.yellow, reason)

    def addExpectedFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        unittest.TestResult.addExpectedFailure(self, test, err)
        self._write_status("XFAIL", test, COLORS.yellow)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:
        unittest.TestResult.addUnexpectedSuccess(self, test)
        self._write_status("XPASS", test, COLORS.red)

    def printErrors(self) -> None:
        """Print warnings before the standard error/failure details."""
        if self.warning_records:
            self.stream.writeln()
            self.stream.writeln(colorize("Warnings", COLORS.yellow + COLORS.bold, self.color))
            for warning in self.warning_records:
                location = f"{warning.filename}:{warning.lineno}"
                category = warning.category.__name__
                message = f"{location}: {category}: {warning.message}"
                self.stream.writeln(colorize(f"  {message}", COLORS.yellow, self.color))
        super().printErrors()

    def _write_status(
        self,
        label: str,
        test: unittest.case.TestCase,
        color: str,
        extra: str | None = None,
    ) -> None:
        status = colorize(f"[{label:<5}]", color + COLORS.bold, self.color)
        name = colorize(self.getDescription(test), COLORS.cyan, self.color)
        line = f"{status} {name}"
        if extra:
            line += colorize(f"  {extra}", COLORS.dim, self.color)
        self.stream.writeln(line)


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Test runner that uses ColoredTextResult."""

    resultclass = ColoredTextResult

    def __init__(self, *args: Any, color: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.color = color

    def _makeResult(self) -> ColoredTextResult:
        return self.resultclass(
            self.stream,
            self.descriptions,
            self.verbosity,
            color=self.color,
        )

    def run(self, test: unittest.suite.TestSuite) -> ColoredTextResult:
        """Run tests while collecting warnings for yellow reporting."""
        start_time = time.perf_counter()
        result = self._makeResult()
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals

        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("default")
            start_test_run = getattr(result, "startTestRun", None)
            if start_test_run is not None:
                start_test_run()
            try:
                test(result)
            finally:
                stop_test_run = getattr(result, "stopTestRun", None)
                if stop_test_run is not None:
                    stop_test_run()
            result.warning_records.extend(warning_records)

        elapsed = time.perf_counter() - start_time
        result.printErrors()
        self.stream.writeln()
        self.stream.writeln(f"Ran {result.testsRun} tests in {elapsed:.3f}s")
        self.stream.writeln()
        if result.wasSuccessful():
            summary = f"OK - {result.testsRun} tests passed"
            self.stream.writeln(colorize(summary, COLORS.green + COLORS.bold, self.color))
        else:
            failed = len(result.failures)
            errors = len(result.errors)
            skipped = len(result.skipped)
            summary = (
                f"FAILED - tests={result.testsRun}, failures={failed}, "
                f"errors={errors}, skipped={skipped}"
            )
            self.stream.writeln(colorize(summary, COLORS.red + COLORS.bold, self.color))
        return result


def discover_tests(start_dir: str, pattern: str) -> unittest.suite.TestSuite:
    """Discover tests using unittest's standard loader."""
    return unittest.defaultTestLoader.discover(start_dir=start_dir, pattern=pattern)


def main() -> int:
    """Run the project test suite with colored output."""
    parser = argparse.ArgumentParser(description="Run Grid-world RL tests.")
    parser.add_argument("-s", "--start-dir", default="tests", help="Test directory.")
    parser.add_argument("-p", "--pattern", default="test*.py", help="Test file pattern.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    args = parser.parse_args()

    suite = discover_tests(args.start_dir, args.pattern)
    runner = ColoredTextTestRunner(verbosity=2, color=not args.no_color)
    result = runner.run(suite)

    if result.wasSuccessful():
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
