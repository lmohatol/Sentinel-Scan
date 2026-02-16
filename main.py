"""
Sentinel-Scan: Production-Grade AI Code Auditing CLI Tool

This module provides a robust CLI tool for automated source code analysis using
reasoning-heavy LLM APIs. It leverages asyncio for non-blocking operations and
includes comprehensive error handling with exponential backoff retry logic.

Author: MiniMax Agent
Version: 1.0.0
"""

import asyncio
import argparse
import os
import json
import time
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.table import Table
from rich import box
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)

# Load environment variables from .env file
load_dotenv()

# Configuration constants
API_KEY = os.getenv("LLM_API_KEY")
API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 3

# Initialize Rich console for beautiful output
console = Console()


class SentinelScanError(Exception):
    """Base exception for Sentinel-Scan errors."""
    pass


class APIError(SentinelScanError):
    """Exception raised for API-related errors."""
    pass


class NetworkTimeoutError(SentinelScanError):
    """Exception raised for network timeout errors."""
    pass


class ValidationError(SentinelScanError):
    """Exception raised for input validation errors."""
    pass


class LLMResponseError(SentinelScanError):
    """Exception raised when LLM fails to return valid response."""
    pass


class SentinelScanner:
    """
    Core scanner class that handles async LLM integration and analysis.

    This class manages the entire scanning workflow including:
    - Async file ingestion
    - LLM API communication with retry logic
    - Human-in-the-loop verification
    - Markdown report generation
    """

    def __init__(self, model: str, verify_mode: bool = False):
        """
        Initialize the SentinelScanner.

        Args:
            model: LLM model identifier to use for analysis
            verify_mode: Enable human-in-the-loop verification for high-risk findings
        """
        self.model = model
        self.verify_mode = verify_mode
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.console = Console()

    def _build_system_prompt(self) -> str:
        """
        Construct the system prompt that defines the AI's analysis persona and rules.

        Returns:
            Formatted system prompt string
        """
        return """You are Sentinel, a senior security auditor and code quality expert.
Your role is to analyze Python source code and identify:

1. SECURITY FLAWS:
   - SQL injection vulnerabilities
   - Cross-site scripting (XSS) risks
   - Race conditions and concurrency issues
   - Hardcoded credentials or secrets
   - Path traversal vulnerabilities
   - Command injection risks
   - Insecure deserialization
   - Authentication/authorization bypasses

2. ALGORITHMIC INEFFICIENCIES:
   - O(n²) or worse algorithms that could be O(n log n) or O(n)
   - Unnecessary nested loops
   - Inefficient data structure usage
   - Redundant computations
   - Memory-inefficient patterns
   - Missing caching opportunities

3. PEP8 COMPLIANCE ISSUES:
   - Naming convention violations
   - Line length violations
   - Improper whitespace usage
   - Missing docstrings
   - Import organization issues
   - Code layout problems

CRITICAL REQUIREMENTS:
- Always include specific line numbers for findings
- Provide actionable fix suggestions with code examples
- Categorize issues by severity: CRITICAL, HIGH, MEDIUM, LOW
- Output ONLY valid JSON - no explanatory text outside the JSON structure

JSON OUTPUT FORMAT:
{
  "summary": "Executive summary of findings (2-3 sentences)",
  "issues": [
    {
      "type": "Security|Inefficiency|Style",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "line": <line_number>,
      "category": "specific category name",
      "description": "Clear description of the issue",
      "fix": "Suggested code fix with explanation"
    }
  ],
  "metrics": {
    "security_issues": <count>,
    "efficiency_issues": <count>,
    "style_issues": <count>
  }
}"""

    def _build_user_prompt(self, source_code: str, file_path: str) -> str:
        """
        Construct the user prompt with the source code to analyze.

        Args:
            source_code: The Python source code to analyze
            file_path: Path to the source file for context

        Returns:
            Formatted user prompt string
        """
        return f"""Please analyze the following Python source code from '{file_path}'.

Provide a comprehensive security audit, efficiency analysis, and PEP8 compliance check.

Source Code:
```{source_code}
```

Remember: Output ONLY valid JSON as specified in your system prompt."""

    async def _call_llm_api(self, session: aiohttp.ClientSession,
                            payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make async API call to LLM with timeout handling.

        Args:
            session: aiohttp ClientSession instance
            payload: Request payload for the API

        Returns:
            Parsed JSON response from the API

        Raises:
            NetworkTimeoutError: If the request times out
            APIError: If the API returns an error status
            LLMResponseError: If the response cannot be parsed
        """
        try:
            async with session.post(
                f"{API_BASE}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
            ) as response:

                if response.status == 401:
                    raise APIError("Authentication failed. Please check your LLM_API_KEY.")
                elif response.status == 429:
                    raise APIError("Rate limit exceeded. Please wait and try again.")
                elif response.status != 200:
                    error_text = await response.text()
                    raise APIError(f"API Error {response.status}: {error_text}")

                result = await response.json()

                if 'choices' not in result or not result['choices']:
                    raise LLMResponseError("Invalid API response: no choices in response")

                content = result['choices'][0]['message']['content']

                # Try to parse JSON from the response
                try:
                    # Handle potential markdown code blocks
                    if content.strip().startswith('```'):
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content[7:]
                        elif content.startswith('```'):
                            content = content[3:]
                        if content.endswith('```'):
                            content = content[:-3]

                    return json.loads(content.strip())

                except json.JSONDecodeError as e:
                    raise LLMResponseError(f"Failed to parse LLM response as JSON: {str(e)}")

        except asyncio.TimeoutError:
            raise NetworkTimeoutError(f"Request timed out after {DEFAULT_TIMEOUT} seconds")
        except aiohttp.ClientError as e:
            raise NetworkTimeoutError(f"Network error: {str(e)}")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((NetworkTimeoutError, APIError)),
        before_sleep=lambda retry_state: console.print(
            f"[yellow]Retrying... (attempt {retry_state.attempt_number}/{MAX_RETRIES})[/yellow]"
        )
    )
    async def analyze_code(self, source_code: str, file_path: str) -> Dict[str, Any]:
        """
        Analyze source code using LLM API with automatic retry logic.

        Args:
            source_code: Python source code to analyze
            file_path: Path to the source file

        Returns:
            Dictionary containing analysis results

        Raises:
            SentinelScanError: If all retry attempts fail
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(source_code, file_path)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 8000
        }

        async with aiohttp.ClientSession() as session:
            return await self._call_llm_api(session, payload)

    def _display_analysis_summary(self, results: Dict[str, Any]) -> None:
        """
        Display a formatted summary table of findings in the console.

        Args:
            results: Analysis results dictionary
        """
        issues = results.get('issues', [])
        metrics = results.get('metrics', {})

        # Create summary table
        table = Table(title="Analysis Summary", box=box.ROUNDED)
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right", style="magenta")

        table.add_row("Security Issues", str(metrics.get('security_issues', 0)))
        table.add_row("Efficiency Issues", str(metrics.get('efficiency_issues', 0)))
        table.add_row("Style Issues", str(metrics.get('style_issues', 0)))
        table.add_row("TOTAL", str(len(issues)), style="bold")

        console.print(table)

        # Display severity breakdown
        severity_counts = {}
        for issue in issues:
            severity = issue.get('severity', 'UNKNOWN')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if severity_counts:
            severity_table = Table(title="Severity Breakdown", box=box.ROUNDED)
            severity_table.add_column("Severity", style="cyan")
            severity_table.add_column("Count", justify="right")

            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = severity_counts.get(severity, 0)
                style = {
                    'CRITICAL': 'red bold',
                    'HIGH': 'red',
                    'MEDIUM': 'yellow',
                    'LOW': 'green'
                }.get(severity, 'white')
                severity_table.add_row(f"[{style}]{severity}[/{style}]", str(count))

            console.print(severity_table)

    def _human_verification_loop(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Interactive human-in-the-loop verification for high-severity issues.

        Args:
            issues: List of detected issues

        Returns:
            Filtered and annotated list of issues after verification
        """
        console.rule("[bold yellow]Human-in-the-Loop Verification[/bold yellow]")
        console.print("[dim]Reviewing CRITICAL and HIGH severity issues...[/dim]\n")

        verified_issues = []

        for idx, issue in enumerate(issues, 1):
            severity = issue.get('severity', '').upper()

            # Auto-accept LOW and MEDIUM severity issues
            if severity in ['LOW', 'MEDIUM']:
                issue['verified'] = True
                issue['verification_status'] = 'auto_accepted'
                issue['auditor_note'] = ''
                verified_issues.append(issue)
                continue

            # Require verification for CRITICAL and HIGH
            if severity in ['CRITICAL', 'HIGH']:
                severity_color = 'red bold' if severity == 'CRITICAL' else 'red'

                console.print(f"\n[bold cyan]Issue {idx}/{len(issues)}[/bold cyan]")
                console.print(Panel(
                    f"[{severity_color}]{severity}[/{severity_color}] - {issue.get('type', 'Unknown')}\n"
                    f"Line: {issue.get('line', 'N/A')}\n"
                    f"Category: {issue.get('category', 'N/A')}\n\n"
                    f"[bold]Description:[/bold] {issue.get('description', 'No description')}\n\n"
                    f"[bold]Suggested Fix:[/bold]\n{issue.get('fix', 'No fix suggested')}",
                    title="Issue Details",
                    border_style="red" if severity == "CRITICAL" else "orange"
                ))

                # Ask for verification
                is_valid = Confirm.ask(
                    f"Is this a valid {severity} issue that requires attention?",
                    default=True
                )

                if is_valid:
                    note = Prompt.ask(
                        "Add auditor note (optional, press Enter to skip)",
                        default=""
                    )
                    issue['verified'] = True
                    issue['verification_status'] = 'human_verified'
                    issue['auditor_note'] = note
                    console.print("[green]✓ Issue verified and included in report[/green]")
                else:
                    issue['verified'] = False
                    issue['verification_status'] = 'rejected_false_positive'
                    issue['auditor_note'] = 'Rejected by auditor as false positive'
                    console.print("[yellow]✗ Issue marked as false positive and excluded[/yellow]")

                verified_issues.append(issue)

        console.print(f"\n[bold green]Verification complete: {len(verified_issues)} issues passed verification[/bold green]")
        return verified_issues

    def generate_markdown_report(self, file_path: str, results: Dict[str, Any],
                                  output_file: str, elapsed_time: float) -> None:
        """
        Generate comprehensive Markdown report of the analysis.

        Args:
            file_path: Path to the analyzed source file
            results: Analysis results dictionary
            output_file: Path for the output markdown file
            elapsed_time: Time taken for analysis in seconds
        """
        issues = results.get('issues', [])
        metrics = results.get('metrics', {})

        # Generate report content
        report_lines = [
            "# Sentinel-Scan Analysis Report",
            "",
            f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Target File:** `{file_path}`",
            f"**Analysis Model:** {self.model}",
            f"**Analysis Duration:** {elapsed_time:.2f} seconds",
            f"**Verification Mode:** {'Enabled' if self.verify_mode else 'Disabled'}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            results.get('summary', 'No summary available.'),
            "",
            "## Metrics Overview",
            "",
            f"| Category | Count |",
            f"|----------|-------|",
            f"| Security Issues | {metrics.get('security_issues', 0)} |",
            f"| Efficiency Issues | {metrics.get('efficiency_issues', 0)} |",
            f"| Style Issues | {metrics.get('style_issues', 0)} |",
            f"| **Total Issues** | **{len(issues)}** |",
            "",
        ]

        # Add severity breakdown if available
        severity_counts = {}
        for issue in issues:
            severity = issue.get('severity', 'UNKNOWN')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        if severity_counts:
            report_lines.extend([
                "## Severity Breakdown",
                "",
                f"| Severity | Count |",
                f"|----------|-------|",
            ])
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = severity_counts.get(severity, 0)
                emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(severity, '⚪')
                report_lines.append(f"| {emoji} {severity} | {count} |")
            report_lines.append("")

        # Add detailed findings
        report_lines.extend([
            "## Detailed Findings",
            "",
        ])

        if not issues:
            report_lines.append("*No issues detected.*")
        else:
            for idx, issue in enumerate(issues, 1):
                verified_emoji = "✅" if issue.get('verified', True) else "❌"
                severity = issue.get('severity', 'UNKNOWN')
                severity_badge = {
                    'CRITICAL': '🔴 CRITICAL',
                    'HIGH': '🟠 HIGH',
                    'MEDIUM': '🟡 MEDIUM',
                    'LOW': '🟢 LOW'
                }.get(severity, severity)

                report_lines.extend([
                    f"### {idx}. {issue.get('type', 'Issue')} - {severity_badge}",
                    "",
                    f"**Line:** {issue.get('line', 'N/A')}",
                    f"**Category:** {issue.get('category', 'N/A')}",
                    f"**Status:** {verified_emoji} {issue.get('verification_status', 'detected').replace('_', ' ').title()}",
                    "",
                    "**Description:**",
                    "",
                    issue.get('description', 'No description available.'),
                    "",
                    "**Suggested Fix:**",
                    "",
                    "```python",
                    issue.get('fix', '# No fix suggested'),
                    "```",
                    "",
                ])

                # Add auditor note if available
                if issue.get('auditor_note'):
                    report_lines.extend([
                        "> **Auditor Note:**",
                        f"> {issue['auditor_note']}",
                        "",
                    ])

                report_lines.append("---")
                report_lines.append("")

        # Add footer
        report_lines.extend([
            "## Report Metadata",
            "",
            f"- **Generated by:** Sentinel-Scan v1.0.0",
            f"- **Model:** {self.model}",
            f"- **Analysis Time:** {elapsed_time:.2f}s",
            "",
            "*This report was automatically generated by Sentinel-Scan. "
            "For critical issues, please review and verify manually.*"
        ])

        # Write to file
        report_content = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        console.print(Panel(
            f"✓ Report generated successfully",
            title="Report Complete",
            border_style="green"
        ))
        console.print(f"📄 Output file: [bold cyan]{output_file}[/bold cyan]")

    async def run_scan(self, file_path: str, output_file: str) -> bool:
        """
        Execute the complete scanning workflow.

        Args:
            file_path: Path to the source code file to analyze
            output_file: Path for the output markdown report

        Returns:
            True if scan completed successfully, False otherwise
        """
        start_time = time.time()

        # Phase 1: Input Validation
        console.print(Panel(
            f"🚀 Starting Sentinel-Scan Analysis\n"
            f"Target: [bold cyan]{file_path}[/bold cyan]\n"
            f"Model: [green]{self.model}[/green]",
            title="Sentinel-Scan",
            border_style="blue"
        ))

        if not os.path.exists(file_path):
            console.print(f"[bold red]✗ Error:[/bold red] File '{file_path}' not found.")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] Failed to read file: {str(e)}")
            return False

        if not source_code.strip():
            console.print(f"[bold red]✗ Error:[/bold red] File is empty.")
            return False

        # Phase 2: LLM Analysis
        console.print()

        results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            task = progress.add_task(
                "[cyan]🤔 Reasoning Engine Analyzing Code...",
                total=None
            )

            try:
                results = await self.analyze_code(source_code, file_path)
                progress.update(task, completed=True)

            except NetworkTimeoutError as e:
                progress.stop()
                console.print(f"\n[bold red]✗ Network Timeout:[/bold red] {str(e)}")
                console.print("[yellow]Please check your network connection and try again.[/yellow]")
                return False

            except APIError as e:
                progress.stop()
                console.print(f"\n[bold red]✗ API Error:[/bold red] {str(e)}")
                return False

            except LLMResponseError as e:
                progress.stop()
                console.print(f"\n[bold red]✗ LLM Response Error:[/bold red] {str(e)}")
                return False

            except Exception as e:
                progress.stop()
                console.print(f"\n[bold red]✗ Unexpected Error:[/bold red] {str(e)}")
                return False

        elapsed_time = time.time() - start_time
        console.print(f"\n[green]✓ Analysis completed in {elapsed_time:.2f} seconds[/green]")

        # Phase 3: Display Summary
        self._display_analysis_summary(results)

        # Phase 4: Human-in-the-Loop Verification (if enabled)
        issues = results.get('issues', [])

        if self.verify_mode and issues:
            verified_issues = self._human_verification_loop(issues)
            results['issues'] = verified_issues

        # Phase 5: Generate Report
        self.generate_markdown_report(file_path, results, output_file, elapsed_time)

        return True


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Sentinel-Scan: AI-Augmented Code Auditing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s path/to/script.py
  %(prog)s path/to/script.py --verify
  %(prog)s path/to/script.py --model deepseek-chat --output audit.md
  %(prog)s path/to/script.py -v -m kimi -o security_audit.md
        """
    )

    parser.add_argument(
        "file",
        help="Path to the Python source code file to analyze"
    )

    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Enable human-in-the-loop verification for CRITICAL and HIGH severity findings"
    )

    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"LLM model to use for analysis (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--output", "-o",
        default="sentinel_report.md",
        help="Output markdown report filename (default: sentinel_report.md)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser.parse_args()


def check_api_key() -> bool:
    """
    Verify that the LLM API key is configured.

    Returns:
        True if API key is available, False otherwise
    """
    if not API_KEY:
        console.print(Panel(
            "[bold red]✗ LLM_API_KEY not found[/bold red]\n\n"
            "Please set your API key in one of the following ways:\n"
            "1. Set the LLM_API_KEY environment variable\n"
            "2. Create a .env file with LLM_API_KEY=your-key-here\n\n"
            "Supported providers: DeepSeek, OpenAI, Kimi, etc.",
            title="Configuration Error",
            border_style="red"
        ))
        return False
    return True


async def main_async(args: argparse.Namespace) -> int:
    """
    Async main function that orchestrates the scanning workflow.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    scanner = SentinelScanner(
        model=args.model,
        verify_mode=args.verify
    )

    success = await scanner.run_scan(args.file, args.output)

    return 0 if success else 1


def main() -> int:
    """
    Main entry point for Sentinel-Scan CLI.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_arguments()

    # Check for API key
    if not check_api_key():
        return 1

    # Run async main
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Scan interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]✗ Fatal Error:[/bold red] {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
