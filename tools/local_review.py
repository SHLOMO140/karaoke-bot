"""Local adversarial code review via Ollama (Codex CLI replacement).

Feeds a git diff to a local model (default: qwen3-coder:30b) chunk by chunk
and prints only concrete defects with file:line and a failure scenario.

Usage examples:
    python tools\\local_review.py                       # review working tree vs HEAD
    python tools\\local_review.py --range main..HEAD    # review a branch
    python tools\\local_review.py --staged              # review staged changes
    python tools\\local_review.py --focus "concurrency and Hebrew text handling"

The machine is CPU-only: expect a few minutes per chunk. Requires the Ollama
service (starts automatically on login; otherwise run `ollama serve`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-coder:30b"
MAX_CHUNK_CHARS = 14_000

SYSTEM_PROMPT = (
    "You are an adversarial senior code reviewer. You receive a git diff. "
    "Report ONLY real defects that would cause wrong behavior, crashes, data "
    "loss, security issues, or deadlocks. For each finding give: file, line, "
    "a one-sentence summary, and a concrete failure scenario (inputs/state -> "
    "wrong outcome). Do NOT report style, naming, or formatting. Hebrew "
    "strings in the code are intentional; verify they are well-formed but do "
    "not translate them. If a chunk contains no real defects, answer exactly: "
    "NO DEFECTS FOUND."
)


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _collect_diff(args: argparse.Namespace) -> str:
    if args.diff_file:
        with open(args.diff_file, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    git_args = ["diff", "--unified=6"]
    if args.staged:
        git_args.append("--cached")
    elif args.range:
        git_args.append(args.range)
    return _git(git_args)


def _split_by_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a unified diff into (filename, file_diff) pairs."""
    files: list[tuple[str, str]] = []
    current_name = ""
    current_lines: list[str] = []
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current_lines:
                files.append((current_name, "".join(current_lines)))
            current_lines = [line]
            parts = line.split(" b/")
            current_name = parts[-1].strip() if len(parts) > 1 else line.strip()
        else:
            current_lines.append(line)
    if current_lines:
        files.append((current_name, "".join(current_lines)))
    return files


def _build_chunks(diff_text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for name, file_diff in _split_by_file(diff_text):
        if len(file_diff) > max_chars:
            # A single huge file: split on hunks.
            if current:
                chunks.append("".join(current))
                current, current_size = [], 0
            hunk_chunk = ""
            for hunk in file_diff.split("\n@@"):
                piece = hunk if not hunk_chunk else "\n@@" + hunk
                if len(hunk_chunk) + len(piece) > max_chars and hunk_chunk:
                    chunks.append(f"[file: {name}, partial]\n{hunk_chunk}")
                    hunk_chunk = f"diff --git (continued) {name}\n@@" + hunk
                else:
                    hunk_chunk += piece
            if hunk_chunk:
                chunks.append(f"[file: {name}, partial]\n{hunk_chunk}")
            continue
        if current_size + len(file_diff) > max_chars and current:
            chunks.append("".join(current))
            current, current_size = [], 0
        current.append(file_diff)
        current_size += len(file_diff)
    if current:
        chunks.append("".join(current))
    return chunks


def _chat(model: str, prompt: str, *, timeout: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 16384},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", "")).strip()


def _server_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5):
            return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", help="git range, e.g. main..HEAD (default: working tree vs HEAD)")
    parser.add_argument("--staged", action="store_true", help="review staged changes")
    parser.add_argument("--diff-file", help="review a saved diff file instead of git")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--focus", default="", help="extra reviewer focus, free text")
    parser.add_argument("--max-chunk-chars", type=int, default=MAX_CHUNK_CHARS)
    parser.add_argument("--timeout", type=int, default=1200, help="seconds per chunk (CPU is slow)")
    args = parser.parse_args(argv)

    if not _server_available():
        print("Ollama server is not reachable at localhost:11434 - run `ollama serve` first.")
        return 2

    diff_text = _collect_diff(args)
    if not diff_text.strip():
        print("Empty diff - nothing to review.")
        return 0

    chunks = _build_chunks(diff_text, args.max_chunk_chars)
    print(f"Reviewing {len(chunks)} chunk(s) with {args.model} (CPU - be patient)...\n")

    focus_line = f"\nExtra reviewer focus: {args.focus}\n" if args.focus else ""
    findings: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        started = time.perf_counter()
        prompt = f"Review this git diff.{focus_line}\n```diff\n{chunk}\n```"
        try:
            answer = _chat(args.model, prompt, timeout=args.timeout)
        except urllib.error.URLError as exc:
            print(f"[chunk {index}/{len(chunks)}] request failed: {exc}")
            continue
        elapsed = time.perf_counter() - started
        print(f"--- chunk {index}/{len(chunks)} ({elapsed:.0f}s) ---")
        print(answer or "(empty response)")
        print()
        if answer and "NO DEFECTS FOUND" not in answer.upper():
            findings.append(answer)

    print("=" * 60)
    if findings:
        print(f"{len(findings)} chunk(s) reported potential defects (see above).")
    else:
        print("No defects reported in any chunk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
