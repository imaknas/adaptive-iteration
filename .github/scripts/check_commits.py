"""Fail if any newly pushed commit's subject isn't a Conventional Commit.

Checks only the commits a push or pull request adds (older history predates the
convention). Merge commits are skipped.

    type(scope)!: summary     type = feat fix docs style refactor perf test build ci chore revert
"""
import os
import re
import subprocess
import sys

PATTERN = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\([a-z0-9._/-]+\))?!?: \S.*$"
)
ZERO = "0" * 40


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def commit_range() -> str:
    head = os.environ.get("HEAD_SHA") or "HEAD"
    base = os.environ.get("BASE_SHA")                 # pull requests
    before = os.environ.get("BEFORE_SHA")             # pushes
    start = base or (before if before and before != ZERO else None)
    if start:
        try:
            git("cat-file", "-e", f"{start}^{{commit}}")
            return f"{start}..{head}"
        except subprocess.CalledProcessError:
            pass                                      # e.g. force-push: old tip is gone
    return f"{head}~1..{head}"


def main() -> int:
    rng = commit_range()
    log = git("log", "--no-merges", "--format=%h %s", rng).splitlines()
    bad = [line for line in log if not PATTERN.match(line.split(" ", 1)[1])]
    print(f"checked {len(log)} commit(s) in {rng}")
    for line in bad:
        print(f"  not a Conventional Commit: {line}")
    if bad:
        print("expected: type(scope)!: summary — e.g. 'feat: add X', 'fix(loop): Y', "
              "'chore(release): 0.9.0'")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
