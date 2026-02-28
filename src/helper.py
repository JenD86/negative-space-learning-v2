from contextlib import contextmanager
from datetime import datetime
import hashlib
import random
import string
from typing import (
    Callable,
)

import subprocess
import threading


def nanoid(size=21) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(size))


@contextmanager
def timeout(seconds: int, callback: Callable = lambda: print()):
    """
    Context manager that raises a TimeoutError if the code inside the context takes longer than the specified time.

    This implementation uses threading.Timer which is thread-safe, unlike signal-based approaches.

    Args:
        seconds (int): Maximum number of seconds to allow the code to run

    Yields:
        None: The context to execute code within the timeout constraint

    Raises:
        TimeoutError: If the code execution exceeds the specified timeout

    Example:
        >>> with timeout(5):
        ...     # Code that should complete within 5 seconds
        ...     long_running_function()
    """
    timer = None
    exception = TimeoutError(f"Execution timed out after {seconds} seconds")

    def timeout_handler():
        nonlocal timer
        callback()
        raise exception

    timer = threading.Timer(seconds, timeout_handler)
    timer.start()

    try:
        yield
    finally:
        if timer:
            timer.cancel()


def int_to_ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def unflatten_toml_dict(d: dict) -> dict:
    result = {}
    for key, value in d.items():
        parts = key.split(".")
        current_level = result
        for i, part in enumerate(parts):
            if i == len(parts) - 1:  # Last part, assign value
                current_level[part] = value
            else:
                current_level = current_level.setdefault(part, {})

    return result


def generate_readable_run_id(
    random_length: int = 6, date_format_str: str = "%Y%m%d", separator: str = "-"
) -> str:
    current_date = datetime.now()

    date_part = current_date.strftime(date_format_str)

    characters = string.ascii_lowercase + string.digits
    random_part = "".join(random.choices(characters, k=random_length))

    # 4. Combine the parts
    run_id = f"{date_part}{separator}{random_part}"

    return run_id


def get_formatted_repo_info():
    """
    Gets current Git repository information formatted as:
    "{commit-hash-capitalized}-{branch-capitalized}-{number-of-uncommitted-files}"
    
    If not in a git repository, returns a default identifier based on timestamp.
    """
    try:
        # 1. Preliminary check: Is this a Git repository?
        subprocess.check_output(
            ["git", "rev-parse", "--git-dir"],
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # 2. Get current commit hash.
        commit_hash_raw = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
        commit_hash_lower = commit_hash_raw.lower()

        # 3. Get current branch name.
        branch_name_raw = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        branch_name_lower = branch_name_raw.lower()

        # 4. Get hash of uncommitted files.
        status_output = subprocess.check_output(
            ["git", "diff", "--stat"], text=True
        ).strip()

        uncommitted_files_hash = "0"
        if status_output:
            uncommitted_files_hash = string_hash(status_output)

        return f"{commit_hash_lower[:6]}-{branch_name_lower}-{uncommitted_files_hash[:6]}"
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repository or git command failed - return default identifier
        import time
        timestamp = str(int(time.time()))
        return f"nogit-{timestamp[-6:]}-000000"


def string_hash(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()
