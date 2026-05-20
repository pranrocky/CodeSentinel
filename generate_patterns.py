# generate_patterns.py
import json
import os
import config

patterns = [
    {
        "pattern": "Mutable default argument",
        "severity": "high",
        "description": "Using list/dict as default arg creates shared state across calls.",
        "bad": "def fn(x, lst=[]):\n    lst.append(x); return lst",
        "good": "def fn(x, lst=None):\n    if lst is None: lst = []\n    lst.append(x); return lst",
        "why": "Default args are evaluated once at function definition, not per call."
    },
    {
        "pattern": "Bare except clause",
        "severity": "medium",
        "description": "Catching all exceptions hides bugs and makes Ctrl+C (KeyboardInterrupt) fail.",
        "bad": "try:\n    do_work()\nexcept:\n    pass",
        "good": "try:\n    do_work()\nexcept Exception as e:\n    log_error(e)",
        "why": "It catches SystemExit and KeyboardInterrupt, breaking standard termination."
    },
    {
        "pattern": "SQL Injection via string formatting",
        "severity": "critical",
        "description": "Using f-strings or .format() to build SQL queries allows injection attacks.",
        "bad": "cursor.execute(f'SELECT * FROM users WHERE name = \"{user_input}\"')",
        "good": "cursor.execute('SELECT * FROM users WHERE name = %s', (user_input,))",
        "why": "String formatting does not escape malicious SQL control characters."
    }
]

for i, pattern in enumerate(patterns):
    filepath = os.path.join(config.BUG_PATTERNS_DIR, f"{i:03d}_{pattern['pattern'].replace(' ', '_').lower()}.json")
    with open(filepath, 'w') as f:
        json.dump(pattern, f, indent=2)

print(f"Successfully generated {len(patterns)} bug patterns in {config.BUG_PATTERNS_DIR}")