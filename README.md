# PySnapping

Snap points to a line string keeping a given order intact.

## Development

### Installation

```bash
python3.7 -m venv env
env/bin/pip install -U pip
env/bin/pip install -r requirements.txt -r dev-requirements.txt -e .
```

Keep env activated for all following instructions.

### Pre-Commit Hooks

```bash
pre-commit install
```

From time to time (not automated yet) run

```bash
pre-commit autoupdate
```

to update frozen revs.
