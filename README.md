# PySnapping

Snap points to a line string keeping a given order intact.

## Development

### Installation

```bash
python3 -m venv env
source venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt -r dev-requirements.txt -e .
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

### Dependencies

Edit [setup.py](setup.py) and/or [dev-requirements.in](dev-requirements.in)
then run:

```bash
pip-compile
pip-compile dev-requirements.in
```

Check in changed files to repo.
