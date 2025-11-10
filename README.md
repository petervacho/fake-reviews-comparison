# AP9SI fake reviews comparison

This project is made as a part of the final thesis for the AP9SI class: **A systematic review of methods for detecting
fake reviews using ML**

## Installation & Running

This project uses [python] 3.14 or later.

The recommended way to run the script is to install the dependencies into a virtual environment. To set up this environment, you should run:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

You can then run the project as a module from the `__main__.py` script using the virtual environment's python interpreter:

```bash
.venv/bin/python -m src
```

## Development setup

If you wish to do some development work on this project, the setup process will be slightly different, as you will want to use the [uv] python project manger and install the development dependencies alongside the runtime ones. If you've already set up your virtual environment from above, you will want to remove it (`rm -rf .venv`), and set it up with uv.

1. Install uv, if you don't already have it on your system. See [docs][uv-installation] for reference.
2. Install the project using uv by running `uv sync` from the project's directory.
3. Activate the virutal environment: `.venv/bin/activate`.
4. Install pre-commit git hook to run the configured linters, formatters and other CI tools: `pre-commit install`.
5. You can now run the project with `python -m src` and do development.

## Usage rights

This is an open-source licensed under the terms of the GPL v3 license. For full license text, see the [`LICENSE.txt`](./LICENSE.txt) file. If you just want a quick overview of the rights and usage requirements under this license, you can see a summary [here][tldrlegal-gplv3].

[python]: https://www.python.org/
[uv]: https://docs.astral.sh/uv/
[uv-installation]: https://docs.astral.sh/uv/getting-started/installation/
[tldrlegal-gplv3]: https://www.tldrlegal.com/license/gnu-general-public-license-v3-gpl-3
