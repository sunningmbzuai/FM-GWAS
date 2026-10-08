#!/usr/bin/env python
r"""Strip absolute-import fallbacks that break transformers' check_imports.

transformers.dynamic_module_utils.get_imports() removes only the *try* body
(regex: \s*try\s*:.*?except.*?:), so an `except ImportError:` fallback of the
form `from configuration_transunet import X` survives as a top-level import and
gets reported as a missing pip package:

    ImportError: This modeling file requires the following packages that were
    not found in your environment: configuration_transunet

Relative imports alone are sufficient -- HF copies every sibling file into one
package directory under
~/.cache/huggingface/modules/transformers_modules/<name>/.

Usage:
    python fix_hf_dynamic_imports.py <model_dir> [--dry-run]

After patching, clear the stale cached copy:
    rm -rf ~/.cache/huggingface/modules/transformers_modules/<model_dir_name>
"""
import argparse
import pathlib
import re

TRY_RELATIVE_EXCEPT_ABSOLUTE = re.compile(
    r"try:\n"
    r"(?P<body>(?:[ \t]+from \.[\w.]+ import [^\n]+\n)+)"
    r"except ImportError:\n"
    r"(?:[ \t]+from [\w.]+ import [^\n]+\n)+"
)


def dedent_body(match: re.Match) -> str:
    lines = match.group("body").strip().split("\n")
    return "".join(line.strip() + "\n" for line in lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="directory holding configuration_*.py / modeling_*.py")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"not a directory: {model_dir}")

    changed = 0
    for path in sorted(model_dir.glob("*.py")):
        source = path.read_text()
        patched = TRY_RELATIVE_EXCEPT_ABSOLUTE.sub(dedent_body, source)
        if patched == source:
            continue
        changed += 1
        if args.dry_run:
            print(f"would patch {path.name}")
        else:
            path.write_text(patched)
            print(f"patched {path.name}")

    if changed == 0:
        print("no try/except import fallbacks found")
    elif not args.dry_run:
        print(
            "\nnow clear the cached copy:\n"
            f"  rm -rf ~/.cache/huggingface/modules/transformers_modules/{model_dir.name}"
        )


if __name__ == "__main__":
    main()
