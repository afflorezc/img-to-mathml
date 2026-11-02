"""Command line interface for the image-to-MathML module."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import Settings
from .llm.base import LLMError
from .llm.factory import build_llm
from .mathml.validator import validate_mathml
from .models import EquationRequest, Flow, MathContext
from .pipeline import EquationPipeline, compare_flows


def _settings_from_args(args: argparse.Namespace) -> Settings:
    changes: dict[str, object] = {}
    if getattr(args, "model", None):
        changes["ollama_model"] = args.model
    backend = getattr(args, "backend", None) or Settings.from_env().llm_backend
    backend = backend.strip().lower()
    if getattr(args, "backend", None):
        changes["llm_backend"] = backend
    if getattr(args, "base_url", None):
        if backend in {"openai", "openai-compatible", "vllm", "vert", "lmstudio"}:
            changes["openai_base_url"] = args.base_url
        else:
            changes["ollama_base_url"] = args.base_url
    if getattr(args, "api_key", None):
        changes["openai_api_key"] = args.api_key
    if getattr(args, "timeout", None) is not None:
        changes["ollama_timeout"] = args.timeout
    if getattr(args, "max_repairs", None) is not None:
        changes["max_repair_attempts"] = args.max_repairs
    if getattr(args, "ocr", None):
        changes["ocr_backend"] = args.ocr
    if getattr(args, "strict_core", False):
        changes["allow_open_concepts"] = False
    if getattr(args, "allow_open_concepts", False):
        changes["allow_all_open_concepts"] = True
    if getattr(args, "allowed_open_concepts", None):
        changes["allowed_open_concepts"] = tuple(
            item.strip()
            for item in args.allowed_open_concepts.split(",")
            if item.strip()
        )
    return replace(Settings.from_env(), **changes)


def _read_latex(args: argparse.Namespace) -> str | None:
    latex = getattr(args, "latex", None)
    image = getattr(args, "image", None)
    if latex == "-":
        return sys.stdin.read()
    if latex:
        return latex
    if not image:
        if not sys.stdin.isatty():
            return sys.stdin.read()
        raise SystemExit(
            "Provide --latex, --image, or pipe a LaTeX expression on stdin."
        )
    return None


def _request_from_args(args: argparse.Namespace) -> EquationRequest:
    latex = _read_latex(args)
    return EquationRequest(
        latex=latex,
        image_path=getattr(args, "image", None),
        context=MathContext(
            area=getattr(args, "area", "") or "",
            topic=getattr(args, "topic", "") or "",
            surrounding_text=getattr(args, "context_text", "") or "",
            language=getattr(args, "lang", "es") or "es",
        ),
        flow=Flow(getattr(args, "flow", "hybrid")),
        ocr_backend=getattr(args, "ocr", None),
        allow_open_concepts=False if getattr(args, "strict_core", False) else None,
        max_repair_attempts=getattr(args, "max_repairs", None),
    )


def _print_report(result, stream=sys.stderr) -> None:
    report = result.validation
    for issue in report.issues:
        print(issue.format(), file=stream)
    print(
        f"validation: {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s), {report.intent_count} intent(s), "
        f"{report.redundant_mrows} redundant mrow(s)",
        file=stream,
    )


def cmd_convert(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    pipeline = EquationPipeline(settings)
    request = _request_from_args(args)
    result = pipeline.run(request)

    if getattr(args, "output", None):
        Path(args.output).write_text(result.mathml + "\n", encoding="utf-8")

    if getattr(args, "json", False):
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.mathml)

    if not getattr(args, "quiet", False):
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if getattr(args, "report", False):
            _print_report(result)

    return 0 if result.validation.ok else 2


def _comparison_markdown(results: dict[str, object]) -> str:
    lines = [
        "# Comparación de flujos",
        "",
        "| flujo | XML válido | aceptable | intents | errores | warnings | intentos | segundos |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, result in results.items():
        report = result.validation
        lines.append(
            f"| {name} | {'sí' if report.ok else 'no'} | "
            f"{'sí' if report.acceptable else 'no'} | {report.intent_count} | "
            f"{len(report.errors)} | {len(report.warnings)} | "
            f"{result.attempts} | {result.elapsed_seconds:.2f} |"
        )
    lines.append("")
    for name, result in results.items():
        if result.warnings:
            lines.append(f"## {name}: advertencias")
            lines.extend(f"- {warning}" for warning in result.warnings)
            lines.append("")
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    pipeline = EquationPipeline(settings)
    request = _request_from_args(args)
    results = compare_flows(request, pipeline)

    payload = {name: result.as_dict() for name, result in results.items()}
    if getattr(args, "output_dir", None):
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            (directory / f"{name}.mathml").write_text(
                result.mathml + "\n", encoding="utf-8"
            )
        (directory / "comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown = _comparison_markdown(results)
        (directory / "comparison.md").write_text(markdown + "\n", encoding="utf-8")

    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif getattr(args, "output_dir", None):
        print(_comparison_markdown(results))
        print(f"\nResultados escritos en {args.output_dir}", file=sys.stderr)
    else:
        print(_comparison_markdown(results))

    return 0 if all(result.validation.ok for result in results.values()) else 2


def cmd_check(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    if getattr(args, "file", None):
        markup = Path(args.file).read_text(encoding="utf-8")
    elif getattr(args, "mathml", None):
        markup = args.mathml
    else:
        markup = sys.stdin.read()

    allowed_open = (
        None
        if settings.allow_all_open_concepts
        else set(settings.allowed_open_concepts)
    )
    if getattr(args, "allowed_open_concepts", None):
        allowed_open = {
            item.strip()
            for item in args.allowed_open_concepts.split(",")
            if item.strip()
        }
    report = validate_mathml(
        markup,
        allow_open_concepts=not getattr(args, "strict_core", False),
        allowed_open_concepts=allowed_open,
    )
    if getattr(args, "json", False):
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        for issue in report.issues:
            print(issue.format())
        print(
            f"{'OK' if report.ok else 'FAIL'}: {len(report.errors)} error(s), "
            f"{len(report.warnings)} warning(s), {report.intent_count} intent(s), "
            f"{report.redundant_mrows} redundant mrow(s)"
        )
    return 0 if report.ok else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    modules = {
        "latex2mathml": importlib.util.find_spec("latex2mathml") is not None,
        "requests": importlib.util.find_spec("requests") is not None,
        "lxml": importlib.util.find_spec("lxml") is not None,
        "Pillow": importlib.util.find_spec("PIL") is not None,
        "pix2tex": importlib.util.find_spec("pix2tex") is not None,
        "texify": importlib.util.find_spec("texify") is not None,
    }
    client = build_llm(settings)
    available = client.is_available()
    models: list[str] = []
    if available:
        try:
            models = client.list_models()
        except LLMError:
            models = []
    has_model = any(
        name == settings.ollama_model
        or name.startswith(settings.ollama_model + ":")
        or settings.ollama_model in name
        for name in models
    )

    payload = {
        "python": sys.version.split()[0],
        "modules": modules,
        "llm": {
            "backend": settings.llm_backend,
            "base_url": getattr(client, "base_url", settings.ollama_base_url),
            "available": available,
            "model": settings.ollama_model,
            "model_installed": has_model,
            "models": models,
        },
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Python: {payload['python']}")
        for name, installed in modules.items():
            print(f"{name}: {'installed' if installed else 'not installed'}")
        print(f"LLM backend: {settings.llm_backend}")
        print(f"LLM URL: {payload['llm']['base_url']}")
        print(f"LLM reachable: {'yes' if available else 'no'}")
        print(
            f"Model {settings.ollama_model}: {'installed' if has_model else 'missing'}"
        )
        if models:
            print("Installed models: " + ", ".join(models))

    required = ("latex2mathml", "requests", "lxml", "Pillow")
    ok = all(modules[name] for name in required) and available and has_model
    return 0 if ok else 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latex", "-l", help="LaTeX expression, or '-' for stdin")
    parser.add_argument("--image", "-i", help="Equation image path")
    parser.add_argument(
        "--area", default="", help="Mathematics area from the book title"
    )
    parser.add_argument(
        "--topic", default="", help="Topic detected by the previous stage"
    )
    parser.add_argument("--context-text", default="", help="Surrounding OCR text")
    parser.add_argument("--lang", default="es", help="Reader language (default: es)")
    parser.add_argument(
        "--flow",
        choices=[flow.value for flow in Flow],
        default=Flow.HYBRID.value,
        help="Generation flow (default: hybrid)",
    )
    parser.add_argument("--ocr", choices=["auto", "texify", "pix2tex", "none"])
    parser.add_argument(
        "--strict-core",
        action="store_true",
        help="Reject all open concepts, including gamma-function",
    )
    parser.add_argument(
        "--allow-open-concepts",
        action="store_true",
        help="Allow any documented W3C open concept (default: only the "
        "documented gamma-function, beta-function and error-function exceptions)",
    )
    parser.add_argument(
        "--allowed-open-concepts",
        default=None,
        help="Comma-separated open concepts to allow, e.g. "
        "gamma-function,airy-function",
    )
    parser.add_argument("--max-repairs", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--backend",
        default=None,
        help="LLM backend: ollama (default) or an OpenAI-compatible server "
        "(openai, vllm, vert, lmstudio, openai-compatible)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM server base URL. For OpenAI-compatible servers include /v1.",
    )
    parser.add_argument("--api-key", default=None, help="API key for the LLM server")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--output", "-o", help="Write the MathML result to this file")
    parser.add_argument("--quiet", "-q", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-to-mathml",
        description=(
            "Local equation recognition: image/LaTeX -> accessible MathML "
            "with W3C intent attributes."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser(
        "convert", help="Recognise one equation and produce MathML"
    )
    _add_common(convert)
    convert.set_defaults(func=cmd_convert)

    compare = subparsers.add_parser(
        "compare", help="Run hybrid, direct and offline flows for the same input"
    )
    _add_common(compare)
    compare.add_argument(
        "--output-dir", help="Write <flow>.mathml, comparison.json and comparison.md"
    )
    compare.set_defaults(func=cmd_compare)

    check = subparsers.add_parser("check", help="Validate a MathML + intent document")
    check.add_argument("--file", "-f", help="MathML file to validate")
    check.add_argument("--mathml", help="MathML string to validate")
    check.add_argument("--strict-core", action="store_true")
    check.add_argument("--allow-open-concepts", action="store_true")
    check.add_argument("--allowed-open-concepts", default=None)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_check)

    doctor = subparsers.add_parser("doctor", help="Check local dependencies and Ollama")
    doctor.add_argument("--model", default=None)
    doctor.add_argument("--backend", default=None)
    doctor.add_argument("--base-url", default=None)
    doctor.add_argument("--api-key", default=None)
    doctor.add_argument("--timeout", type=float, default=None)
    doctor.add_argument("--strict-core", action="store_true")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
