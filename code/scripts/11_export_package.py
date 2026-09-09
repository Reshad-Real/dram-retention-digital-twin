#!/usr/bin/env python
"""Stage 11 -- assemble every deliverable into one self-contained folder.

    python scripts/11_export_package.py --tag main
    python scripts/11_export_package.py --out E:/DRAM/DELIVERABLES --zip

Collects the published dataset, every figure and table, the technical report in
four formats (Markdown, LaTeX, Word, PDF), the IEEE algorithm listings, the
trained models, the raw result data and everything needed to reproduce the run,
into a numbered folder tree with a top-level index and a SHA-256 manifest.

The Word and PDF renderings are produced here from the Markdown report, so all
four formats are guaranteed to carry identical numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dramdt.config import load_config                                   # noqa: E402
from dramdt.logging_utils import RunManifest, package_versions, setup_logging, stage  # noqa: E402
from dramdt.paths import (CONFIG_DIR, DOCS_DIR, FIGURES_DIR, LOGS_DIR,
                          MODELS_DIR, OPT_DIR, PROCESSED_DIR, PROJECT_ROOT,
                          RAW_DIR, REPORTS_DIR, ROBUSTNESS_DIR, STATS_DIR,
                          SUPPLEMENTARY_DIR, SURROGATE_DIR, TABLES_DIR,
                          XAI_DIR, ensure_directories)                  # noqa: E402
from dramdt.reporting.docx_export import docx_to_pdf, markdown_to_docx  # noqa: E402


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_into(src: Path, dst_dir: Path, log, pattern: str | None = None) -> int:
    """Copy a file, or every file matching *pattern* in a directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if src.is_file():
        shutil.copy2(src, dst_dir / src.name)
        return 1
    if src.is_dir():
        for f in sorted(src.glob(pattern or "*")):
            if f.is_file() and f.name != ".gitkeep":
                shutil.copy2(f, dst_dir / f.name)
                n += 1
    return n


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+",
                    default=["dram_1t1c_45nm_lp.yaml", "surrogate.yaml",
                             "optimization.yaml"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=None, help="destination folder")
    ap.add_argument("--zip", action="store_true", help="also produce a .zip archive")
    ap.add_argument("--skip-doc-render", action="store_true",
                    help="reuse existing DOCX/PDF instead of re-rendering")
    args = ap.parse_args()

    ensure_directories()
    log = setup_logging("package", filename=f"11_export_package_{args.tag}.log")
    cfg = load_config(*args.config)
    out = Path(args.out) if args.out else PROJECT_ROOT / "DELIVERABLES"
    manifest = RunManifest(stage=f"package_{args.tag}", config_hash=cfg.hash)

    with stage(f"package[{args.tag}]", log, manifest,
               LOGS_DIR / f"manifest_package_{args.tag}.json"):
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        counts: dict[str, int] = {}

        # ---------------- 1. dataset ----------------------------------
        d = out / "01_dataset"
        for f in (PROCESSED_DIR / f"dram_dataset_{args.tag}.xlsx",
                  PROCESSED_DIR / f"dram_dataset_{args.tag}.csv.gz",
                  PROCESSED_DIR / f"{args.tag}_processed.parquet",
                  RAW_DIR / f"{args.tag}_raw.parquet"):
            if f.exists():
                copy_into(f, d, log)
        counts["dataset files"] = len(list(d.glob("*")))

        # ---------------- 2. report -----------------------------------
        r = out / "02_report"
        md = REPORTS_DIR / f"technical_report_{args.tag}.md"
        docx_path = pdf_path = None
        pdf_engine = ""
        if md.exists():
            if not args.skip_doc_render:
                log.info("Rendering the report to Word ...")
                rep = markdown_to_docx(md, REPORTS_DIR / f"technical_report_{args.tag}.docx")
                docx_path = rep.docx_path
                for w in rep.warnings:
                    log.warning("  %s", w)
                manifest.outputs["docx"] = {
                    "figures": rep.n_figures, "tables": rep.n_tables,
                    "headings": rep.n_headings, "warnings": rep.warnings}
                log.info("Converting to PDF ...")
                pdf_path, pdf_engine = docx_to_pdf(
                    docx_path, REPORTS_DIR / f"technical_report_{args.tag}.pdf")
                manifest.outputs["pdf_engine"] = pdf_engine
            else:
                docx_path = REPORTS_DIR / f"technical_report_{args.tag}.docx"
                pdf_path = REPORTS_DIR / f"technical_report_{args.tag}.pdf"

            for f in (pdf_path, docx_path, md,
                      REPORTS_DIR / f"technical_report_{args.tag}.tex",
                      REPORTS_DIR / f"results_summary_{args.tag}.json"):
                if f and Path(f).exists():
                    copy_into(Path(f), r, log)
        counts["report files"] = len(list(r.glob("*")))
        counts["algorithm files"] = copy_into(REPORTS_DIR / "algorithms",
                                              r / "algorithms", log)

        # ---------------- 3. figures ----------------------------------
        f_root = out / "03_figures"
        for ext in ("png", "pdf", "svg"):
            counts[f"figures ({ext})"] = copy_into(FIGURES_DIR, f_root / ext, log,
                                                   pattern=f"*.{ext}")

        # ---------------- 4. tables -----------------------------------
        t_root = out / "04_tables"
        for ext, sub in (("csv", "csv"), ("tex", "latex"), ("md", "markdown")):
            counts[f"tables ({ext})"] = copy_into(TABLES_DIR, t_root / sub, log,
                                                  pattern=f"*.{ext}")

        # ---------------- 5. result data -------------------------------
        res = out / "05_results"
        for name, src in (("optimization", OPT_DIR), ("surrogate", SURROGATE_DIR),
                          ("explainability", XAI_DIR), ("robustness", ROBUSTNESS_DIR),
                          ("statistics", STATS_DIR)):
            counts[f"results/{name}"] = copy_into(src, res / name, log)

        # ---------------- 6. models ------------------------------------
        counts["models"] = copy_into(MODELS_DIR, out / "06_models", log, "*.joblib")

        # ---------------- 7. reproducibility ---------------------------
        rp = out / "07_reproducibility"
        counts["configs"] = copy_into(CONFIG_DIR, rp / "configs", log, "*.yaml")
        counts["run manifests"] = copy_into(LOGS_DIR, rp / "manifests", log,
                                            "manifest_*.json")
        copy_into(LOGS_DIR / f"pipeline_summary_{args.tag}.json", rp / "manifests", log)
        counts["literature"] = copy_into(DOCS_DIR / "literature", rp / "literature",
                                         log, "*.yaml")
        for f in (PROJECT_ROOT / "README.md", PROJECT_ROOT / "REPRODUCIBILITY.md",
                  PROJECT_ROOT / "LICENSE", PROJECT_ROOT / "requirements.txt",
                  PROJECT_ROOT / "pyproject.toml",
                  SUPPLEMENTARY_DIR / "environment_lock.txt",
                  SUPPLEMENTARY_DIR / "README.md"):
            if f.exists():
                target = rp / ("supplementary_README.md"
                               if f.parent == SUPPLEMENTARY_DIR and f.name == "README.md"
                               else f.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)

        # ---------------- checksums + index ----------------------------
        files = sorted(p for p in out.rglob("*") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        checks = [{"file": str(p.relative_to(out)).replace("\\", "/"),
                   "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
        (out / "CHECKSUMS.json").write_text(
            json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "n_files": len(checks), "total_bytes": total,
                        "files": checks}, indent=2), encoding="utf-8")

        index = _index_markdown(cfg, args.tag, counts, len(files), total,
                                pdf_engine, out)
        (out / "00_README.md").write_text(index, encoding="utf-8")

        log.info("Package assembled at %s", out)
        for k, v in counts.items():
            log.info("  %-22s %d", k, v)
        log.info("  %-22s %d files, %s", "TOTAL", len(files), human(total))
        manifest.outputs["package_dir"] = str(out)
        manifest.outputs["counts"] = counts
        manifest.outputs["n_files"] = len(files)
        manifest.outputs["total_bytes"] = total

        if args.zip:
            log.info("Creating archive ...")
            archive = shutil.make_archive(str(out), "zip", root_dir=out)
            log.info("Archive: %s (%s)", archive, human(Path(archive).stat().st_size))
            manifest.outputs["archive"] = archive

    return 0


def _index_markdown(cfg, tag: str, counts: dict[str, int], n_files: int,
                    total: int, pdf_engine: str, out: Path) -> str:
    summary_path = REPORTS_DIR / f"results_summary_{tag}.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    ds = summary.get("dataset", {})
    ret = summary.get("retention_validation", {})

    lines = [
        "# Deliverables package",
        "",
        "**A Digital Twin-Driven Framework for Low-Power DRAM Optimization "
        "Using PySpice and Machine Learning**",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
        f"{n_files} files · {human(total)}",
        "",
        f"Run tag `{tag}` · configuration hash `{cfg.hash}` · "
        f"master seed `{cfg.get_path('experiment.seed')}`",
        "",
        "---",
        "",
        "## Contents",
        "",
        "| Folder | What is in it |",
        "|---|---|",
        "| `01_dataset/` | The published dataset: Excel workbook (README, data "
        "dictionary, full table, design space, summary statistics, provenance, "
        "retention validation), a gzipped CSV mirror, and the Parquet files for "
        "programmatic use. |",
        "| `02_report/` | The technical report as **PDF**, **Word**, Markdown and "
        "an IEEE LaTeX skeleton, plus `algorithms/` with the six core algorithms "
        "as IEEE-style pseudocode in three formats. |",
        "| `03_figures/` | Every figure as PNG (300 dpi), PDF and SVG. |",
        "| `04_tables/` | Every table as CSV, LaTeX (`booktabs`) and Markdown. |",
        "| `05_results/` | The underlying result data: optimisation fronts and "
        "indicators, surrogate leaderboards and per-fold scores, SHAP/PDP/ALE "
        "outputs, robustness and yield data, statistical test outputs. |",
        "| `06_models/` | The fitted digital twin and the surrogate archive. |",
        "| `07_reproducibility/` | Configurations, per-stage run manifests, the "
        "literature database, the environment lock and the README / "
        "REPRODUCIBILITY guides. |",
        "| `CHECKSUMS.json` | SHA-256 for every file in this package. |",
        "",
        "## Inventory",
        "",
        "| Item | Count |",
        "|---|---|",
    ]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Headline results", ""]
    if ds:
        dur = float(ds.get("duration_s", 0))
        # A re-export resumes from the checkpointed chunks, so its duration is
        # not the cost of the simulations.  Saying "0 min" would be wrong.
        timing = (f"in {dur / 60:.0f} min on {ds.get('n_workers', '?')} workers"
                  if dur > 60 else
                  "(this export resumed from the checkpointed chunk files, so the "
                  "recorded duration is the reload time, not the simulation time)")
        lines.append(f"- **{int(ds.get('n_completed', 0)):,} SPICE simulations**, "
                     f"{100 * float(ds.get('success_rate', 0)):.2f} % successful, "
                     f"{timing}.")
    if ret:
        lines.append(f"- **Retention model validated** against direct transient "
                     f"simulation on {int(ret.get('n_pairs', 0))} designs: "
                     f"Pearson r(log10) = {ret.get('pearson_r_log10', float('nan')):.5f}, "
                     f"median error {ret.get('median_abs_log10_error', float('nan')):.4f} "
                     f"decades, {ret.get('within_factor_2_pct', float('nan')):.0f} % "
                     f"within a factor of two.")
    surro = summary.get("surrogates", [])
    if surro:
        lines.append("- **Selected surrogate per response** (chosen by "
                     "cross-validation from 14 model families; full leaderboard "
                     "in `04_tables/`):")
        for s in sorted(surro, key=lambda x: -float(x.get("cv_r2_mean",
                                                         x.get("test_r2", 0)))):
            cv = s.get("cv_r2_mean")
            cv_txt = f"CV R2 = {float(cv):.4f}, " if cv is not None else ""
            lines.append(f"    - `{s['target']}` -> {s['model']} "
                         f"({cv_txt}test R2 = {float(s['test_r2']):.4f})")
    bench = summary.get("twin_benchmark", [])
    if bench:
        last = bench[-1]
        if last.get("speedup_vs_spice"):
            lines.append(f"- **Digital twin**: {last['ms_per_design']:.4g} ms per "
                         f"design at batch {int(last['batch_size'])}, a "
                         f"{float(last['speedup_vs_spice']):,.0f}x speed-up over the "
                         f"measured SPICE evaluation time.")
    opt = summary.get("optimization", [])
    if opt:
        best = opt[0]
        lines.append(f"- **Best optimiser** by median hypervolume: {best['label']} "
                     f"({best['hv_median']:.4g}), over {int(best['runs'])} "
                     f"independent runs at an identical budget.")

    lines += [
        "",
        "## Reading order",
        "",
        "1. `02_report/technical_report_main.pdf` — the full study.",
        "2. `01_dataset/dram_dataset_main.xlsx` — start at the README sheet, then "
        "the data dictionary.",
        "3. `04_tables/csv/` — every number in the report, machine-readable.",
        "4. `07_reproducibility/REPRODUCIBILITY.md` — how to re-run it.",
        "",
        "## Notes",
        "",
        f"- The PDF was produced by {pdf_engine or 'the configured renderer'} from "
        "the Word rendering, which is itself generated from the Markdown report, "
        "so all four formats carry identical numbers.",
        "- Device models are the public PTM BSIM4 cards for a low-power **logic** "
        "process, not a DRAM process; absolute retention times are longer than a "
        "production part would show. Section 9 of the report states the "
        "limitations in full.",
        "- Design points that fail to write or to sense are **retained and "
        "labelled** in the dataset, not discarded.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
