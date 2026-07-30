#!/usr/bin/env python3
"""
figures/mcnemar_table.py

Per-seed McNemar's test comparison table (ViT vs. ResNet-50), from
master_summary.json's "mcnemar_per_seed" list (the same structure
src/code.py::run_comparison() already writes into comparison_summary.json).
Produces a rendered table figure (PNG) and a markdown table.

Usage:
    python figures/mcnemar_table.py --master-summary results/master_summary.json --out-dir figures/output

    # Test against mock fixtures:
    python figures/mcnemar_table.py --master-summary figures/fixtures/master_summary.json --out-dir figures/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MissingDataError, ensure_output_dir, load_master_summary, set_style,
    stamp_provenance,
)


def render_table_figure(rows: list, out_dir: Path, mock: bool) -> Path:
    set_style()
    col_labels = ["Seed", "ViT acc.", "ResNet-50 acc.", "χ²", "p-value", "Direction", "p < 0.05"]
    table_data = []
    for r in rows:
        table_data.append([
            str(r["seed"]),
            f"{r['vit_acc']:.4f}",
            f"{r['rn_acc']:.4f}",
            f"{r['chi2']:.4f}",
            f"{r['p_value']:.3e}" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}",
            r["direction"].replace("_", " "),
            "yes" if r["significant"] else "no",
        ])

    n_sig = sum(r["significant"] for r in rows)
    dirs_sig = [r["direction"] for r in rows if r["significant"]]
    consistent = len(set(dirs_sig)) <= 1
    dominant = max(set(dirs_sig), key=dirs_sig.count).replace("_", " ") if dirs_sig else "n/a"
    summary_line = (
        f"{n_sig}/{len(rows)} seeds significant (p<0.05); "
        f"dominant direction: {dominant}; "
        f"{'consistent' if consistent else 'INCONSISTENT'} across significant seeds"
    )

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 2))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4C72B0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(rows, start=1):
        if r["significant"]:
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor("#EAF2E3")

    ax.set_title("McNemar's test — ViT vs. ResNet-50, per seed", pad=20, fontsize=12)
    fig.text(0.5, 0.02, summary_line, ha="center", fontsize=9, style="italic")
    if mock:
        stamp_provenance(fig, [{"_mock": True}])

    out_path = out_dir / "mcnemar_table.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_markdown(rows: list, out_dir: Path, mock: bool) -> Path:
    lines = ["# McNemar's test — ViT vs. ResNet-50, per seed", ""]
    if mock:
        lines += ["> **MOCK fixture data — do not cite.**", ""]
    lines += [
        "| Seed | ViT acc. | ResNet-50 acc. | χ² | p-value | Direction | p < 0.05 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pstr = f"{r['p_value']:.3e}" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        lines.append(
            f"| {r['seed']} | {r['vit_acc']:.4f} | {r['rn_acc']:.4f} | {r['chi2']:.4f} | "
            f"{pstr} | {r['direction'].replace('_', ' ')} | {'yes' if r['significant'] else 'no'} |"
        )

    n_sig = sum(r["significant"] for r in rows)
    dirs_sig = [r["direction"] for r in rows if r["significant"]]
    consistent = len(set(dirs_sig)) <= 1
    dominant = max(set(dirs_sig), key=dirs_sig.count).replace("_", " ") if dirs_sig else "n/a"
    lines += [
        "",
        f"**Summary:** {n_sig}/{len(rows)} seeds significant (p<0.05); "
        f"dominant direction: {dominant}; "
        f"{'consistent' if consistent else 'INCONSISTENT'} across significant seeds.",
    ]

    out_path = out_dir / "mcnemar_table.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-summary", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)

    try:
        master = load_master_summary(args.master_summary)
    except MissingDataError as exc:
        print(str(exc))
        sys.exit(1)

    rows = master.get("mcnemar_per_seed", [])
    if not rows:
        print("[PLACEHOLDER: awaiting results/master_summary.json 'mcnemar_per_seed'] - "
              "master summary loaded but contains no per-seed McNemar results.")
        sys.exit(1)

    mock = bool(master.get("_mock"))
    png_path = render_table_figure(rows, out_dir, mock)
    md_path = render_markdown(rows, out_dir, mock)
    print(f"Wrote {png_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
