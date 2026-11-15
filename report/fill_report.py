"""Fill the Lab 6 report template with the experiment results and TensorBoard screenshots.

Run from the repository root:  python report/fill_report.py
Reads results.jsonl and figures/*.png, writes report/Lab_6_-_Deep_Learning_Lab_Report.docx.
"""

import copy
import json
from pathlib import Path

import docx
from docx.shared import Inches

import answers

TEMPLATE = Path("report/template.docx")
OUTPUT = Path("report/Lab_6_-_Deep_Learning_Lab_Report.docx")
FIGURES = Path("figures")
IMAGE_WIDTH = Inches(6.0)


def load_results(path: Path = Path("results.jsonl")) -> dict[str, dict]:
    with path.open() as f:
        return {r["name"]: r for r in map(json.loads, f)}


def pct(result: dict) -> str:
    return f"{100 * result['test_acc']:.2f}%"


def seconds(result: dict) -> str:
    return f"{result['train_time_s']:.1f} s"


def strip_empty_paragraphs(cell) -> None:
    for p in cell.paragraphs:
        if not p.text.strip() and not p._p.xpath(".//w:drawing"):
            p._p.getparent().remove(p._p)


def append_inline(cell, text: str) -> None:
    """Append text to the last labelled paragraph of a cell (e.g. 'Testing Accuracy: 68.49%')."""
    labelled = [p for p in cell.paragraphs if p.text.strip()]
    run = labelled[-1].add_run(f" {text}")
    run.bold = True


def fill_block(cell, paragraphs: list[str]) -> None:
    """Replace a cell's blank placeholder lines with answer paragraphs below its label."""
    strip_empty_paragraphs(cell)
    for text in paragraphs:
        cell.add_paragraph(text)


def fill_image(cell, image: Path) -> None:
    strip_empty_paragraphs(cell)
    cell.add_paragraph().add_run().add_picture(str(image), width=IMAGE_WIDTH)


def set_text(cell, text: str) -> None:
    strip_empty_paragraphs(cell)
    if not cell.paragraphs:
        cell.add_paragraph()
    paragraph = cell.paragraphs[0]
    for run in paragraph.runs:
        run._r.getparent().remove(run._r)
    paragraph.add_run(text)


def fill_rows(table, rows: list[list[str]], first_row: int = 1) -> None:
    """Write data rows starting at first_row, cloning the last data row when more are needed.

    The final row of each results table is the merged question-response row, so data rows
    occupy [first_row, len(table.rows) - 1).
    """
    question_tr = table.rows[-1]._tr
    template_tr = table.rows[first_row]._tr
    while len(table.rows) - 1 - first_row < len(rows):
        question_tr.addprevious(copy.deepcopy(template_tr))
    for row, values in zip(table.rows[first_row:], rows):
        for cell, value in zip(row.cells, values):
            set_text(cell, value)


def fill_part1(doc, results: dict) -> None:
    table = doc.tables[0]
    baseline = results["part1_baseline"]
    append_inline(table.rows[0].cells[0], pct(baseline))
    append_inline(table.rows[0].cells[1], f"{seconds(baseline)} (5 epochs, CPU)")
    fill_image(table.rows[1].cells[0], FIGURES / "part1_train_loss.png")
    fill_image(table.rows[2].cells[0], FIGURES / "part1_val_acc.png")


def fill_part2(doc, results: dict) -> None:
    table = doc.tables[1]
    fill_block(table.rows[0].cells[0], answers.part2_changes(results))
    append_inline(table.rows[1].cells[0], answers.part2_accuracy(results))
    fill_block(table.rows[2].cells[0], answers.part2_question(results))


def fill_part3(doc, results: dict) -> None:
    table = doc.tables[2]
    rows = [[label, pct(results[name]), seconds(results[name])] for label, name in answers.PART3_RUNS]
    fill_rows(table, rows)
    fill_block(table.rows[-1].cells[0], answers.part3_question(results))


def fill_part4(doc, results: dict) -> None:
    table = doc.tables[3]
    rows = [[label, pct(results[name]), seconds(results[name])] for label, name in answers.PART4_RUNS]
    fill_rows(table, rows)
    fill_block(table.rows[-1].cells[0], answers.part4_question(results))


def fill_part5(doc, results: dict) -> None:
    table = doc.tables[4]
    fill_image(table.rows[0].cells[0], FIGURES / "part5_train_loss.png")
    fill_image(table.rows[1].cells[0], FIGURES / "part5_val_acc.png")
    fill_block(table.rows[2].cells[0], answers.part5_question(results))


def fill_part6(doc, results: dict) -> None:
    table = doc.tables[5]
    rows = [[label, pct(results[name]), seconds(results[name])] for label, name in answers.PART6_RUNS]
    fill_rows(table, rows)
    fill_block(table.rows[-1].cells[0], answers.part6_question(results))


def fill_part7(doc, results: dict) -> None:
    table = doc.tables[6]
    final = results[answers.FINAL_RUN]
    for cell, value in zip(table.rows[1].cells, [*answers.part7_config(final), pct(final)]):
        set_text(cell, value)
    bonus = next(p for p in doc.paragraphs if p.text.startswith("As a fun bonus"))
    for paragraph in answers.part7_notes(results):
        bonus.insert_paragraph_before(paragraph)


def main() -> None:
    results = load_results()
    doc = docx.Document(str(TEMPLATE))
    for fill in (fill_part1, fill_part2, fill_part3, fill_part4, fill_part5, fill_part6, fill_part7):
        fill(doc, results)
    doc.save(str(OUTPUT))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
