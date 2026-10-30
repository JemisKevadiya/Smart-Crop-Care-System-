# Report Validation — Smart_Crop_Care_Final_Report_25_Pages.docx

**Date:** 30 October 2026

**Output:**
- `Smart_Crop_Care_Final_Report_25_Pages.docx`, plus a PDF render `Smart_Crop_Care_Final_Report_25_Pages.pdf`.
- Both are in the `Project` folder that contains this repository.
- **Format:** the report is built on the college template `7thSem-Project Report Format.docx`. It uses the template's own page setup, page border, title page, styles (Title, numbered Heading 1–3, Figure and Table Caption, Abstract Content, Table Content), Index field, list fields and reference numbering. The template's placeholders were filled in and its sample chapters replaced.
- **Untouched files:** the template and the earlier reference report were not modified.

## 1. Page count and rendering

- **Final page count: 25 pages** (limit 25), counted by Microsoft Word (`ComputeStatistics`) and in the exported PDF.
- The document was opened in Word, all fields updated (Index, list of tables, list of figures, caption numbers), saved, and exported to PDF.
- Every page was rendered to an image and inspected.
- Problems found and fixed during review:
  - **Index:** it ran onto a second page; its line spacing was tightened.
  - **Acknowledgements:** an empty page appeared after it; two template blank lines were removed.
  - **Figure 4:** it left a gap in Chapter 4 and pushed text onto an extra page; it was made smaller.
  - **Text:** two Chapter 4 paragraphs were shortened with no facts removed.
- Final layout checks:
  - **Headings and tables:** no heading is stranded at the bottom of a page, and every table is kept on one page.
  - **Images and fields:** no broken images, no "Error!" field text, and no template placeholder text left.
- **Page numbering:** the front matter uses Roman numerals I–IX (title page I, number not shown), and the chapters use Arabic numerals 1–16. The page border appears on the front matter only, as in the template.

## 2. Sections included (template order)

| Part | Page |
|---|---|
| Title page | I |
| Index | II |
| Undertaking | III |
| Certificate | IV |
| Acknowledgements (with "Name of Student / Enrollment No" lines for all three students) | V |
| Abstract (with keywords) | VI |
| List of Tables | VII |
| List of Figures | VIII |
| List of Symbols, Abbreviations and Nomenclature | IX |
| 1 Introduction | 1–2 |
| 2 System Design | 3–5 |
| 3 Implementation | 6–10 |
| 4 Testing and Performance Analysis | 11–13 |
| 5 Conclusion and Future Scope | 14 |
| References (24, all cited, numbered [1]–[24] by the template's list style) | 15 |
| Appendix (running the application, configuration settings) | 16 |

The report has 11 tables and 4 figures:
- **Figure 1:** architecture diagram.
- **Figure 2:** workflow diagram.
- **Figure 3:** a genuine screenshot from the real-browser test of the running app.
- **Figure 4:** the EfficientNetB0 confusion matrix from the project's own evaluation.

## 3. Names and details on the front pages

| Item | Value | Source |
|---|---|---|
| Students | Shrey Dhola (ET23BTAI010), Harshav Moradiya (ET23BTAI017), Jemis Kevadiya (ET23BTAI024) | earlier reference report |
| Mentor | **Prof. Dhaval Rana**, Assistant Professor, Department of Artificial Intelligence and Data Science | name as instructed by the user ("only Prof. Dhaval Rana"); designation from the reference report |
| Head of Department | Prof. (Dr.) Mita Parikh, Head, Department of IT and AI-DS | college template |
| Date / academic year | December 2026 / 2026-27 | college template and user |
| Degree | Bachelor of Technology in Artificial Intelligence and Data Science | reference report; the template's placeholder text says "Information Technology" |
| Department and college | Department of Information Technology, Sarvajanik College of Engineering and Technology, Sarvajanik University | template |

Signatures and dates are left blank. None were added.

## 4. Model documented

- **Production model:** EfficientNetB0. `src/disease_detection/model_config.py` selects it by default, `.env` does not set `DISEASE_MODEL`, and the running application showed "Identified from the image by the EfficientNetB0 model".
- **Fallback:** ResNet50.
- **Comparison only:** DenseNet121, which the app cannot select.
- **Results:** all results come from the current runs on one common leakage-free split. The older numbers in the earlier reference report (98.09 %, 97.62 %, 97.94 %, 8,810 and 533 leaves) are not used; the PDF was checked for them.

## 5. Evidence used

- **Code and configuration:** `src/` (disease detection, fertilizer, weather, news, context, chatbot), `app.py`, `app_pages/`, `Dockerfile`, `.env.example`.
- **Models:** `models/<model>/`. All three were loaded: input 224×224×3, 38 outputs, same class order.
- **Evaluation:**
  - `artifacts/reports/<model>/model_metrics.json`, `training_history.json`, `confusion_matrix_test.csv` and `test_probabilities.npy`;
  - `artifacts/reports/model_comparison.csv`;
  - the training notebooks.
- **Split and leakage:** `data/splits_kaggle/split_report.json` and `leakage_check.json`.
- **Tests:**
  - `reports/model_validation.md`, `module_test_results.md`, `end_to_end_test.md` and `fix_log.md`;
  - `artifacts/reports/efficientnetb0/prediction_test.json`, `integration_test.json` and `leaf_scope_check_app.json`.
- **Hardware:** CPU, RAM and Windows version were re-checked on the development PC.

## 6. Metrics in the report and their evidence

| Metric | Value | Split and images | Evidence |
|---|---|---|---|
| EfficientNetB0 accuracy, train / validation / test | 98.84 % / 98.26 % / **98.45 %** | 70,138 / 8,904 / 8,799 | `efficientnetb0/model_metrics.json` |
| EfficientNetB0 macro precision / recall / F1, test | 0.9847 / 0.9844 / 0.9844 | 8,799 test | same |
| ResNet50 / DenseNet121 test accuracy | 98.35 % / 96.95 % | the same 8,799 test leaves | `model_metrics.json` of each model |
| Errors, McNemar p-values | 136 / 145 / 268; p = 0.55 and p < 0.0001 | the same 8,799 test leaves | saved test probabilities |
| Confidence threshold table | 0.8: 3.3 % flagged, 64.9 % errors caught, 99.37 % | 5,430 **validation** leaves | `prediction_test.json` |
| Non-leaf check | 0 / 10,703 real rejected; 33 / 33 non-leaf rejected | held-out leaves | `leaf_scope_check_app.json` |
| App reproduces Kaggle results | 100 % top-1 agreement | 5,273 local test leaves | `model_validation.md` |
| Tests | 178 per model; 12/12 browser stages, local and Docker; 15/15 configuration | — | test reports |

An automated check confirmed that all 21 key metric strings in the PDF match the metric files. The
report states clearly which numbers are test-set results and which are validation results. It makes no
claim about field accuracy.

## 7. Claims that remain unverified

- **Front-page details:** the mentor's designation, the HoD's name and designation, and the enrollment numbers come from the reference report or the template; they could not be checked against the repository.
- **Degree name:** the report keeps "Artificial Intelligence and Data Science", while the template's placeholder sentence reads "Information Technology". Confirm which wording the college requires.
- **Bibliography:** the details were not re-checked online.
- **Untested features:** IP-based location and real Hindi or Gujarati chatbot answers were not tested against the live services; the report says so.
- **Figure 4 size:** the confusion matrix was reduced to fit the page limit, so its class labels are small in print.

## 8. Exclusions and safety

- **Excluded research:** the separate lightweight-CNN research was excluded. The rendered text contains no reference to it.
- **Secrets and paths:** the DOCX and PDF contain no API keys, key patterns or absolute local paths (scanned).
- **Unchanged material:** no application code, dataset or model file was changed to produce the report.
