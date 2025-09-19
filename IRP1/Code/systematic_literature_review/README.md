# Systematic Literature Review (SLR) – Reinforcement Learning in Finance

This repository contains all data, code, and intermediate outputs for the systematic literature review (SLR) on **Reinforcement Learning in Finance**. It follows a transparent and reproducible workflow from database query to final corpus, analysis, and LaTeX tables.


## Repository Structure
```text
systematic_literature_review/
├─ images/                           # Figures and plots used in reports/papers
├─ 240911_scopus_query_results.ris    # Raw Scopus export of initial search
├─ 250602_slr_data_extraction.csv     # Data extraction sheet (key metadata)
├─ 250602_preprocessing.ipynb         # Cleans and deduplicates RIS results
├─ 250602_analysis.ipynb              # Main analysis & descriptive stats
├─ 250711_metadata_enrichment.ipynb   # Adds extra metadata (citations, DOIs)
├─ 250711_final_corpus.bib            # Final set of included papers (BibTeX)
├─ 250714_excluded_papers.ipynb       # Creates table of excluded papers
├─ cleaned_output.csv                 # Cleaned metadata for analysis
├─ enriched_final_corpus.bib          # Final corpus with enriched metadata
├─ excluded_studies_grouped.tex       # LaTeX table: grouped excluded studies
├─ excluded_studies_table.tex         # LaTeX table: individual excluded studies
├─ excluded_studies_table2.tex        # Alternate LaTeX table format
├─ review_523015_excluded_csv_20250714191040.csv  # Detailed exclusion log
├─ requirements.txt                   # Required Python libraries
└─ README.md                          # This file
```


## Workflow Overview

1. **Database Search & Export**  
   - Initial query executed on **Scopus**.  
   - Raw results saved as `240911_scopus_query_results.ris`.

2. **Screening & Data Extraction in Covidence**  
   - All titles/abstracts and full texts screened in **Covidence**.  
   - Inclusion/exclusion decisions and key fields extracted directly within Covidence.  
   - Exported as `review_523015_excluded_csv_20250714191040.csv` and
     `250602_slr_data_extraction.csv`.

3. **Preprocessing in Python**  
   - `250602_preprocessing.ipynb` cleans the Covidence exports and standardizes metadata.  
   - Generates `cleaned_output.csv` and intermediate files.

4. **Analysis & Visualization**  
   - `250602_analysis.ipynb` performs descriptive statistics and visualizations.
   - `250714_exluded_papers.ipynb` prepares LaTeX tables of excluded studies (`excluded_studies_*.tex`).

5. **Metadata Enrichment (Optional)**  
   - `250711_metadata_enrichment.ipynb` adds citation counts, DOIs, or other bibliographic enhancements.  
   - Final enriched corpus stored in `enriched_final_corpus.bib`.

6. **Reporting**  
   - Final corpus provided in `250711_final_corpus.bib`.  
   - Exclusion tables (`excluded_studies_*.tex`, etc.) integrated into the manuscript.

## Reproducibility

Because this project lives as a **subfolder** of a larger repository, the simplest way to get only this part is to **download a ZIP** and extract just this folder.

1. **Download the code**
   - On GitHub, click **Code ▸ Download ZIP** on the main repo page.
   - Unzip it locally and **navigate into** the `systematic_literature_review/` subfolder.

2. **Create and activate a Python environment** (Python ≥3.10 recommended)

   **Using pip + venv**
    ```bash
    cd systematic_literature_review
    python3 -m venv venv
    source venv/bin/activate          # Linux/Mac
    # .\venv\Scripts\activate         # Windows
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
   **Or using conda**
    ```bash
    cd systematic_literature_review
    conda create -n slr-env python=3.11
    conda activate slr-env
    pip install -r requirements.txt
    ```
3. **Launch Jupyter Notebook** (Or other IDE)
    ```bash
    jupyter notebook
    ```


## Acknowledgments

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

![EU Logo](images/eu_funded_logo.jpg)

## License

MIT — see LICENSE.
