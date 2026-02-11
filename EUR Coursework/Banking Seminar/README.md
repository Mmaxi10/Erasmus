Banking Seminar - quick notes

This folder contains the full workflow for the seminar project: raw FFIEC data, data processing scripts, and the empirical model notebook.

What is where
- DATA/
  - FFIEC (raw): original tab-delimited FFIEC files by year
  - FFIEC (csv): yearly formatted CSV files created from raw files
  - ffiec_combined.csv: combined cleaned panel used by the model
  - DFF.csv: Fed Funds Rate series
- Data Processing/
  - DataFormatting.py: converts raw yearly FFIEC txt files into yearly CSV files
  - DataCleaning.py: cleans yearly CSVs and builds one combined dataset
  - RunDataPipeline.py: runs formatting first, then cleaning
- Empirical Model/
  - Empirical_Model.ipynb: main notebook with baseline, extended, full model, and robustness checks

How to run/replicate
1) Install dependencies listed in this folder's `requirements.txt`

2) Run `RunDataPipeline.py` from `Data Processing/`.

3) Open and run `Empirical_Model.ipynb` in `Empirical Model/`.

Notes
- The model is panel fixed-effects with bank and time FE.
- Standard errors are clustered by bank.
- NIM and policy rate changes are in basis points in the notebook.
- The notebook exports paper tables to:
  output/spreadsheet/
