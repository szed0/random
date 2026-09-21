# Example output

Generated from 140 real patents (Harvard USPTO Patent Dataset, `main_cpc`
starting `H01M` — electrochemical energy — abstracts over 300 characters).

- `technical_terms.csv` — step 1 over the whole corpus, 360 terms.
- `subterms_nonaqueous_electrolyte.csv` — step 2, scoped to the 17 patents
  carrying `nonaqueous electrolyte`.

Both are here to show the column shape and the kind of term that survives the
filters. The corpus is single-year, so it exercises extraction and says
nothing about trends.
