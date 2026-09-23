# Example output

Generated from 150 patents in the Harvard USPTO Patent Dataset with `main_cpc`
starting `H01M` (batteries and fuel cells): 50 filed in 2004, 100 in 2016.
Terms come from titles and abstracts; terms found in a single patent are kept.

- `technical_terms.csv` — 705 technical terms in 9 topics and 30 subtopics,
  kept from 1,236 candidates.
- `rejected_terms.csv` — the 531 candidates left out.
- `secondary_membrane_electrode_assembly.csv` — the 28 technical terms that
  share a patent with `membrane electrode assembly` (row 134, 5 patents).

The topic and subtopic assignments here are illustrative of the format; in
use they are made by the model in the classify step and will differ between
runs.
