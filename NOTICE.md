# Sources and research status

This repository publishes original research code, derived factual macroeconomic
inputs, estimated parameter draws, tables and charts. Original Banque de France
PDFs/images, raw provider responses and the larger local research archive are
not distributed in the public Git tree.

Source attribution, units and release vintages are documented in
[data/sources.json](data/sources.json) and [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
Upstream publishers and data providers retain their respective rights. Public
access to an upstream file does not establish an unrestricted redistribution
license; no license grant over upstream material is asserted here.

The calculations are methodological adaptations of Eco Notepad 454, with
explicitly declared equations and numerical diagnostics. Exact author equations
and coefficients remain unavailable. The estimates are conditional associations
and GDP accounting allocations; they do not establish causal wealth effects or
general-equilibrium GDP responses.

The optional AI/S&P extension requires a locally provided quarterly aggregate
subset of a user-supplied market-cap workbook from the earlier AI wealth study. Its
upstream vendor, constituent classification and redistribution terms are not
independently verified; these inputs are not represented as verified public
provider data. The original workbook, AI/non-AI capitalization observations
and detailed capitalization/ownership ledger are excluded from the published
repository. Derived model contribution tables and charts remain included.
The baseline ECM is reproducible from its published public-data inputs;
rerunning the optional extension requires the locally supplied input. See
[data/spx_ai_split_sources.json](data/spx_ai_split_sources.json). The extension
uses frozen ECM coefficients and an explicit ownership bridge, not an estimated
AI-specific causal response.
