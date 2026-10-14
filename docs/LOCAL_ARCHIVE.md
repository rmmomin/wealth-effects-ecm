# Local research archive and public release

The ECM calculations were developed in the separate local
`broad_wealth_effects` research working tree. The original wealth-effects
repository remains a separate project. The public `wealth-effects-ecm`
release curates the portable quarterly ECM calculation, its derived data,
tests and showcase results from that working tree.

The earlier broad asset accounting, ARDL estimates, HANK/SSJ/IKC benchmarks,
Banque de France survey-methodology examples, provider caches and dated
publication outputs are preserved locally at their existing paths. The
previous root README is also preserved in the local research archive. Those
materials are outside the public release; a public clone does not contain or
promise the full local archive.

The public calculation has no runtime dependency on the original private
source checkout. It includes the derived observations needed to reproduce
the ECM tables and figures offline. Raw provider archives, author papers,
credentials, machine-specific logs and large intermediate draw histories are
excluded from the public Git history.

This curation changes the public entry point and presentation. It does not
delete or overwrite earlier estimator outputs, replace the original
wealth-effects workflow, or reclassify illustrative structural calculations
as empirical ECM estimates.
