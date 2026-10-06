# Format review: related top-journal papers and the conventions adopted

Full texts on publisher sites (nature.com, sciencedirect.com, rsc.org, etc.) were blocked by this session's network policy, so the papers below were reviewed through search-engine summaries and abstracts, combined with the published author guidelines of the target journals. The conventions that the English manuscript follows are listed after the papers.

## Papers reviewed (14)

| # | Paper | Journal | Relevance to this work |
|---|---|---|---|
| 1 | Nowacki et al., Rapid estimation of lithium-ion battery capacity and resistances from short duration current pulses | J. Power Sources 628 (2025) 235813 | Data source; raw-pulse MLP baseline |
| 2 | Nowacki et al., Fine-tuning for rapid capacity estimation of lithium-ion batteries | Energy Storage Mater. 81 (2025) 104425 | Same pulse protocol on LFP and NMC; NMC data used for external replication |
| 3 | Li et al., Health and performance diagnostics in Li-ion batteries with pulse-injection-aided machine learning | Appl. Energy 315 (2022) 119005 | Pulse injection + FNN; Applied Energy structure (highlights, numbered sections) |
| 4 | Tao et al., Generative learning assisted SOH estimation for battery recycling with random retirement conditions | Nat. Commun. 15 (2024) 10154 | Pulse SOH at scale; Nature-style figure/caption conventions |
| 5 | Tao et al., Rapid and sustainable battery health diagnosis using fast pulse test and random forest | J. Power Sources 597 (2024) 234156 | Pulse features + tree ensemble |
| 6 | Ran et al., Fast remaining capacity estimation based on short-time pulse test and GPR | Energy Environ. Mater. 6 (2023) e12386 | Pulse features + GPR |
| 7 | Zhu et al., Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation | Nat. Commun. 13 (2022) 2261 | Feature-based capacity estimation; external-dataset validation |
| 8 | Roman et al., Machine learning pipeline for battery state-of-health estimation | Nat. Mach. Intell. 3 (2021) 447 | Feature engineering + automatic selection + calibrated uncertainty |
| 9 | Lu et al., Deep learning to estimate battery SOH without additional degradation experiments | Nat. Commun. 14 (2023) 2760 | Cross-manufacturer generalisation reporting |
| 10 | Wang et al., Physics-informed neural network for battery degradation stable modeling and prognosis | Nat. Commun. 15 (2024) 4332 | Monotonic/physics priors; multi-dataset validation |
| 11 | Che et al., Diagnostic-free onboard battery health assessment | Joule 9 (2025) 102010 | Onboard deployment framing |
| 12 | Liu et al., Multi-modal framework for battery SOH evaluation using open-source EV data | Nat. Commun. 16 (2025) 1137 | Multi-branch fusion |
| 13 | Guo et al., Uncovering the impact of battery design parameters on health and lifetime using short charging segments | Energy Environ. Sci. 18 (2025) 8462 | Short-segment diagnostics |
| 14 | Severson et al., Data-driven prediction of battery cycle life before capacity degradation | Nat. Energy 4 (2019) 383 | Handcrafted features + regularised regression; benchmark of the field |

## Conventions adopted in the manuscript

- **Target style:** Elsevier energy journals (Applied Energy / J. Power Sources / eTransportation), single column as requested.
- **Front matter:** title; authors with superscript affiliation letters; asterisk for the corresponding author; 3–5 highlights of at most 85 characters each; one-paragraph abstract of about 250 words or fewer; 6 keywords separated by semicolons.
- **Sections:** numbered "1. Introduction", "2.1.", "2.3.1."; first-level headings bold, second-level bold, third-level italic.
- **Equations:** native Word equations (editable), numbered (1), (2), … right-aligned; cited as "Eq. (n)". Variables italic, units and operators upright.
- **Figures:** cited as "Fig. n"; caption below the figure, starting with bold "Fig. n."; a title sentence, then panel descriptions (a), (b), … Panel letters in the images are bold lowercase (Nature-family style, also common in Elsevier). Error bars, n and tests are defined in each caption.
- **Tables:** caption above ("Table n." bold); three-line (booktabs) rules; table notes below the table with superscript letters ^a^, ^b^.
- **Numbers and units:** thousands separators (16,938); en dashes for ranges (1.247–1.513); true minus signs (−0.123); SOH errors in percentage points (pp); exact *p* values with Holm correction stated.
- **References:** numbered in order of first citation, in square brackets [1], [2–4]; Elsevier numbered style with abbreviated journal names and DOIs.
- **Back matter:** CRediT authorship contribution statement, Declaration of competing interest, Data availability, Acknowledgements, Appendix A. Supplementary data (separate Word file).
- **Statistics reporting** (Nature Portfolio reporting standard): unit of analysis (64 cells), paired tests, intervals, multiplicity correction and post hoc analyses are stated explicitly.
