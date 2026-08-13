# Evaluation case set — `rfq_cases_real_v1.json`

30 real-postcode evaluation cases, built by `evaluation/build_cases.py`. Each case drives
the same production code the live API uses.

**Honest framing** (per the thesis): the property attributes, EPC recommendations, and
site_context (planning constraints) are **real**, pulled from the live EPC and
planning.data.gov.uk APIs. The homeowner **form answers** (contact preference, timeline,
emitter type, roof orientation, and so on) are **assigned** per case, since a postcode
cannot supply them. They are chosen to be valid and varied for coverage.

## Coverage at a glance

- **Technology** (balanced): heat pump 8, solar PV 8, battery 7, solar thermal 7.
- **EPC rating** (all bands A-G): A 2, B 4, C 5, D 8, E 6, F 2, G 2, plus 1 Case B (no rating).
- **EPC source**: normal single-EPC 23, `proxy_nearby` aggregate 3, `proxy_nearby` pick-one 2,
  `proxy_same` Layer-1 1, Case B manual 1. All three proxy phrasings are exercised.
- **Planning constraints**: conservation areas (many), Article 4 (7), AONB (Cotswolds, Cornwall),
  World Heritage Sites (Bath, Lake District), and **6 National Parks** (Lake District,
  North York Moors, New Forest, Dartmoor, Yorkshire Dales, Peak District).
- **Completeness**: 29 cases at 1.00; **REAL_021 is deliberately incomplete (0.93)** so the
  completeness metric can actually fail, and the `missing_fields` path is exercised.
- **Household and supply fields** (heat pump, added Aug 2026): bedrooms 1-4, bathrooms 1-2,
  occupants 1-5, tracking each case's property type rather than repeated. `smart_meter_installed`
  is 5 yes / 3 no. Solar PV `roof_condition` is 6 yes / 2 no; battery `battery_space_available`
  is 6 yes / 1 no (REAL_018, a mid-floor flat, which also carries
  `battery_location_preference: unsure` so the pair stays coherent).

`TestEvalCases` in `test_api.py` guards this file: every case complete except REAL_021, every
form answer a valid option, and no drift from `build_cases.py`. Run `python test_api.py` after
editing either file.

**Not covered (documented limitations, not fixable by adding cases):**
- **Grid headroom** — the UKPN dataset the pipeline queried has been retired, so
  `site_context.grid` is `null` for every postcode.
- **Listed buildings** — the planning check is postcode-centroid based and effectively never
  resolves a listed building, even in historic cores (Bath, Oxford, York, Warwick all return false).

## Per-case breakdown

| Case | Postcode | Tech | EPC source | Rating | Site constraints | Compl. |
|---|---|---|---|---|---|---|
| REAL_001 | E1 6AN | heat_pump | normal | D | conservation | 1.00 |
| REAL_002 | OX7 3EL | solar_pv | normal | E | AONB: Cotswolds | 1.00 |
| REAL_003 | BA2 6AA | solar_thermal | normal | B | conservation, WHS | 1.00 |
| REAL_004 | SE5 8AA | battery | normal | D | conservation | 1.00 |
| REAL_005 | RG9 1AY | heat_pump | normal | E | none | 1.00 |
| REAL_006 | CV12 8UE | solar_pv | normal | C | none | 1.00 |
| REAL_007 | GL54 2BP | heat_pump | proxy_nearby (aggregate) | D | conservation, AONB: Cotswolds | 1.00 |
| REAL_008 | SW1A 2AA | battery | Case B (manual) | - | conservation, Article 4 | 1.00 |
| REAL_009 | LS6 1AA | solar_pv | normal | D | conservation, Article 4 | 1.00 |
| REAL_010 | PL4 7AA | solar_thermal | normal | D | Article 4 | 1.00 |
| REAL_011 | LA22 9SH | heat_pump | normal | **A** | NP: Lake District, conservation, WHS | 1.00 |
| REAL_012 | YO62 5AD | solar_pv | normal | D | NP: North York Moors, conservation | 1.00 |
| REAL_013 | SO43 7BQ | battery | normal | C | NP: New Forest, conservation, Article 4 | 1.00 |
| REAL_014 | TQ13 7TB | solar_thermal | normal | **F** | NP: Dartmoor, conservation | 1.00 |
| REAL_015 | DL8 3RA | heat_pump | normal | **G** | NP: Yorkshire Dales | 1.00 |
| REAL_016 | DE45 1BT | solar_pv | proxy_nearby (aggregate) | C | NP: Peak District, conservation | 1.00 |
| REAL_017 | BA1 1LZ | heat_pump | proxy_nearby (pick one) | C | conservation, WHS | 1.00 |
| REAL_018 | CV34 4BJ | battery | proxy_same (Layer 1) | C | conservation | 1.00 |
| REAL_019 | MK10 9AA | solar_pv | normal | **A** | conservation | 1.00 |
| REAL_020 | NR21 9AA | heat_pump | normal | **G** | conservation | 1.00 |
| REAL_021 | PL15 8AA | solar_thermal | normal | D | conservation | **0.93** (deliberately incomplete) |
| REAL_022 | TR19 7AA | battery | proxy_nearby (aggregate) | E | AONB: Cornwall | 1.00 |
| REAL_023 | OX1 3BG | solar_pv | proxy_nearby (pick one) | F | conservation, Article 4 | 1.00 |
| REAL_024 | YO1 7HH | solar_thermal | normal | E | conservation, Article 4 | 1.00 |
| REAL_025 | CV34 4BJ | solar_thermal | normal | B | conservation | 1.00 |
| REAL_026 | NR21 9AA | battery | normal | D | conservation | 1.00 |
| REAL_027 | YO62 5AD | solar_thermal | normal | E | NP: North York Moors, conservation | 1.00 |
| REAL_028 | SO43 7BQ | solar_pv | normal | B | NP: New Forest, conservation, Article 4 | 1.00 |
| REAL_029 | MK10 9AA | battery | normal | B | conservation | 1.00 |
| REAL_030 | TQ13 7TB | heat_pump | normal | E | NP: Dartmoor, conservation | 1.00 |

Note: REAL_011-030 were added to fill coverage gaps. Six postcodes (CV34, NR21, YO62, SO43,
MK10, TQ13) appear in two cases each, paired with a different technology to broaden
technology coverage. Where both cases of a postcode use a normal EPC, the second is pinned
to a different certificate (via `select_rating`), so all 30 cases are distinct properties.

To rebuild the set: `python -m evaluation.build_cases`. To change or add a case, edit the
`CONFIG` list in `evaluation/build_cases.py` (see the mode comments there).
