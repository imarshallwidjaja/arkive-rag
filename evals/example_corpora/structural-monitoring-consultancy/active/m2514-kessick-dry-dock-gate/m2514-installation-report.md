# No. 2 dry dock gate: vibrating-wire strain gauge installation report

- Our ref: MI-IR-2514
- Client: Port of Kessick
- Client contact: Capt. Ewan Moss
- Author: Tobias Achterberg
- Checked by: Priya Venn
- Approved by: Callum Draycott
- Date: 24 July 2025
- Status: Issue 1, final

## 1. Summary

Marrick Instrumentation installed 10 Geovane VW-4200 vibrating-wire strain
gauges on the caisson gate sill and gate recess walls of No. 2 dry dock gate
from 7 to 9 July 2025. The gauges will monitor the structure during the
refurbishment of the caisson gate seals. Baseline readings were taken on 11
July 2025 and alarm levels are set at 75% (amber) and 100% (red) of the
designer's allowable strain change.

## 2. Scope and instrumented elements

The scope agreed with Capt. Ewan Moss covers the caisson gate sill and gate
recess walls. The gauges are in two groups so that each can carry its own
alarm level.

- Group A: gauges DG-01 to DG-05.
- Group B: gauges DG-06 to DG-10.

## 3. Instruments

| Item | Model | Serial or batch | Notes |
| --- | --- | --- | --- |
| Strain gauge | Geovane VW-4200 | Batch GV-2514-79 | Gauge factor 3.304 |
| Thermistor | Integral, 3 kΩ | - | One per gauge |
| Datalogger | Corlen CR-16 | CR16-0412 | 16 channels |
| Telemetry | 4G modem | MOD-7095 | Hourly upload |

## 4. Installation

The gauges were installed from 7 to 9 July 2025. Mounting blocks were bonded
with two-part epoxy to surfaces ground back to sound material, and the epoxy
cured for 24 hours before the gauges were set. Each gauge has a steel cover.
Cables run in flexible conduit to the logger enclosure, which is on a post
behind the safety fence.

## 5. Thermistor correction and baseline

Each reading pairs a resonant frequency with the gauge's thermistor
temperature. Strain change is calculated as G × (f1² − f0²) × 10⁻³ plus a
thermal term (T1 − T0) × (C1 − C2), where G = 3.304 is the batch gauge factor
from the Geovane certificate and C1 − C2 is the difference between the
expansion coefficients of the gauge wire and the host material. The thermistor
correction is applied to every reading before it is reported.

The reference datum is the set of initial readings taken on 11 July 2025
between 10:00 and 17:00.

## 6. Baseline readings

| Gauge | Location | Initial frequency (Hz) | Thermistor (°C) |
| --- | --- | --- | --- |
| DG-01 | north face | 843.2 | 15.5 |
| DG-02 | south face | 861.5 | 9.3 |
| DG-03 | east face | 815.3 | 11.1 |
| DG-04 | west face | 880.0 | 14.7 |
| DG-05 | soffit | 783.9 | 6.0 |
| DG-06 | top surface | 833.7 | 10.2 |
| DG-07 | upstand | 836.3 | 10.3 |
| DG-08 | toe | 788.5 | 14.9 |
| DG-09 | north face | 853.7 | 7.4 |
| DG-10 | south face | 813.7 | 13.6 |

## 7. Alarm levels

| Gauge group | Designer's allowable change (µε) | Amber, 75% (µε) | Red, 100% (µε) |
| --- | --- | --- | --- |
| A | 280 | 210 | 280 |
| B | 280 | 210 | 280 |

An amber or red reading is handled under the project trigger action response
plan (**TARP**) agreed with the client.

## 8. Data delivery

The logger records hourly and uploads to the Marrick Atlas portal. Alarm
messages go by SMS and email to Capt. Ewan Moss and to the Marrick duty
engineer.

## 9. Limitations

- Strain gauges measure strain change at their own location only.
- Direct sunlight on the exposed face can cause short daily cycles.

## 10. Reliance

This report has been prepared by Marrick Instrumentation Ltd for the sole use
of Port of Kessick in connection with No. 2 dry dock gate. No other party may
rely on it without our written consent. The readings describe the instrumented
elements only, at the times stated, and say nothing about parts of the
structure without instruments.

## Appendix A. Gauge schedule

| Gauge | Group | Location | Orientation | Cable (m) |
| --- | --- | --- | --- | --- |
| DG-01 | A | north face | vertical | 33 |
| DG-02 | A | south face | longitudinal | 10 |
| DG-03 | A | east face | horizontal | 29 |
| DG-04 | A | west face | longitudinal | 46 |
| DG-05 | A | soffit | longitudinal | 31 |
| DG-06 | B | top surface | horizontal | 15 |
| DG-07 | B | upstand | vertical | 20 |
| DG-08 | B | toe | longitudinal | 9 |
| DG-09 | B | north face | longitudinal | 12 |
| DG-10 | B | south face | vertical | 26 |
