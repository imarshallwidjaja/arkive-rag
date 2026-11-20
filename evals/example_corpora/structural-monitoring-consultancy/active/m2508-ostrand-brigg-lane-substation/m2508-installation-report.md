# Brigg Lane 132 kV substation: vibrating-wire strain gauge installation report

- Our ref: MI-IR-2508
- Client: Ostrand Energy Networks
- Client contact: Mr Idris Fell
- Author: Tobias Achterberg
- Checked by: Priya Venn
- Approved by: Callum Draycott
- Date: 28 May 2025
- Status: Issue 1, final

## 1. Summary

Marrick Instrumentation installed 10 Geovane VW-4200 vibrating-wire strain
gauges on the cable basement wall and transformer T1 plinth of Brigg Lane 132
kV substation from 12 to 13 May 2025. The gauges will monitor the structure
during the sheet piling for the neighbouring retail park drainage. Baseline
readings were taken on 15 May 2025 and alarm levels are set at 75% (amber) and
100% (red) of the designer's allowable strain change.

## 2. Scope and instrumented elements

The scope agreed with Mr Idris Fell covers the cable basement wall and
transformer T1 plinth. The gauges are in two groups so that each can carry its
own alarm level.

- Group A: gauges BL-01 to BL-05.
- Group B: gauges BL-06 to BL-10.

## 3. Instruments

| Item | Model | Serial or batch | Notes |
| --- | --- | --- | --- |
| Strain gauge | Geovane VW-4200 | Batch GV-2508-63 | Gauge factor 3.304 |
| Thermistor | Integral, 3 kΩ | - | One per gauge |
| Datalogger | Corlen CR-16 | CR16-0519 | 16 channels |
| Telemetry | 4G modem | MOD-7814 | Hourly upload |

## 4. Installation

The gauges were installed from 12 to 13 May 2025. Mounting blocks were bonded
with two-part epoxy to surfaces ground back to sound material, and the epoxy
cured for 24 hours before the gauges were set. Each gauge has a steel cover.
Cables run in flexible conduit to the logger enclosure, which is fixed to the
wall at the access stair.

## 5. Thermistor correction and baseline

Each reading pairs a resonant frequency with the gauge's thermistor
temperature. Strain change is calculated as G × (f1² − f0²) × 10⁻³ plus a
thermal term (T1 − T0) × (C1 − C2), where G = 3.304 is the batch gauge factor
from the Geovane certificate and C1 − C2 is the difference between the
expansion coefficients of the gauge wire and the host material. The thermistor
correction is applied to every reading before it is reported.

The reference datum is the set of initial readings taken on 15 May 2025
between 14:00 and 17:00.

## 6. Baseline readings

| Gauge | Location | Initial frequency (Hz) | Thermistor (°C) |
| --- | --- | --- | --- |
| BL-01 | north face | 899.7 | 16.7 |
| BL-02 | south face | 837.8 | 7.6 |
| BL-03 | east face | 905.5 | 16.3 |
| BL-04 | west face | 878.7 | 7.9 |
| BL-05 | soffit | 872.1 | 9.9 |
| BL-06 | top surface | 870.1 | 6.7 |
| BL-07 | upstand | 849.7 | 16.4 |
| BL-08 | toe | 884.3 | 15.6 |
| BL-09 | north face | 869.5 | 15.9 |
| BL-10 | south face | 844.3 | 6.7 |

## 7. Alarm levels

| Gauge group | Designer's allowable change (µε) | Amber, 75% (µε) | Red, 100% (µε) |
| --- | --- | --- | --- |
| A | 190 | 142 | 190 |
| B | 190 | 142 | 190 |

An amber or red reading is handled under the project trigger action response
plan (**TARP**) agreed with the client.

## 8. Data delivery

The logger records hourly and uploads to the Marrick Atlas portal. Alarm
messages go by SMS and email to Mr Idris Fell and to the Marrick duty
engineer.

## 9. Limitations

- Strain gauges measure strain change at their own location only.
- Readings in the first two weeks may include settling of the mounting blocks.

## 10. Reliance

This report has been prepared by Marrick Instrumentation Ltd for the sole use
of Ostrand Energy Networks in connection with Brigg Lane 132 kV substation. No
other party may rely on it without our written consent. The readings describe
the instrumented elements only, at the times stated, and say nothing about
parts of the structure without instruments.

## Appendix A. Gauge schedule

| Gauge | Group | Location | Orientation | Cable (m) |
| --- | --- | --- | --- | --- |
| BL-01 | A | north face | horizontal | 19 |
| BL-02 | A | south face | horizontal | 12 |
| BL-03 | A | east face | horizontal | 30 |
| BL-04 | A | west face | vertical | 46 |
| BL-05 | A | soffit | horizontal | 17 |
| BL-06 | B | top surface | horizontal | 14 |
| BL-07 | B | upstand | vertical | 10 |
| BL-08 | B | toe | horizontal | 21 |
| BL-09 | B | north face | horizontal | 39 |
| BL-10 | B | south face | longitudinal | 27 |
