# Ro-Ro Ramp 2: vibrating-wire strain gauge installation report

- Our ref: MI-IR-2411
- Client: Port of Kessick
- Client contact: Capt. Ewan Moss
- Author: Tobias Achterberg
- Checked by: Priya Venn
- Approved by: Callum Draycott
- Date: 17 September 2024
- Status: Issue 1, final

## 1. Summary

Marrick Instrumentation installed 8 Geovane VW-4200 vibrating-wire strain
gauges on the hinge-end bearing shelf and ramp soffit of Ro-Ro Ramp 2 from 2
to 3 September 2024. The gauges will monitor the structure during the
replacement of the ramp hinge pins. Baseline readings were taken on 5
September 2024 and alarm levels are set at 75% (amber) and 100% (red) of the
designer's allowable strain change.

## 2. Scope and instrumented elements

The scope agreed with Capt. Ewan Moss covers the hinge-end bearing shelf and
ramp soffit. The gauges are in two groups so that each can carry its own alarm
level.

- Group A: gauges RR-01 to RR-04.
- Group B: gauges RR-05 to RR-08.

## 3. Instruments

| Item | Model | Serial or batch | Notes |
| --- | --- | --- | --- |
| Strain gauge | Geovane VW-4200 | Batch GV-2411-85 | Gauge factor 3.304 |
| Thermistor | Integral, 3 kΩ | - | One per gauge |
| Datalogger | Corlen CR-16 | CR16-0412 | 16 channels |
| Telemetry | 4G modem | MOD-8397 | Hourly upload |

## 4. Installation

The gauges were installed from 2 to 3 September 2024. Mounting blocks were
bonded with two-part epoxy to surfaces ground back to sound material, and the
epoxy cured for 24 hours before the gauges were set. Each gauge has a steel
cover. Cables run in flexible conduit to the logger enclosure, which is bolted
to the parapet.

## 5. Thermistor correction and baseline

Each reading pairs a resonant frequency with the gauge's thermistor
temperature. Strain change is calculated as G × (f1² − f0²) × 10⁻³ plus a
thermal term (T1 − T0) × (C1 − C2), where G = 3.304 is the batch gauge factor
from the Geovane certificate and C1 − C2 is the difference between the
expansion coefficients of the gauge wire and the host material. The thermistor
correction is applied to every reading before it is reported.

The reference datum is the set of initial readings taken on 5 September 2024
between 11:00 and 15:00.

## 6. Baseline readings

| Gauge | Location | Initial frequency (Hz) | Thermistor (°C) |
| --- | --- | --- | --- |
| RR-01 | north face | 803.3 | 10.0 |
| RR-02 | south face | 906.4 | 9.6 |
| RR-03 | east face | 857.5 | 16.8 |
| RR-04 | west face | 780.6 | 6.8 |
| RR-05 | soffit | 843.0 | 8.8 |
| RR-06 | top surface | 906.0 | 10.7 |
| RR-07 | upstand | 802.5 | 7.8 |
| RR-08 | toe | 889.6 | 6.6 |

## 7. Alarm levels

| Gauge group | Designer's allowable change (µε) | Amber, 75% (µε) | Red, 100% (µε) |
| --- | --- | --- | --- |
| A | 240 | 180 | 240 |
| B | 240 | 180 | 240 |

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
of Port of Kessick in connection with Ro-Ro Ramp 2. No other party may rely on
it without our written consent. The readings describe the instrumented
elements only, at the times stated, and say nothing about parts of the
structure without instruments.

## Appendix A. Gauge schedule

| Gauge | Group | Location | Orientation | Cable (m) |
| --- | --- | --- | --- | --- |
| RR-01 | A | north face | horizontal | 21 |
| RR-02 | A | south face | horizontal | 30 |
| RR-03 | A | east face | horizontal | 13 |
| RR-04 | A | west face | vertical | 35 |
| RR-05 | B | soffit | vertical | 30 |
| RR-06 | B | top surface | vertical | 10 |
| RR-07 | B | upstand | longitudinal | 36 |
| RR-08 | B | toe | longitudinal | 30 |
