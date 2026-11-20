# Wendle Tunnel: vibrating-wire strain gauge installation report

- Our ref: MI-IR-2503
- Client: Northbray Rail Infrastructure
- Client contact: Ms Opal Stanridge
- Author: Priya Venn
- Checked by: Callum Draycott
- Date: 6 March 2025
- Status: Issue 1, final

## 1. Summary

Marrick Instrumentation installed 14 Geovane VW-4200 vibrating-wire strain
gauges on the brick-lined north portal and the first 40 m of lining of Wendle
Tunnel from 17 to 20 February 2025. The gauges will monitor the structure
during the track lowering through the tunnel for overhead line clearance.
Baseline readings were taken on 22 February 2025 and alarm levels are set at
75% (amber) and 100% (red) of the designer's allowable strain change.

## 2. Scope and instrumented elements

The scope agreed with Ms Opal Stanridge covers the brick-lined north portal
and the first 40 m of lining. The gauges are in two groups so that each can
carry its own alarm level.

- Group A: gauges WT-01 to WT-07.
- Group B: gauges WT-08 to WT-14.

## 3. Instruments

| Item | Model | Serial or batch | Notes |
| --- | --- | --- | --- |
| Strain gauge | Geovane VW-4200 | Batch GV-2503-96 | Gauge factor 3.304 |
| Thermistor | Integral, 3 kΩ | - | One per gauge |
| Datalogger | Corlen CR-16 | CR16-0533 | 16 channels |
| Telemetry | 4G modem | MOD-3547 | Hourly upload |

## 4. Installation

The gauges were installed from 17 to 20 February 2025. Mounting blocks were
bonded with two-part epoxy to surfaces ground back to sound material, and the
epoxy cured for 24 hours before the gauges were set. Each gauge has a steel
cover. Cables run in flexible conduit to the logger enclosure, which is fixed
to the wall at the access stair.

## 5. Thermistor correction and baseline

Each reading pairs a resonant frequency with the gauge's thermistor
temperature. Strain change is calculated as G × (f1² − f0²) × 10⁻³ plus a
thermal term (T1 − T0) × (C1 − C2), where G = 3.304 is the batch gauge factor
from the Geovane certificate and C1 − C2 is the difference between the
expansion coefficients of the gauge wire and the host material. The thermistor
correction is applied to every reading before it is reported.

The reference datum is the set of initial readings taken on 22 February 2025
between 14:00 and 15:00.

## 6. Baseline readings

| Gauge | Location | Initial frequency (Hz) | Thermistor (°C) |
| --- | --- | --- | --- |
| WT-01 | crown | 844.7 | 8.6 |
| WT-02 | left haunch | 787.2 | 12.2 |
| WT-03 | right haunch | 893.4 | 6.6 |
| WT-04 | left springing | 836.4 | 10.9 |
| WT-05 | right springing | 859.9 | 9.1 |
| WT-06 | spandrel | 848.4 | 12.9 |
| WT-07 | crown | 796.1 | 15.9 |
| WT-08 | portal face | 792.7 | 9.5 |
| WT-09 | crown | 868.5 | 8.6 |
| WT-10 | left haunch | 832.0 | 7.3 |
| WT-11 | right haunch | 810.6 | 10.4 |
| WT-12 | left springing | 782.5 | 13.0 |
| WT-13 | right springing | 887.4 | 15.5 |
| WT-14 | spandrel | 825.6 | 6.2 |

## 7. Alarm levels

| Gauge group | Designer's allowable change (µε) | Amber, 75% (µε) | Red, 100% (µε) |
| --- | --- | --- | --- |
| A | 230 | 172 | 230 |
| B | 230 | 172 | 230 |

An amber or red reading is handled under the project trigger action response
plan (**TARP**) agreed with the client.

## 8. Data delivery

The logger records hourly and uploads to the Marrick Atlas portal. Alarm
messages go by SMS and email to Ms Opal Stanridge and to the Marrick duty
engineer.

## 9. Limitations

- Strain gauges measure strain change at their own location only.
- Readings in the first two weeks may include settling of the mounting blocks.

## 10. Reliance

This report has been prepared by Marrick Instrumentation Ltd for the sole use
of Northbray Rail Infrastructure in connection with Wendle Tunnel. No other
party may rely on it without our written consent. The readings describe the
instrumented elements only, at the times stated, and say nothing about parts
of the structure without instruments.

## Appendix A. Gauge schedule

| Gauge | Group | Location | Orientation | Cable (m) |
| --- | --- | --- | --- | --- |
| WT-01 | A | crown | longitudinal | 10 |
| WT-02 | A | left haunch | longitudinal | 12 |
| WT-03 | A | right haunch | vertical | 28 |
| WT-04 | A | left springing | longitudinal | 10 |
| WT-05 | A | right springing | vertical | 40 |
| WT-06 | A | spandrel | horizontal | 11 |
| WT-07 | A | crown | horizontal | 20 |
| WT-08 | B | portal face | horizontal | 10 |
| WT-09 | B | crown | horizontal | 37 |
| WT-10 | B | left haunch | horizontal | 43 |
| WT-11 | B | right haunch | longitudinal | 12 |
| WT-12 | B | left springing | longitudinal | 25 |
| WT-13 | B | right springing | longitudinal | 29 |
| WT-14 | B | spandrel | vertical | 22 |
