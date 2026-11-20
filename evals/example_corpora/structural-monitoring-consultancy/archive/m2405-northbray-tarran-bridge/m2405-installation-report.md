# Tarran Bridge: vibrating-wire strain gauge installation report

- Our ref: MI-IR-2405
- Client: Northbray Rail Infrastructure
- Client contact: Mr Jarrow Keld
- Author: Priya Venn
- Checked by: Callum Draycott
- Date: 25 April 2024
- Status: Issue 1, final

## 1. Summary

Marrick Instrumentation installed 12 Geovane VW-4200 vibrating-wire strain
gauges on the steel half-through main girders of span 2 of Tarran Bridge from
8 to 10 April 2024. The gauges will monitor the structure during the deck
waterproofing and ballast renewal. Baseline readings were taken on 12 April
2024 and alarm levels are set at 75% (amber) and 100% (red) of the designer's
allowable strain change.

## 2. Scope and instrumented elements

The scope agreed with Mr Jarrow Keld covers the steel half-through main
girders of span 2. The gauges are in two groups so that each can carry its own
alarm level.

- Group A: gauges TB-01 to TB-06.
- Group B: gauges TB-07 to TB-12.

## 3. Instruments

| Item | Model | Serial or batch | Notes |
| --- | --- | --- | --- |
| Strain gauge | Geovane VW-4200 | Batch GV-2405-40 | Gauge factor 3.304 |
| Thermistor | Integral, 3 kΩ | - | One per gauge |
| Datalogger | Corlen CR-16 | CR16-0519 | 16 channels |
| Telemetry | 4G modem | MOD-2165 | Hourly upload |

## 4. Installation

The gauges were installed from 8 to 10 April 2024. Mounting blocks were arc
welded to the cleaned steel and each weld was inspected before the gauge was
tensioned. Each gauge has a steel cover. Cables run in flexible conduit to the
logger enclosure, which is on a post behind the safety fence.

## 5. Thermistor correction and baseline

Each reading pairs a resonant frequency with the gauge's thermistor
temperature. Strain change is calculated as G × (f1² − f0²) × 10⁻³ plus a
thermal term (T1 − T0) × (C1 − C2), where G = 3.304 is the batch gauge factor
from the Geovane certificate and C1 − C2 is the difference between the
expansion coefficients of the gauge wire and the host material. The thermistor
correction is applied to every reading before it is reported.

The reference datum is the set of initial readings taken on 12 April 2024
between 14:00 and 15:00.

## 6. Baseline readings

| Gauge | Location | Initial frequency (Hz) | Thermistor (°C) |
| --- | --- | --- | --- |
| TB-01 | top flange | 866.6 | 10.8 |
| TB-02 | bottom flange | 830.8 | 16.6 |
| TB-03 | web, mid-depth | 818.8 | 16.1 |
| TB-04 | stiffener | 804.3 | 11.1 |
| TB-05 | top flange | 918.5 | 6.5 |
| TB-06 | bottom flange | 919.0 | 8.5 |
| TB-07 | web, upper third | 866.0 | 9.7 |
| TB-08 | web, lower third | 826.4 | 8.6 |
| TB-09 | top flange | 823.7 | 14.4 |
| TB-10 | bottom flange | 872.8 | 7.3 |
| TB-11 | web, mid-depth | 809.9 | 10.4 |
| TB-12 | stiffener | 866.3 | 6.4 |

## 7. Alarm levels

| Gauge group | Designer's allowable change (µε) | Amber, 75% (µε) | Red, 100% (µε) |
| --- | --- | --- | --- |
| A | 260 | 195 | 260 |
| B | 260 | 195 | 260 |

An amber or red reading is handled under the project trigger action response
plan (**TARP**) agreed with the client.

## 8. Data delivery

The logger records hourly and uploads to the Marrick Atlas portal. Alarm
messages go by SMS and email to Mr Jarrow Keld and to the Marrick duty
engineer.

## 9. Limitations

- Strain gauges measure strain change at their own location only.
- Readings in the first two weeks may include settling of the mounting blocks.

## 10. Reliance

This report has been prepared by Marrick Instrumentation Ltd for the sole use
of Northbray Rail Infrastructure in connection with Tarran Bridge. No other
party may rely on it without our written consent. The readings describe the
instrumented elements only, at the times stated, and say nothing about parts
of the structure without instruments.

## Appendix A. Gauge schedule

| Gauge | Group | Location | Orientation | Cable (m) |
| --- | --- | --- | --- | --- |
| TB-01 | A | top flange | longitudinal | 16 |
| TB-02 | A | bottom flange | vertical | 10 |
| TB-03 | A | web, mid-depth | longitudinal | 43 |
| TB-04 | A | stiffener | longitudinal | 35 |
| TB-05 | A | top flange | horizontal | 11 |
| TB-06 | A | bottom flange | vertical | 44 |
| TB-07 | B | web, upper third | vertical | 36 |
| TB-08 | B | web, lower third | longitudinal | 23 |
| TB-09 | B | top flange | longitudinal | 25 |
| TB-10 | B | bottom flange | longitudinal | 32 |
| TB-11 | B | web, mid-depth | longitudinal | 30 |
| TB-12 | B | stiffener | vertical | 46 |
