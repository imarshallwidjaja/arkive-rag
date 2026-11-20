# Greyscar Dam spillway: fibre Bragg grating installation and baseline report

- Our ref: MI-IR-2505
- Client: Halverton Water Authority
- Client contact: Ms Rowan Peake
- Author: Maren Lisk
- Checked by: Priya Venn
- Approved by: Callum Draycott
- Date: 8 July 2025
- Status: Final v1.0

## 1. Summary

Marrick Instrumentation installed 3 Lumenstrand LS-8 fibre Bragg grating
arrays, 24 gratings in all, on the spillway chute walls and the training wall
at the stilling basin of Greyscar Dam spillway from 3 to 5 June 2025. The
arrays will monitor strain during the replacement of the spillway chute
joints. The baseline was logged from 7 June 2025 12:00 to 10 June 2025 12:00.
Alarm levels are set at 60% (amber) and 85% (red) of the designer's allowable
strain change.

## 2. Scope and instrumented elements

The scope agreed with Ms Rowan Peake covers the spillway chute walls and the
training wall at the stilling basin. Each array runs along one element, and
its seven bonded gratings are spaced to cover the zone the designer expects to
carry the largest change.

| Array | Element | Bonded gratings | SFR grating |
| --- | --- | --- | --- |
| G1 | spillway chute walls, position 1 | 7 | G1-8 |
| G2 | spillway chute walls, position 2 | 7 | G2-8 |
| G3 | spillway chute walls, position 3 | 7 | G3-8 |

## 3. Instruments

| Item | Model | Serial | Notes |
| --- | --- | --- | --- |
| FBG array | Lumenstrand LS-8 | LS8-2505-01 to -03 | 8 gratings per array |
| Interrogator | Prisma FI-4 | PF4-1876 | 4 channels, 1 Hz |
| Adhesive | Two-part methacrylate | - | Cure 24 h at 10 °C |
| Telemetry | 4G router | - | Hourly upload |

## 4. Installation

The arrays were installed from 3 to 5 June 2025. Bond zones were ground to
sound material, cleaned with solvent and primed. Each bonded grating was fixed
in methacrylate adhesive under a clamp and the SFR grating was left free in
its loose tube. Fibre leads run in armoured conduit to the interrogator
cabinet.

## 5. Temperature compensation and baseline

Each Lumenstrand LS-8 array carries eight gratings on one fibre. Seven are
bonded to the structure and respond to both strain and temperature. The
eighth, the strain-free reference (**SFR**) grating, sits in a loose tube
beside them and responds to temperature only. We convert each bonded grating's
wavelength shift to strain after subtracting the SFR shift, using the SFR
temperature sensitivity of 9.7 pm/°C from the array calibration certificate
and a strain sensitivity of 1.21 pm/µε.

The baseline for each grating is the mean of the readings logged over 72 hours
of stable conditions after the adhesive had cured. Table 3 states the baseline
window with its start and end times and the number of readings averaged. Every
later strain value is a change from that baseline, after temperature
compensation.

| Baseline window start | Baseline window end | Readings averaged per grating |
| --- | --- | --- |
| 7 June 2025 12:00 | 10 June 2025 12:00 | 4,320 (one per minute for 72 h) |

## 6. Baseline readings

| Array | Grating | Role | Baseline wavelength (nm) | Readings averaged | Std dev (µε) |
| --- | --- | --- | --- | --- | --- |
| G1 | G1-1 | bonded | 1530.0665 | 4319 | 0.8 |
| G1 | G1-2 | bonded | 1534.1829 | 4319 | 0.7 |
| G1 | G1-3 | bonded | 1539.2353 | 4318 | 1.1 |
| G1 | G1-4 | bonded | 1543.5934 | 4318 | 0.7 |
| G1 | G1-5 | bonded | 1548.2848 | 4318 | 1.0 |
| G1 | G1-6 | bonded | 1552.4995 | 4320 | 0.5 |
| G1 | G1-7 | bonded | 1556.7426 | 4320 | 0.6 |
| G1 | G1-8 | SFR | 1561.5206 | 4320 | - |
| G2 | G2-1 | bonded | 1529.6179 | 4320 | 0.7 |
| G2 | G2-2 | bonded | 1534.8402 | 4320 | 0.6 |
| G2 | G2-3 | bonded | 1539.2266 | 4320 | 0.6 |
| G2 | G2-4 | bonded | 1543.5122 | 4318 | 1.1 |
| G2 | G2-5 | bonded | 1547.6231 | 4319 | 0.9 |
| G2 | G2-6 | bonded | 1552.4307 | 4320 | 0.8 |
| G2 | G2-7 | bonded | 1557.0308 | 4320 | 1.2 |
| G2 | G2-8 | SFR | 1561.1671 | 4320 | - |
| G3 | G3-1 | bonded | 1529.8129 | 4320 | 0.9 |
| G3 | G3-2 | bonded | 1534.5099 | 4320 | 1.2 |
| G3 | G3-3 | bonded | 1538.6511 | 4320 | 0.8 |
| G3 | G3-4 | bonded | 1543.8993 | 4318 | 0.5 |
| G3 | G3-5 | bonded | 1547.7039 | 4318 | 0.7 |
| G3 | G3-6 | bonded | 1552.3212 | 4320 | 0.9 |
| G3 | G3-7 | bonded | 1556.9820 | 4320 | 0.8 |
| G3 | G3-8 | SFR | 1561.6546 | 4319 | - |

## 7. Alarm levels

| Array | Designer's allowable change (µε) | Amber, 60% (µε) | Red, 85% (µε) |
| --- | --- | --- | --- |
| G1 | 200 | 120 | 170 |
| G2 | 200 | 120 | 170 |
| G3 | 200 | 120 | 170 |

An amber or red reading on any bonded grating is handled under the project
trigger action response plan (**TARP**). A reading is confirmed against the
next two minutes of data before an alarm is sent.

## 8. Data delivery

The interrogator samples at 1 Hz and stores one-minute means. Data upload
hourly to the Marrick Atlas portal, where strain is shown after temperature
compensation. Alarm messages go by SMS and email to Ms Rowan Peake and the
Marrick duty engineer.

## 9. Limitations

- Each grating measures strain over its own 10 mm gauge length only.
- Strain values before the end of the baseline window are not reported.
- A broken fibre lead stops every grating beyond the break; the interrogator
  flags it within one minute.

## 10. Reliance

This report has been prepared by Marrick Instrumentation Ltd for the sole use
of Halverton Water Authority in connection with Greyscar Dam spillway. No
other party may rely on it without our written consent. The readings describe
the instrumented elements only, at the times stated, and say nothing about
parts of the structure without instruments.

## Appendix A. Sensor schedule

| Grating | Array | Role | Position along array (m) |
| --- | --- | --- | --- |
| G1-1 | G1 | bonded | 0.0 |
| G1-2 | G1 | bonded | 1.5 |
| G1-3 | G1 | bonded | 3.0 |
| G1-4 | G1 | bonded | 4.5 |
| G1-5 | G1 | bonded | 6.0 |
| G1-6 | G1 | bonded | 7.5 |
| G1-7 | G1 | bonded | 9.0 |
| G1-8 | G1 | SFR | 10.5 |
| G2-1 | G2 | bonded | 0.0 |
| G2-2 | G2 | bonded | 1.5 |
| G2-3 | G2 | bonded | 3.0 |
| G2-4 | G2 | bonded | 4.5 |
| G2-5 | G2 | bonded | 6.0 |
| G2-6 | G2 | bonded | 7.5 |
| G2-7 | G2 | bonded | 9.0 |
| G2-8 | G2 | SFR | 10.5 |
| G3-1 | G3 | bonded | 0.0 |
| G3-2 | G3 | bonded | 1.5 |
| G3-3 | G3 | bonded | 3.0 |
| G3-4 | G3 | bonded | 4.5 |
| G3-5 | G3 | bonded | 6.0 |
| G3-6 | G3 | bonded | 7.5 |
| G3-7 | G3 | bonded | 9.0 |
| G3-8 | G3 | SFR | 10.5 |
