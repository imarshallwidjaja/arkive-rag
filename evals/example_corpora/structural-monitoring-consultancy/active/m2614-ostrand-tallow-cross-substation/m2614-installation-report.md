# Tallow Cross 132 kV substation: fibre Bragg grating installation and baseline report

- Our ref: MI-IR-2614
- Client: Ostrand Energy Networks
- Client contact: Ms Delphine Okonkwo-Hart
- Author: Maren Lisk
- Checked by: Priya Venn
- Approved by: Callum Draycott
- Date: 24 March 2026
- Status: Issue 1, final

## 1. Summary

Marrick Instrumentation installed two Lumenstrand LS-8 fibre Bragg grating
arrays, 16 gratings in all, on the transformer T2 plinth and the cable tunnel
portal at Tallow Cross 132 kV substation from 9 to 11 March 2026. The arrays
will monitor strain while Harrowby Developments builds a sheet-piled basement
14 m east of the compound fence. The baseline was logged from 13 March 2026
00:00 to 16 March 2026 00:00. Alarm levels are set at 60% (amber) and 85%
(red) of the designer's allowable strain change.

## 2. Scope and instrumented elements

The scope agreed with Ms Delphine Okonkwo-Hart covers the two elements nearest
the basement excavation. Each array runs along one element, and its seven
bonded gratings are spaced to cover the face closest to the excavation.

| Array | Element | Bonded gratings | SFR grating |
| --- | --- | --- | --- |
| TX-A | Transformer T2 plinth, east face | 7 | TX-A-8 |
| TX-B | Cable tunnel portal, east wall | 7 | TX-B-8 |

## 3. Instruments

| Item | Model | Serial | Notes |
| --- | --- | --- | --- |
| FBG array | Lumenstrand LS-8 | LS8-2614-01 to -02 | 8 gratings per array |
| Interrogator | Prisma FI-4 | PF4-2291 | 4 channels, 1 Hz |
| Adhesive | Two-part methacrylate | - | Cure 24 h at 10 °C |
| Telemetry | 4G router outside the compound fence | - | Hourly upload |

## 4. Installation

The arrays were installed from 9 to 11 March 2026 under Ostrand Energy
Networks permits to work, with the T2 transformer kept in service. Bond zones
were ground to sound concrete, cleaned with solvent and primed. Each bonded
grating was fixed in methacrylate adhesive under a clamp and the SFR grating
was left free in its loose tube. The fibre leads are dielectric, so no bonding
or earthing was needed near live plant. They run in armoured conduit to the
interrogator cabinet in the relay room.

| Date | Work | Permit |
| --- | --- | --- |
| 9 March 2026 | Array TX-A bonded to the T2 plinth east face | PTW-TC-0311 |
| 10 March 2026 | Array TX-B bonded to the cable tunnel portal east wall | PTW-TC-0312 |
| 11 March 2026 | Conduit, cabinet termination and commissioning | PTW-TC-0313 |

The relay room cabinet is fed from the substation's 110 V auxiliary supply
through a fused spur installed by Ostrand's electricians, with an eight-hour
UPS.

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
| 13 March 2026 00:00 | 16 March 2026 00:00 | 4,320 (one per minute for 72 h) |

Transformer T2 ran between 41% and 68% load during the baseline window. Its
heat output raised the plinth surface by up to 6 °C each afternoon; the SFR
compensation removed this cycle to within ±1.1 µε.

## 6. Baseline readings

| Array | Grating | Role | Baseline wavelength (nm) | Readings averaged | Std dev (µε) |
| --- | --- | --- | --- | --- | --- |
| TX-A | TX-A-1 | bonded | 1529.8371 | 4,320 | 0.7 |
| TX-A | TX-A-2 | bonded | 1534.6773 | 4,320 | 0.9 |
| TX-A | TX-A-3 | bonded | 1539.3976 | 4,320 | 0.8 |
| TX-A | TX-A-4 | bonded | 1543.4224 | 4,320 | 0.9 |
| TX-A | TX-A-5 | bonded | 1548.0537 | 4,320 | 1.0 |
| TX-A | TX-A-6 | bonded | 1552.3459 | 4,320 | 0.4 |
| TX-A | TX-A-7 | bonded | 1557.3574 | 4,320 | 0.9 |
| TX-A | TX-A-8 | SFR | 1561.5755 | 4,320 | - |
| TX-B | TX-B-1 | bonded | 1530.1016 | 4,320 | 0.6 |
| TX-B | TX-B-2 | bonded | 1534.3521 | 4,320 | 0.7 |
| TX-B | TX-B-3 | bonded | 1538.6371 | 4,320 | 0.7 |
| TX-B | TX-B-4 | bonded | 1543.4900 | 4,320 | 0.9 |
| TX-B | TX-B-5 | bonded | 1548.3121 | 4,320 | 0.5 |
| TX-B | TX-B-6 | bonded | 1552.7570 | 4,320 | 0.8 |
| TX-B | TX-B-7 | bonded | 1556.7341 | 4,320 | 0.4 |
| TX-B | TX-B-8 | SFR | 1561.1832 | 4,320 | - |

## 7. Alarm levels

| Array | Designer's allowable change (µε) | Amber, 60% (µε) | Red, 85% (µε) |
| --- | --- | --- | --- |
| TX-A, plinth | 180 | 108 | 153 |
| TX-B, tunnel portal | 220 | 132 | 187 |

The allowable changes come from Ostrand Energy Networks' asset protection
engineer. An amber or red reading on any bonded grating is handled under the
project trigger action response plan (**TARP**). A reading is confirmed
against the next two minutes of data before an alarm is sent.

## 8. Data delivery

The interrogator samples at 1 Hz and stores one-minute means. Data upload
hourly to the Marrick Atlas portal, where strain is shown after temperature
compensation. Alarm messages go by SMS and email to Ms Delphine Okonkwo-Hart,
the Ostrand control room at Hessle and the Marrick duty engineer.

## 9. Limitations

- Each grating measures strain over its own 10 mm gauge length only.
- Strain values before the end of the baseline window are not reported.
- A broken fibre lead stops every grating beyond the break; the interrogator
  flags it within one minute.
- Access to the arrays for maintenance needs an Ostrand permit to work and an
  authorised person on site.

## 10. Reliance

This report has been prepared by Marrick Instrumentation Ltd for the sole use
of Ostrand Energy Networks in connection with Tallow Cross 132 kV substation.
No other party may rely on it without our written consent. The readings
describe the instrumented elements only, at the times stated, and say nothing
about parts of the structure without instruments.

## Appendix A. Sensor schedule

| Grating | Array | Role | Position along array (m) |
| --- | --- | --- | --- |
| TX-A-1 | TX-A | bonded | 0.0 |
| TX-A-2 | TX-A | bonded | 1.2 |
| TX-A-3 | TX-A | bonded | 2.4 |
| TX-A-4 | TX-A | bonded | 3.6 |
| TX-A-5 | TX-A | bonded | 4.8 |
| TX-A-6 | TX-A | bonded | 6.0 |
| TX-A-7 | TX-A | bonded | 7.2 |
| TX-A-8 | TX-A | SFR | 8.4 |
| TX-B-1 | TX-B | bonded | 0.0 |
| TX-B-2 | TX-B | bonded | 1.2 |
| TX-B-3 | TX-B | bonded | 2.4 |
| TX-B-4 | TX-B | bonded | 3.6 |
| TX-B-5 | TX-B | bonded | 4.8 |
| TX-B-6 | TX-B | bonded | 6.0 |
| TX-B-7 | TX-B | bonded | 7.2 |
| TX-B-8 | TX-B | SFR | 8.4 |

## Appendix B. Commissioning checks

| Check | Criterion | TX-A | TX-B |
| --- | --- | --- | --- |
| Return loss | > 40 dB | 47 dB | 45 dB |
| Peak width | < 0.3 nm | 0.21 nm | 0.24 nm |
| SFR free in tube | Moves freely | Yes | Yes |
| Conduit sealed | Both ends | Yes | Yes |
| Baseline std dev | < 1.5 µε | Pass | Pass |
| Alarm test message | Received by client | 12 March 2026 15:10 | 12 March 2026 15:10 |
