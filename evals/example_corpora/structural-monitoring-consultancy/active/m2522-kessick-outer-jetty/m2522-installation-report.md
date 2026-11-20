# Kessick Outer Jetty: fibre Bragg grating installation and baseline report

- Our ref: MI-IR-2522
- Client: Port of Kessick
- Client contact: Capt. Ewan Moss
- Author: Maren Lisk
- Checked by: Priya Venn
- Approved by: Callum Draycott
- Date: 6 October 2025
- Status: Issue 1, final

## 1. Summary

Marrick Instrumentation installed 4 Lumenstrand LS-8 fibre Bragg grating
arrays, 32 gratings in all, on the tubular steel piles of the jetty head,
bents 4 to 7 of Kessick Outer Jetty from 15 to 18 September 2025. The arrays
will monitor strain during the driving of fender piles on the berthing face.
The baseline was logged from 20 September 2025 00:00 to 23 September 2025
00:00. Alarm levels are set at 60% (amber) and 85% (red) of the designer's
allowable strain change.

## 2. Scope and instrumented elements

The scope agreed with Capt. Ewan Moss covers the tubular steel piles of the
jetty head, bents 4 to 7. Each array runs along one element, and its seven
bonded gratings are spaced to cover the zone the designer expects to carry the
largest change.

| Array | Element | Bonded gratings | SFR grating |
| --- | --- | --- | --- |
| J1 | tubular steel piles of the jetty head, position 1 | 7 | J1-8 |
| J2 | tubular steel piles of the jetty head, position 2 | 7 | J2-8 |
| J3 | tubular steel piles of the jetty head, position 3 | 7 | J3-8 |
| J4 | tubular steel piles of the jetty head, position 4 | 7 | J4-8 |

## 3. Instruments

| Item | Model | Serial | Notes |
| --- | --- | --- | --- |
| FBG array | Lumenstrand LS-8 | LS8-2522-01 to -04 | 8 gratings per array |
| Interrogator | Prisma FI-4 | PF4-2043 | 4 channels, 1 Hz |
| Adhesive | Two-part methacrylate | - | Cure 24 h at 10 °C |
| Telemetry | 4G router | - | Hourly upload |

## 4. Installation

The arrays were installed from 15 to 18 September 2025. Bond zones were ground
to sound material, cleaned with solvent and primed. Each bonded grating was
fixed in methacrylate adhesive under a clamp and the SFR grating was left free
in its loose tube. Fibre leads run in armoured conduit to the interrogator
cabinet.

Gratings below the splash zone were installed from a diver-supported cradle at
low water. Each bonded grating sits under a stainless cover plate sealed
against sea water.

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
| 20 September 2025 00:00 | 23 September 2025 00:00 | 4,320 (one per minute for 72 h) |

## 6. Baseline readings

| Array | Grating | Role | Baseline wavelength (nm) | Readings averaged | Std dev (µε) |
| --- | --- | --- | --- | --- | --- |
| J1 | J1-1 | bonded | 1529.8266 | 4320 | 0.4 |
| J1 | J1-2 | bonded | 1534.5063 | 4320 | 0.7 |
| J1 | J1-3 | bonded | 1538.7965 | 4320 | 0.6 |
| J1 | J1-4 | bonded | 1543.6936 | 4320 | 0.5 |
| J1 | J1-5 | bonded | 1547.7903 | 4319 | 1.2 |
| J1 | J1-6 | bonded | 1552.1062 | 4318 | 0.8 |
| J1 | J1-7 | bonded | 1556.8808 | 4318 | 0.7 |
| J1 | J1-8 | SFR | 1561.7885 | 4319 | - |
| J2 | J2-1 | bonded | 1529.7205 | 4319 | 0.8 |
| J2 | J2-2 | bonded | 1534.8891 | 4319 | 1.1 |
| J2 | J2-3 | bonded | 1538.7820 | 4320 | 0.7 |
| J2 | J2-4 | bonded | 1543.6241 | 4318 | 0.8 |
| J2 | J2-5 | bonded | 1548.3874 | 4319 | 0.5 |
| J2 | J2-6 | bonded | 1552.5233 | 4320 | 1.1 |
| J2 | J2-7 | bonded | 1557.0069 | 4319 | 1.0 |
| J2 | J2-8 | SFR | 1561.8690 | 4319 | - |
| J3 | J3-1 | bonded | 1529.8133 | 4320 | 0.7 |
| J3 | J3-2 | bonded | 1534.6956 | 4320 | 0.5 |
| J3 | J3-3 | bonded | 1538.9776 | 4319 | 0.5 |
| J3 | J3-4 | bonded | 1543.4584 | 4318 | 0.9 |
| J3 | J3-5 | bonded | 1548.1681 | 4318 | 0.9 |
| J3 | J3-6 | bonded | 1552.3013 | 4318 | 1.1 |
| J3 | J3-7 | bonded | 1557.0273 | 4319 | 0.8 |
| J3 | J3-8 | SFR | 1561.2944 | 4319 | - |
| J4 | J4-1 | bonded | 1529.8917 | 4320 | 0.9 |
| J4 | J4-2 | bonded | 1534.5205 | 4320 | 0.7 |
| J4 | J4-3 | bonded | 1539.0947 | 4320 | 0.7 |
| J4 | J4-4 | bonded | 1543.2254 | 4319 | 1.2 |
| J4 | J4-5 | bonded | 1547.7301 | 4319 | 0.9 |
| J4 | J4-6 | bonded | 1552.3690 | 4320 | 0.5 |
| J4 | J4-7 | bonded | 1557.2675 | 4320 | 0.6 |
| J4 | J4-8 | SFR | 1561.8144 | 4320 | - |

## 7. Alarm levels

| Array | Designer's allowable change (µε) | Amber, 60% (µε) | Red, 85% (µε) |
| --- | --- | --- | --- |
| J1 | 300 | 180 | 255 |
| J2 | 300 | 180 | 255 |
| J3 | 300 | 180 | 255 |
| J4 | 300 | 180 | 255 |

An amber or red reading on any bonded grating is handled under the project
trigger action response plan (**TARP**). A reading is confirmed against the
next two minutes of data before an alarm is sent.

## 8. Data delivery

The interrogator samples at 1 Hz and stores one-minute means. Data upload
hourly to the Marrick Atlas portal, where strain is shown after temperature
compensation. Alarm messages go by SMS and email to Capt. Ewan Moss and the
Marrick duty engineer.

## 9. Marine mammal watch and hydrophone shutdown protocol

The harbour master's consent for the fender piling requires a marine mammal
watch whenever divers or piling work near the instrumented bents. Marrick's
divers follow the same protocol while they maintain the submerged gratings.

- A trained marine mammal observer watches a 500 m mitigation zone for 30
  minutes before any piling or diving starts.
- At night and in poor visibility a towed hydrophone provides passive acoustic
  monitoring in place of the visual watch.
- Work stops if a seal, porpoise or dolphin enters the mitigation zone and
  restarts only 20 minutes after the animal was last seen or heard.
- Every shutdown is logged with time, species and duration, and the log goes
  to the harbour master each week.

## 10. Limitations

- Each grating measures strain over its own 10 mm gauge length only.
- Strain values before the end of the baseline window are not reported.
- A broken fibre lead stops every grating beyond the break; the interrogator
  flags it within one minute.

## 11. Reliance

This report has been prepared by Marrick Instrumentation Ltd for the sole use
of Port of Kessick in connection with Kessick Outer Jetty. No other party may
rely on it without our written consent. The readings describe the instrumented
elements only, at the times stated, and say nothing about parts of the
structure without instruments.

## Appendix A. Sensor schedule

| Grating | Array | Role | Position along array (m) |
| --- | --- | --- | --- |
| J1-1 | J1 | bonded | 0.0 |
| J1-2 | J1 | bonded | 1.5 |
| J1-3 | J1 | bonded | 3.0 |
| J1-4 | J1 | bonded | 4.5 |
| J1-5 | J1 | bonded | 6.0 |
| J1-6 | J1 | bonded | 7.5 |
| J1-7 | J1 | bonded | 9.0 |
| J1-8 | J1 | SFR | 10.5 |
| J2-1 | J2 | bonded | 0.0 |
| J2-2 | J2 | bonded | 1.5 |
| J2-3 | J2 | bonded | 3.0 |
| J2-4 | J2 | bonded | 4.5 |
| J2-5 | J2 | bonded | 6.0 |
| J2-6 | J2 | bonded | 7.5 |
| J2-7 | J2 | bonded | 9.0 |
| J2-8 | J2 | SFR | 10.5 |
| J3-1 | J3 | bonded | 0.0 |
| J3-2 | J3 | bonded | 1.5 |
| J3-3 | J3 | bonded | 3.0 |
| J3-4 | J3 | bonded | 4.5 |
| J3-5 | J3 | bonded | 6.0 |
| J3-6 | J3 | bonded | 7.5 |
| J3-7 | J3 | bonded | 9.0 |
| J3-8 | J3 | SFR | 10.5 |
| J4-1 | J4 | bonded | 0.0 |
| J4-2 | J4 | bonded | 1.5 |
| J4-3 | J4 | bonded | 3.0 |
| J4-4 | J4 | bonded | 4.5 |
| J4-5 | J4 | bonded | 6.0 |
| J4-6 | J4 | bonded | 7.5 |
| J4-7 | J4 | bonded | 9.0 |
| J4-8 | J4 | SFR | 10.5 |
