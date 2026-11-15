# Technical Memorandum: Fennick Canal Siltation Survey (Q103)

- Client: Fennick Waterways Trust
- Author: Imani Castellane
- Checked by: Dmitri Halloway
- Date: 2 June 2026
- Sensor: Sonavex R4 multibeam echosounder

## 1. Summary

Depths are below the Fennick canal datum (top water level at Hob Lock). Silt
has built up by an average of 0.31 m since 2021 across the 7.8 km reach, a
total of 9,870 m³. The thickest silt, 0.74 m, is in the pound above Carrow
aqueduct.

## 2. Datum and control

Levels are tied to the canal datum through the HWX tie-in at Hob Lock, last
repeated on 21 May 2026. Under the two road bridges the total station gave
positions within 0.03 m.

## 3. Method

The multibeam echosounder (MBES) ran at 400 kHz from a 5 m workboat. Data were
cleaned in Hydrolith 7 and differenced against the 2021 surface in SurfGrid.

## 4. Uncertainty

The mean TPU at 95 % confidence is 0.07 m for depths and 0.10 m for silt
thickness.

## 5. Silt by pound

| Pound | Length (km) | Mean silt (m) | Volume (m³) |
| --- | --- | --- | --- |
| Hob Lock to Wren Lock | 2.9 | 0.22 | 2,640 |
| Wren Lock to Mill Lock | 2.6 | 0.29 | 3,180 |
| Mill Lock to Carrow aqueduct | 2.3 | 0.44 | 4,050 |

## 6. Deliverables

| File | Contents |
| --- | --- |
| `fennick_reach_25cm.tif` | 25 cm gridded surface, canal datum |
| `fennick_silt_thickness.tif` | Silt thickness since 2021 |

## 7. Limitations

The 2021 surface was gridded at 1 m, so thickness below 0.10 m is not
meaningful.
