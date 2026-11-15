# Technical Memorandum: Brenlow Estuary Channel Survey (Q101)

- Client: Brenlow Port Authority
- Author: Imani Castellane
- Checked by: Petra Valdane
- Date: 14 April 2026
- Sensor: Sonavex R4 multibeam echosounder

## 1. Summary

Quillfen Geomatics surveyed 12.6 km² of the Brenlow inner channel between 16
and 19 March 2026. The minimum depth in the dredged channel is 6.42 m below
Chart Datum, at the bend south of the Ferrow road bridge.

## 2. Datum and control

All depths are reduced to Chart Datum. Vertical control came from the HWX
tie-in at the Saltgate tide gauge, checked against two benchmarks on the
breakwater. Horizontal positions use RTK GNSS from the Brenlow reference
station.

## 3. Method

The survey used a multibeam echosounder (MBES), the Sonavex R4, at 400 kHz.
Sound speed casts were taken every 3 hours. Soundings were cleaned in
Hydrolith 7 and gridded in SurfGrid.

## 4. Uncertainty

Total propagated uncertainty (TPU) combines the sounder, positioning, motion
and tide errors for each sounding. The mean TPU at 95 % confidence is 0.18 m,
inside the 0.25 m limit in the proposal.

## 5. Deliverables

| File | Contents |
| --- | --- |
| `brenlow_dtm_50cm.tif` | 50 cm gridded surface, Chart Datum |
| `brenlow_lines_2026.shp` | 214 run lines |
| `brenlow_soundings.csv` | 1,284,330 cleaned soundings |

## 6. Limitations

- Two lines near the Saltgate pontoon were blocked by moored vessels; that
  area has 70 % coverage.
- The HWX tie-in was not repeated after 18 March, so the last day relies on
  the earlier check.
