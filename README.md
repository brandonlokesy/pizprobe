# PizProbe

AFM scan loading and analysis. *Piz* is Romansh for peak.

```bash
pip install -e ".[test]"
```

```python
from pizprobe import AsylumScan

scan = AsylumScan("dose_280_wg0002.ibw")   # undoes Asylum's saved line flatten by default
print(scan)                                 # scan size, pixel size, rate, drive, setpoint, ...
scan.channels                               # ['HeightRetrace', 'AmplitudeRetrace', ...]
z = scan["ZSensorRetrace"].data             # 2-D array in metres
scan.metadata["ScanRate"]                   # every wave-note entry, parsed
```

Data stay in SI units (m, deg). Images use the Asylum/AFMReader orientation
(`np.flipud(wData.T)`). Pass `undo_default_correction=False` to get the data exactly as saved.
