# STAGED / EXPERIMENTAL — EAGLE-I power-outage timing overlay

**This is not a headline result and must not be presented as validated accuracy.**
It is a directional, aggregate timing comparison with structural confounds:

- **County-level, not node-level.** EAGLE-I reports Suffolk County (FIPS 25025) customers-without-power; it cannot validate which infrastructure NODES failed. It is only comparable to the model's *aggregate* power-failure curve.
- **Mixed hazard.** January 2018 Boston outages were partly wind/snow-driven, not purely coastal flood. The flood model represents only the flood driver, so the real curve contains outages the model cannot (and should not) reproduce.
- **Clock anchoring.** The model has no absolute clock; t=0 is anchored to the storm peak **2018-01-04 17:42 UTC** (= 12:42 EST, the NOAA Phase-1 observed-peak timestamp) and the model horizon (t=6..96 h) is laid on that. EAGLE-I run_start_time stamps are UTC, so both curves share one clock.

Real Suffolk peak: 1393 customers out @ 2018-01-07 08:30:00 UTC.

**Key confound made visible:** the real Suffolk outage peak lands on **Jan 7**, ~3 days AFTER the Jan-4 flood/storm peak. That lag is the clearest evidence the January 2018 Boston outages were largely NOT flood-driven (wind/snow/cold), so the model's flood-driven power ramp should NOT be expected to reproduce it. Use only as a sanity check that the model's ramp is not grossly mistimed relative to the event window. Real node-level outage validation requires utility circuit/feeder records, which we do not have.
