Legacy research note

This file records the historical reverse-moving-average research input shape. It is not a
production data contract and must not be used to start QMT, submit orders, delete data, or
infer the active strategy set.

The current data ownership and retention rules are documented in
[`../docs/DATA_CONTRACTS.md`](../docs/DATA_CONTRACTS.md). Runtime boundaries are documented
in [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

Historical research data was organized below `dataset/source_data/<YYYYMMDD>/` with one
market snapshot file per timestamp. Any historical copy must remain local and must not be
committed with the repository. Use sanitized fixtures for tests and examples.

The former research question covered stock-pool filtering and reverse-movement signals.
Current production behavior is defined by the active configuration and the strategy code,
not by this note.
