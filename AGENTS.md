# Hermes Agent Guide

> **Document type:** repository-level agent contract
> **Applies to:** this repository and every file below it
> **Source of truth for:** safe changes, validation boundaries, and handoff behavior
> **Reviewed:** 2026-11-17

Hermes is a live trading system. Treat this repository as production-sensitive software, not as a generic sample application.

## Reading order

1. Read this file before changing anything.
2. Read [`docs/INDEX.md`](docs/INDEX.md) to locate the authoritative document for the task.
3. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for runtime ownership and control flow.
4. Read [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) before touching `conf/` or paths.
5. Read [`docs/OPERATIONS.md`](docs/OPERATIONS.md) before touching startup, scheduling, accounts, gateways, or data refresh.
6. Read [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) before changing code or tests.

Use the words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** in their normal normative sense. If the repository and a document disagree, inspect the current code/configuration and update the affected document as part of the same change.

## Non-negotiable safety rules

- **MUST NOT** launch `run.py`, `main.py`, a scheduled task, an ETL script, or a strategy against QMT as a development test. They may connect to brokerage accounts, write shared production data, or submit orders.
- **MUST NOT** change account identifiers, QMT paths, active strategies, `is_test`, order gates, risk limits, gateway selection, or scheduled-task behavior unless the user explicitly requests that change.
- **MUST** keep order submission fail-closed. Startup reconciliation, connection health, account validation, risk checks, and the managed order path must not be bypassed.
- **MUST NOT** add offline training to Hermes. Training and model evaluation belong in `D:/tool/pycharm_workspace/Hephaestus-Forge`; Hermes consumes prepared data, predictions, or scores.
- The former `incubate` directory is in `D:/tool/pycharm_workspace/Hephaestus-Forge/incubate`. **MUST NOT** restore it or re-enable legacy MLP strategies without resolving imports and documenting the intended runtime boundary.
- **MUST NOT** delete, regenerate, overwrite, or bulk-move `D:/tool/dataset`, order databases, logs, model files, strategy state, or broker data as cleanup without explicit authorization.
- **MUST NOT** put account IDs, credentials, private broker paths, or raw production data in documentation, tests, commits, or summaries. Use sanitized placeholders.

## Architecture boundaries

- `run.py`: current full service entry point and lifecycle coordinator.
- `conf/`: runtime and strategy configuration. The active strategy set is `[strategy] name` in `conf/config.ini`; a section existing in the file does not make it active.
- `execution/`: order protocol, broker gateways, risk checks, health, reconciliation, recovery, and managed compatibility facade.
- `strategy/`: strategy implementations. Preserve the dynamically loaded class name and `do(accounts)` contract unless the migration is deliberate and documented.
- `etl/`: market, pool, factor, and strategy-input preparation. It is not the offline model-training home.
- `test/`: mixed tests. Unit tests can be safe; QMT, market-data, network, and account tests are not safe to run blindly.
- `docs/`: operational, architectural, configuration, data-contract, development, and decision documents.

Strategy orders **MUST** go through `OrderManager` or the `ManagedTrader` compatibility facade. Do not call a raw broker submission API directly from a strategy.

## Change protocol

Before editing:

1. Run `git status --short` and inspect the existing diff.
2. Identify the authoritative code/configuration and the documents that describe it.
3. Separate user changes, generated files, live data, and the requested change.
4. Decide whether the change is code, configuration, operational, documentation, or a decision record.

While editing:

- Keep the change scoped; preserve unrelated modified and untracked files.
- Keep machine-specific paths in configuration where the existing design permits it.
- Avoid import-time network, QMT, account, or order side effects.
- Update the relevant document when runtime ownership, startup order, integration boundaries, or operator steps change.
- Add an ADR under `docs/decisions/` when a change alters a durable architecture or safety boundary.

Before handoff:

- Summarize changed files, behavior, assumptions, and operational risk.
- State which safe checks ran and which live/integration checks were intentionally skipped.
- Recheck referenced paths, active strategy names, test-mode switches, and document dates.

## Verification policy

- Prefer static checks, syntax compilation, focused unit tests, fake gateways, temporary SQLite files, and isolated fixtures.
- The execution boundary is covered by focused tests such as `test/test_order_protocol.py`, `test/test_managed_trader.py`, `test/test_multi_gateway.py`, `test/test_recovery.py`, and `test/test_risk.py`; inspect selected tests before running them.
- **MUST NOT** run the whole test tree blindly. It contains QMT, market-data, network, and account integration checks.
- **MUST NOT** use live brokerage connectivity to verify a code change.
- Documentation-only changes should verify links, referenced paths, current strategy names, and operating-mode claims against code/configuration.

## Current operating facts

- The current full service entry point is `run.py`; `main.py` is an older minimal example.
- The configured active strategy list is in `conf/config.ini`.
- `conf/top10_v8_v1.ini` has its own `is_test` switch, separate from `[client] is_test`.
- Startup creates order managers with submission disabled and only opens the gate after clean connection, subscription, reconciliation, and account validation.
- Some startup checks invoke Hephaestus-Forge or Atropos scripts. Treat them as external process boundaries, not as Hermes training code.
