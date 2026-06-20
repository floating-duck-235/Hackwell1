# RepoHeal Migration Document

**Immutable**: this document is an append-only migration record and must not be modified after creation.
**Repository**: floating-duck-235/Hackwell1
**Generated**: 2026-06-20T10:54:16.022450+00:00

---

| Metric | Score |
|--------|------:|
| **Repository Health** | 72/100 |
| **Migration Risk** | 36/100 (LOW) |
| **Average Confidence** | 0.90 |
- ⚠ flask: failed
- ⚠ flask-cors: degraded (budget_exhausted)
- ⚠ joblib: failed
- ⚠ pandas: failed
- ⚠ scikit-learn: failed
- ⚠ numpy: failed


---

# Migration Assessment

**Repository**: floating-duck-235/Hackwell1
**Generated**: 2026-06-20T10:54:16.022450+00:00
**Overall Health Score**: 72/100
**Migration Risk Score**: 36/100
**Migration Risk Level**: LOW
**Migration Intelligence Status**: DEGRADED

## Executive Summary

- Critical findings: 0
- Deprecated APIs: 0
- Breaking changes: 1

**Migration Intelligence Status**: DEGRADED
**Note**: Primary intelligence provider had errors, but local fallback provided data.
**Error**: flask: failed
**Intelligence Source**: local_kb

---

## Intelligence Warnings

- flask: failed
- flask-cors: degraded (budget_exhausted)
- joblib: failed
- pandas: failed
- scikit-learn: failed
- numpy: failed

---

## Dependency Inventory

| Dependency | Installed | Latest | Status | Type |
| --- | --- | --- | --- | --- |
| datetime | unknown | 6.0 | missing | detected |
| flask | 2.3.3 | 3.1.3 | declared | third-party |
| flask-cors | 4.0.0 | 6.0.5 | declared | third-party |
| joblib | 1.3.2 | 1.5.3 | declared | third-party |
| json | unknown | unknown | missing | detected |
| matplotlib | unknown | 3.11.0 | missing | detected |
| numpy | 1.24.3 | 2.4.6 | declared | third-party |
| os | unknown | unknown | missing | detected |
| pandas | 2.0.3 | 3.0.3 | declared | third-party |
| random | unknown | unknown | missing | detected |
| scikit-learn | 1.3.0 | 1.9.0 | declared | third-party |
| seaborn | unknown | 0.13.2 | missing | detected |
| sys | unknown | unknown | missing | detected |
| warnings | unknown | unknown | missing | detected |

---

## Deprecated APIs

None found.

---

## Breaking Changes

### flask@2.3.3 → 3.1.3
- Library: flask
- Installed version: 2.3.3
- Latest version: 3.1.3
- Status: at_risk

### flask-cors@4.0.0 → 6.0.5
- Library: flask-cors
- Installed version: 4.0.0
- Latest version: 6.0.5
- Status: breaking

### pandas@2.0.3 → 3.0.3
- Library: pandas
- Installed version: 2.0.3
- Latest version: 3.0.3
- Status: at_risk

### numpy@1.24.3 → 2.4.6
- Library: numpy
- Installed version: 1.24.3
- Latest version: 2.4.6
- Status: at_risk

---

## Migration Paths
- `flask@2.3.3 → 3.1.3` -> `3.1.3` (major_version_gap, confidence: 0.90)
- `flask-cors@4.0.0 → 6.0.5` -> `6.0.5` (major_version_gap, confidence: 0.90)
- `pandas@2.0.3 → 3.0.3` -> `3.0.3` (major_version_gap, confidence: 0.90)
- `numpy@1.24.3 → 2.4.6` -> `2.4.6` (major_version_gap, confidence: 0.90)

---

## Risk Assessment

### flask@2.3.3 → 3.1.3
- Risk score: 24/100 (low)
- Affected files: 0 (0.0%)
- Breaking severity: medium
- Replacement confidence: 0.90
- Call chain depth: 0

### flask-cors@4.0.0 → 6.0.5
- Risk score: 36/100 (low)
- Affected files: 0 (0.0%)
- Breaking severity: high
- Replacement confidence: 0.90
- Call chain depth: 0

### pandas@2.0.3 → 3.0.3
- Risk score: 23/100 (low)
- Affected files: 0 (0.0%)
- Breaking severity: medium
- Replacement confidence: 0.90
- Call chain depth: 0

### numpy@1.24.3 → 2.4.6
- Risk score: 29/100 (low)
- Affected files: 0 (0.0%)
- Breaking severity: medium
- Replacement confidence: 0.90
- Call chain depth: 0

---

## Recommended Actions
- [MEDIUM] `flask-cors@4.0.0 → 6.0.5` (confidence: 0.90): Migrate flask-cors@4.0.0 → 6.0.5 to avoid potential breakage. Impacted files: 0
- [MEDIUM] `numpy@1.24.3 → 2.4.6` (confidence: 0.90): Migrate numpy@1.24.3 → 2.4.6 to avoid potential breakage. Impacted files: 0
- [MEDIUM] `flask@2.3.3 → 3.1.3` (confidence: 0.90): Migrate flask@2.3.3 → 3.1.3 to avoid potential breakage. Impacted files: 0
- [MEDIUM] `pandas@2.0.3 → 3.0.3` (confidence: 0.90): Migrate pandas@2.0.3 → 3.0.3 to avoid potential breakage. Impacted files: 0
