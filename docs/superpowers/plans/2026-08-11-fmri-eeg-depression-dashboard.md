# fMRI-EEG Depression Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Streamlit report so the main story is fMRI/EEG maternal depression assessment, with physiological results moved into a focused auxiliary module.

**Architecture:** Split the app into configuration/domain, data access, service/view-model, and presentation layers. Keep `app.py` as a thin entrypoint and keep every Python file under 500 lines.

**Tech Stack:** Python, Streamlit, pandas, numpy, plotly, scipy, unittest/pytest-compatible tests.

---

### Task 1: Add Tests for Data and Architecture

**Files:**
- Create: `tests/test_dashboard_services.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `dashboard.data`, `dashboard.services`, and `dashboard.config`. Assert that physiological summaries include lift columns, feature-cloud terms are produced, pretraining summaries expose routeB rows without promoting routeA, and dashboard Python files stay under 500 lines.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_services.py -q`
Expected: FAIL because `dashboard` modules do not exist yet.

### Task 2: Build Four-Layer Backend Modules

**Files:**
- Create: `dashboard/config.py`
- Create: `dashboard/data.py`
- Create: `dashboard/services.py`
- Create: `dashboard/__init__.py`

- [ ] **Step 1: Implement configuration/domain layer**

Define paths, baseline definitions, method labels, Plotly config, and display constants.

- [ ] **Step 2: Implement data access layer**

Load physiological CSV files, PCA files, pretraining JSON/CSV files, and available source-space images.

- [ ] **Step 3: Implement service layer**

Compute method outcomes, best rows, deltas, clinical threshold table, feature cloud terms, pretraining cards, and routeB comparison rows.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_dashboard_services.py -q`
Expected: PASS.

### Task 3: Build Chart Layer

**Files:**
- Create: `dashboard/charts.py`

- [ ] **Step 1: Move existing Plotly figures into chart functions**

Preserve lift, delta, and PCA behavior.

- [ ] **Step 2: Add richer physiological charts**

Add feature-cloud, heatmap, radar, and pretraining comparison figures.

### Task 4: Build Streamlit Presentation Layer

**Files:**
- Create: `dashboard/views.py`
- Replace: `app.py`

- [ ] **Step 1: Add global style and layout helpers**

Create cards, status bands, section headers, and compact table rendering.

- [ ] **Step 2: Implement pages**

Implement overview, joint-analysis placeholder, pretraining, and physiological auxiliary pages with sidebar navigation.

- [ ] **Step 3: Keep `app.py` thin**

`app.py` should set page config, load data via services, and route to view functions.

### Task 5: Verification

**Files:**
- All dashboard and app files.

- [ ] **Step 1: Run automated tests**

Run: `python -m pytest tests/test_dashboard_services.py -q`

- [ ] **Step 2: Run import/syntax check**

Run: `python -m py_compile app.py dashboard/config.py dashboard/data.py dashboard/services.py dashboard/charts.py dashboard/views.py`

- [ ] **Step 3: Check file line budget**

Run: `Get-ChildItem app.py,dashboard\\*.py | ForEach-Object { "$($_.Name) $((Get-Content -LiteralPath $_.FullName).Count)" }`
