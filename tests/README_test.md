# Testing & Integration Manual

This directory documents and organizes all tests for the DSIPTS library, including both integration and unit tests.

## Test Folder Structure

- `integration/`
  - **test_time_series_workflow.py**: Integration tests for the full time series workflow (pipeline, data, model, etc).
- `unit/`
  - **test_time_series_dataset.py**: Unit tests for time series dataset utilities and classes (e.g., `extend_time_df`, `MultiSourceTSDataSet`, `TSDataModule`).
- **test_time_series_model_integration.py**: Comprehensive integration test for D1/D2 layers and model compatibility.

## Integration Tests

### D1/D2 Layer and Model Integration Test

Validates the full pipeline from data loading (D1), windowing (D2), and model prediction. Checks both dummy and real-world (Monash) datasets.

#### Running the Integration Test

- **Default (Dummy Data):**
  ```bash
  python tests/integration/test_time_series_model_integration.py
  ```
- **Using Monash Data:**
  Set the environment variable `USE_MONASH_DATA=1` before running the test. For example:
  - **Linux/macOS:**
    ```bash
    USE_MONASH_DATA=1 python tests/integration/test_time_series_model_integration.py
    ```
  - **Windows (PowerShell):**
    ```powershell
    $env:USE_MONASH_DATA="1"; python tests/integration/test_time_series_model_integration.py
    ```

#### What the Test Does
- Loads and preprocesses time series data (dummy or Monash)
- Constructs D1 (raw) and D2 (windowed) datasets
- Runs a minimal model for compatibility and shape checks
- Asserts correct integration at each stage

#### Interpreting Results
- On success, you will see:
  ```
  SUCCESS: D1/D2 layers work properly with the model!
  ```
- On failure, error details will be logged for debugging.

#### Customizing the Data Source
- To use a different Monash dataset, edit the `id` parameter in the test file (`main()` function).

---

## Unit Tests

Unit tests are located in `unit/`. They cover core utility functions and dataset logic:

- **test_time_series_dataset.py**: Tests for
  - `extend_time_df`: Filling time gaps, handling numeric/datetime/grouped data
  - `MultiSourceTSDataSet`: Data loading, splitting, memory efficiency, NaN handling, group-based/percentage splits
  - `TSDataModule`: Windowing, caching, PyTorch Lightning integration, batch shape checks

### Running Unit Tests

From the project root:
```bash
python -m unittest discover -s tests/unit
```
Or run individual files:
```bash
python tests/unit/test_time_series_dataset.py
```

---

## Best Practices & Recommendations

- **Integration tests** should be placed in `tests/integration/` for clarity.
- **Unit tests** should remain in `tests/unit/`.
- All test scripts should be documented here for discoverability.

## Requirements
- Python 3.7+
- All dependencies installed (see main [README.md](../README.md))
- For Monash data: internet connection required
- Temporary files are written to `test_data/` in the project root

## Troubleshooting
- **Monash data download fails:** Check your internet connection, or try running with dummy data.
- **Missing dependencies:** Make sure all required Python packages are installed (see main [README.md](../README.md)).
- **Other errors:** Review the log output for details.

---

For more details on the data/model conventions, see the main [README.md](../README.md).