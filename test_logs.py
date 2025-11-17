============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0 -- /home/jorballcor/Aplica/vehicle-insurance-mlops-pipeline/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/jorballcor/Aplica/vehicle-insurance-mlops-pipeline
configfile: pytest.ini
collecting ... collected 16 items

tests/test_data_validation.py::TestDataValidation::test_init_success PASSED [  6%]
tests/test_data_validation.py::TestDataValidation::test_validate_number_of_columns_correct PASSED [ 12%]
tests/test_data_validation.py::TestDataValidation::test_validate_number_of_columns_incorrect PASSED [ 18%]
tests/test_data_validation.py::TestDataValidation::test_is_column_exist_all_present PASSED [ 25%]
tests/test_data_validation.py::TestDataValidation::test_is_column_exist_missing_columns PASSED [ 31%]
tests/test_data_validation.py::TestDataValidation::test_read_data_success PASSED [ 37%]
tests/test_data_validation.py::TestDataValidation::test_read_data_file_not_found PASSED [ 43%]
tests/test_data_validation.py::TestDataValidation::test_initiate_data_validation_success PASSED [ 50%]
tests/test_data_validation.py::TestDataValidation::test_initiate_data_validation_failure PASSED [ 56%]
tests/test_data_validation.py::TestDataValidation::test_column_existence_various_scenarios[missing_columns0-False] PASSED [ 62%]
tests/test_data_validation.py::TestDataValidation::test_column_existence_various_scenarios[missing_columns1-False] PASSED [ 68%]
tests/test_data_validation.py::TestDataValidation::test_column_existence_various_scenarios[missing_columns2-False] PASSED [ 75%]
tests/test_data_validation.py::TestDataValidation::test_column_existence_various_scenarios[missing_columns3-False] PASSED [ 81%]
tests/test_data_validation.py::TestDataValidation::test_column_existence_various_scenarios[missing_columns4-True] PASSED [ 87%]
tests/test_data_validation.py::TestDataValidation::test_large_dataframe_validation PASSED [ 93%]
tests/test_data_validation.py::TestDataValidation::test_validation_report_content PASSED [100%]

============================== 16 passed in 0.17s ==============================
