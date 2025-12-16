# Unit test file for data processing module.

def test_feature_columns_exist():
    expected_cols = {
        "Total_Transaction_Amount",
        "Transaction_Count",
        "Transaction_Recency",
        "Dormant_Flag"
    }

    assert expected_cols.issubset(set(X.columns))

def test_target_binary():
    assert set(y.unique()).issubset({0, 1})
