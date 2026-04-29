import pandas as pd

def test_no_missing_values():
    df = pd.read_csv('data/processed/credit_data_cleaned.csv')
    assert df.isnull().sum().sum() == 0
    print("✅ No missing values")

def test_age_range():
    df = pd.read_csv('data/processed/credit_data_cleaned.csv')
    assert df['age'].min() >= 18
    assert df['age'].max() <= 100
    print("✅ Age range valid")

def test_utilization_range():
    df = pd.read_csv('data/processed/credit_data_cleaned.csv')
    assert df['RevolvingUtilizationOfUnsecuredLines'].max() <= 1
    print("✅ Utilization range valid")

if __name__ == '__main__':
    test_no_missing_values()
    test_age_range()
    test_utilization_range()
    print("\n✅ All tests passed!")
