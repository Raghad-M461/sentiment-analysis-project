

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = Path("sentiment_dataset_enriched.csv")
OUTPUT_DIR = Path("data")

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

RANDOM_SEED = 42


def validate_dataset(df: pd.DataFrame) -> None:

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}.\n"
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df[TEXT_COLUMN].isna().any():
        missing_text_count = df[TEXT_COLUMN].isna().sum()
        raise ValueError(
            f"The '{TEXT_COLUMN}' column contains "
            f"{missing_text_count} missing value(s)."
        )

    if df[LABEL_COLUMN].isna().any():
        missing_label_count = df[LABEL_COLUMN].isna().sum()
        raise ValueError(
            f"The '{LABEL_COLUMN}' column contains "
            f"{missing_label_count} missing value(s)."
        )

    class_counts = df[LABEL_COLUMN].value_counts()

    if len(class_counts) < 2:
        raise ValueError(
        )

    if (class_counts < 4).any():
        raise ValueError(
       
        )


def create_class_balance_table(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:

    splits = {
        "Train": train_df,
        "Validation": validation_df,
        "Test (Frozen)": test_df,
    }

    rows = []

    for split_name, split_df in splits.items():
        counts = split_df[LABEL_COLUMN].value_counts().sort_index()
        percentages = (
            split_df[LABEL_COLUMN]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
        )

        for label in counts.index:
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(counts[label]),
                    "percentage": round(float(percentages[label]), 2),
                    "total_in_split": len(split_df),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:

    print("=" * 60)
    print("Fine-Tuning Dataset Preparation")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find '{INPUT_FILE}'.\n"
           
        )

    print(f"\nLoading dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    validate_dataset(df)

    print(f"Dataset loaded successfully: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    print("\nOverall class distribution:")
    print(df[LABEL_COLUMN].value_counts())

   
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df[LABEL_COLUMN],
    )

   
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_df[LABEL_COLUMN],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test_frozen.csv"
    balance_path = OUTPUT_DIR / "class_balance_table.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    class_balance_table = create_class_balance_table(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    class_balance_table.to_csv(balance_path, index=False)

    total_rows = len(df)

    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)

    print(
        f"Train:      {len(train_df):>4} rows "
        f"({len(train_df) / total_rows * 100:.2f}%)"
    )
    print(
        f"Validation: {len(validation_df):>4} rows "
        f"({len(validation_df) / total_rows * 100:.2f}%)"
    )
    print(
        f"Test:       {len(test_df):>4} rows "
        f"({len(test_df) / total_rows * 100:.2f}%)"
    )

    print("\nClass-balance table:")
    print(class_balance_table.to_string(index=False))

    print("\n" + "=" * 60)
    print("Files Created")
    print("=" * 60)

    print(f"Training set:      {train_path}")
    print(f"Validation set:    {validation_path}")
    print(f"Frozen test set:   {test_path}")
    print(f"Class balance:     {balance_path}")

    print(
        "\nIMPORTANT: Do not use or modify data/test_frozen.csv "
        "until the final model comparison."
    )

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()
