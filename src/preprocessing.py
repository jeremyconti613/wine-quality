"""Preprocessing script for Wine Quality datasets.

Creates a combined dataset from red and white CSVs, encodes labels,
scales numeric features, splits into train/test and saves artifacts.

Usage:
    python src/preprocessing.py
    python src/preprocessing.py --help
"""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(red_path: Path, white_path: Path) -> pd.DataFrame:
    df_red = pd.read_csv(red_path, sep=';')
    df_red['wine_type'] = 'red'
    df_white = pd.read_csv(white_path, sep=';')
    df_white['wine_type'] = 'white'
    df = pd.concat([df_red, df_white], ignore_index=True)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Return a concise data quality report (missing values, dtypes, shape)."""
    total_rows, total_cols = df.shape
    missing_per_col = df.isna().sum()
    missing_total = int(missing_per_col.sum())
    missing_pct = (missing_per_col / total_rows * 100).round(3)

    dtypes = df.dtypes.astype(str).to_dict()

    report = {
        'rows': total_rows,
        'cols': total_cols,
        'missing_per_column': missing_per_col.to_dict(),
        'missing_total': missing_total,
        'missing_percent_per_column': missing_pct.to_dict(),
        'dtypes': dtypes,
    }
    return report


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarizing missing counts and percentages per column."""
    total = len(df)
    s = df.isna().sum()
    pct = (s / total * 100).round(3)
    return pd.DataFrame({'missing_count': s, 'missing_percent': pct}).sort_values('missing_count', ascending=False)


def detect_outliers_iqr(df: pd.DataFrame, cols: list | None = None, k: float = 1.5) -> pd.DataFrame:
    """Detect outliers per numeric column using IQR rule. Returns a DataFrame with counts and bounds."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for c in cols:
        col = df[c].dropna()
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        outlier_mask = (df[c] < lower) | (df[c] > upper)
        out_count = int(outlier_mask.sum())
        pct = round(out_count / len(df) * 100, 3) if len(df) else 0.0
        rows.append({'column': c, 'outliers': out_count, 'outlier_percent': pct, 'lower_bound': float(lower), 'upper_bound': float(upper)})
    return pd.DataFrame(rows).sort_values('outliers', ascending=False)


def detect_outliers_zscore(df: pd.DataFrame, cols: list | None = None, thresh: float = 3.0) -> pd.DataFrame:
    """Detect outliers per numeric column using z-score thresholding."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for c in cols:
        col = df[c]
        mean = col.mean()
        std = col.std(ddof=0)
        if std == 0 or np.isnan(std):
            out_count = 0
        else:
            z = (col - mean).abs() / std
            out_count = int((z > thresh).sum())
        pct = round(out_count / len(df) * 100, 3) if len(df) else 0.0
        rows.append({'column': c, 'outliers_z': out_count, 'outlier_percent_z': pct})
    return pd.DataFrame(rows).sort_values('outliers_z', ascending=False)


def handle_missing(df: pd.DataFrame, strategy: str = 'median', columns: list | None = None, constant=None) -> pd.DataFrame:
    """Handle missing values with several strategies:
    - 'drop': drop rows with any NA (or only in `columns` if provided)
    - 'median' / 'mean' / 'mode': impute numeric columns
    - 'ffill' / 'bfill': forward/backward fill
    - 'constant': fill with `constant` (scalar or dict)
    Returns a new DataFrame.
    """
    df_out = df.copy()
    if columns is None:
        columns = df_out.columns.tolist()

    if strategy == 'drop':
        return df_out.dropna(subset=columns)

    if strategy in ('median', 'mean'):
        for c in columns:
            if pd.api.types.is_numeric_dtype(df_out[c]):
                if strategy == 'median':
                    val = df_out[c].median()
                else:
                    val = df_out[c].mean()
                df_out[c] = df_out[c].fillna(val)
        return df_out

    if strategy == 'mode':
        for c in columns:
            mode = df_out[c].mode()
            if not mode.empty:
                df_out[c] = df_out[c].fillna(mode.iloc[0])
        return df_out

    if strategy in ('ffill', 'bfill'):
        return df_out.fillna(method=strategy)

    if strategy == 'constant':
        if isinstance(constant, dict):
            return df_out.fillna(value=constant)
        else:
            return df_out.fillna(constant)

    raise ValueError(f"Unknown missing value strategy: {strategy}")


def handle_outliers(df: pd.DataFrame, method: str = 'cap', cols: list | None = None, k: float = 1.5) -> pd.DataFrame:
    """Handle outliers detected by IQR rule.
    - 'cap': cap values to lower/upper bounds
    - 'remove': drop rows that contain outliers
    - 'none': do nothing
    Returns a new DataFrame.
    """
    df_out = df.copy()
    if cols is None:
        cols = df_out.select_dtypes(include=[np.number]).columns.tolist()

    bounds = {}
    for c in cols:
        col = df_out[c].dropna()
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        bounds[c] = (lower, upper)

    if method == 'none':
        return df_out

    if method == 'cap':
        for c, (lower, upper) in bounds.items():
            df_out[c] = df_out[c].clip(lower=lower, upper=upper)
        return df_out

    if method == 'remove':
        mask = pd.Series(False, index=df_out.index)
        for c, (lower, upper) in bounds.items():
            mask = mask | (df_out[c] < lower) | (df_out[c] > upper)
        return df_out.loc[~mask].reset_index(drop=True)

    raise ValueError(f"Unknown outlier handling method: {method}")


def prepare_features(df: pd.DataFrame, quality_threshold: int = 7):
    df = df.copy()
    df['quality_label'] = (df['quality'] >= quality_threshold).astype(int)

    # One-hot encode wine_type
    df = pd.get_dummies(df, columns=['wine_type'], drop_first=True)

    # Features: all except target(s)
    feature_cols = [c for c in df.columns if c not in ('quality', 'quality_label')]

    X = df[feature_cols]
    y = df['quality_label']
    return X, y, feature_cols


def scale_and_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def save_artifacts(out_dir: Path, X_train, X_test, y_train, y_test, scaler, feature_cols):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / 'X_train.csv', index=False)
    X_test.to_csv(out_dir / 'X_test.csv', index=False)
    y_train.to_csv(out_dir / 'y_train.csv', index=False)
    y_test.to_csv(out_dir / 'y_test.csv', index=False)
    joblib.dump(scaler, out_dir / 'scaler.joblib')
    # Save feature list
    pd.Series(feature_cols).to_csv(out_dir / 'feature_columns.txt', index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description='Preprocess wine quality datasets')
    parser.add_argument('--red', default='../data/winequality-red.csv')
    parser.add_argument('--white', default='../data/winequality-white.csv')
    parser.add_argument('--out', default='../data/processed')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--quality-threshold', type=int, default=7,
                        help='Threshold to consider a wine as "good" (quality >= threshold)')

    args = parser.parse_args()

    red_path = Path(args.red)
    white_path = Path(args.white)
    out_dir = Path(args.out)

    print('Loading data...')
    df = load_data(red_path, white_path)
    print(f'Data loaded: {df.shape[0]} rows, {df.shape[1]} columns')

    missing = df.isna().sum().sum()
    print(f'Missing values: {missing}')

    print('Preparing features...')
    X, y, feature_cols = prepare_features(df, quality_threshold=args.quality_threshold)
    print(f'Feature matrix: {X.shape}, Target: {y.shape}')

    print('Scaling and splitting...')
    X_train, X_test, y_train, y_test, scaler = scale_and_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print('Saving artifacts...')
    save_artifacts(out_dir, X_train, X_test, y_train, y_test, scaler, feature_cols)

    print('Done. Artifacts saved to:', out_dir)


if __name__ == '__main__':
    main()
