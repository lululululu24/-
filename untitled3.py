# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:54:43 2025

@author: 75235
"""

"""
Created on Wed Nov 26 15:28:23 2025

@author: 75235
"""

import ast
from typing import Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



FILE_PATH = "motor_insurance_recovery.csv"




def load_data(file_path: str = FILE_PATH) -> pd.DataFrame:
    """
    Load the motor insurance dataset (CSV file) and print some basic info.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Loaded table as a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {file_path}")
        raise e
    except Exception as e:
        print(f"[ERROR] Failed to load data from: {file_path}")
        raise e

    print("Dataset loaded successfully.")
    print("Shape (rows, columns):", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    return df


def safe_literal_set(s: str) -> Set[str]:
    """
    Safely convert a string that looks like a set/list/etc into a real Python set.
    If parsing fails or the value is missing, return an empty set().

    Examples
    --------
    - "{'Head Injury', 'Soft Tissue Injury'}"
    - "['Head Injury', 'Soft Tissue Injury']"
    """
    if pd.isna(s) or s == "":
        return set()

    if not isinstance(s, str):
        s = str(s)

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (set, list, tuple)):
            return set(map(str, parsed))
        return {str(parsed)}
    except (ValueError, SyntaxError):
        return set()


def summarise_categorical(series: pd.Series) -> pd.DataFrame:
    """
    Create a frequency table for a categorical variable, including
    count (absolute frequency) and percentage.

    Parameters
    ----------
    series : pandas.Series
        Categorical variable.

    Returns
    -------
    freq_table : pandas.DataFrame
        DataFrame with columns ['count', 'percentage'].
    """
    counts = series.value_counts(dropna=False)
    perc = series.value_counts(normalize=True, dropna=False) * 100
    freq_table = pd.DataFrame({
        "count": counts,
        "percentage": perc.round(2)
    })
    return freq_table


# =========================
# 2. TASK A – Basic exploration
# =========================

def task_a_basic_exploration(df: pd.DataFrame) -> None:
    """
    TASK A: Basic exploratory data analysis (EDA).

    Includes:
    - Dataset shape and basic info
    - Number of missing values per column
    - Descriptive statistics for numeric variables
    - Frequency tables for key categorical variables
    """
    print("\n===== TASK A: BASIC EXPLORATION =====")

    # 1. Overall dataset information
    print("\nDataset shape (rows, columns):", df.shape)
    print("\nDataFrame info:")
    df.info()

    # 2. Missing values per column
    print("\nMissing values per column:")
    print(df.isna().sum())

    # 3. Descriptive statistics for numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nDescriptive statistics for numeric variables:")
        print(df[numeric_cols].describe().T)
    else:
        print("\n[Note] No numeric columns found.")

    # 4. Frequency tables for selected categorical variables
    categorical_cols = [
        "claim_value_category",
        "hospital_visit_required",
        "hospital_admission_required",
        "rehabilitation_recommended",
        "rehabilitation_completed",
        "car_damage_severity",
        "liability_admission_status",
        "liability_type",
        "emergency_services_attended"
    ]

    for col in categorical_cols:
        if col in df.columns:
            print(f"\nFrequency table for {col}:")
            freq_table = summarise_categorical(df[col])
            print(freq_table)
        else:
            print(f"\n[Note] Column '{col}' not found in the dataset.")


# =========================
# 3. TASK A – Data quality checks
# =========================

def _standardise_text_column(series: pd.Series) -> pd.Series:
    """
    Standardise a text / categorical column:
    - Keep original missing values (NaN) as NaN
    - For non-missing values: convert to string, strip whitespace, lowercase
    """
    is_na = series.isna()

    cleaned = (
        series.astype(str)
              .str.strip()
              .str.lower()
    )

    cleaned[is_na] = np.nan
    return cleaned


def task_a_data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK A: Data quality checks and simple cleaning.

    Includes:
    - Detect and remove fully duplicated rows
    - Standardise text format for some key categorical variables
    - Check how many negative values there are in numeric variables

    Returns
    -------
    cleaned_df : pandas.DataFrame
        A lightly cleaned version of the dataset.
    """
    print("\n===== TASK A: DATA QUALITY CHECKS =====")

    cleaned_df = df.copy()

    # 1. Find and remove duplicate rows
    num_duplicates = cleaned_df.duplicated().sum()
    print(f"\nNumber of fully duplicated rows: {num_duplicates}")
    if num_duplicates > 0:
        cleaned_df = cleaned_df.drop_duplicates()
        print("Duplicates removed. New shape:", cleaned_df.shape)

    # 2. Standardise text in selected categorical columns
    cat_cols_to_clean = [
        "car_damage_severity",
        "liability_admission_status",
        "liability_type",
        "emergency_services_attended"
    ]
    existing_cat_cols = [c for c in cat_cols_to_clean if c in cleaned_df.columns]

    if existing_cat_cols:
        print("\nStandardising text in categorical variables:")
        for col in existing_cat_cols:
            print(f"- Cleaning column: {col}")
            cleaned_df[col] = _standardise_text_column(cleaned_df[col])
    else:
        print("\n[Note] None of the specified categorical columns were found.")

    # 3. Range checks: count negative values in numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    print("\nRange checks for numeric variables (number of negative values):")
    if len(numeric_cols) == 0:
        print("[Note] No numeric columns found.")
    else:
        for col in numeric_cols:
            num_negative = (cleaned_df[col] < 0).sum()
            if num_negative > 0:
                print(f"- {col}: {num_negative} negative values")
            else:
                print(f"- {col}: no negative values")

    return cleaned_df


# =========================
# 4. Code executed when script runs
# =========================


df = load_data()
task_a_basic_exploration(df)
cleaned_df = task_a_data_quality_checks(df)
