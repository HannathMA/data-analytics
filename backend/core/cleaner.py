"""
Waterfall Data Cleaning Pipeline for Smart Meter Readings.
Implements multi-stage filtering logic:
1. Missing recordings removal
2. Inactive / flat meters removal (zero mean, zero median, zero variance)
3. Vacant home / sudden jump filtering (>200% week-on-week consumption increase)
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
from backend.config import SERIES_HOURS, MAX_WEEK_ON_WEEK_INCREASE


class WaterfallCleaner:
    """
    Cleans raw smart meter hourly readings using a waterfall filtering pipeline.
    Tracks statistics at each stage for audit logging and visualization.
    """

    def __init__(
        self,
        series_hours: int = SERIES_HOURS,
        max_jump_ratio: float = MAX_WEEK_ON_WEEK_INCREASE,
    ):
        self.series_hours = series_hours
        self.max_jump_ratio = max_jump_ratio
        self.audit_log: Dict[str, Union[int, List[str]]] = {}

    def clean_matrix(
        self,
        df: pd.DataFrame,
        meter_id_col: Optional[str] = None,
        has_two_weeks: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Executes the waterfall cleaning pipeline on a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame where rows represent individual meters.
            Can have an explicit meter_id column or meter IDs as index.
            Columns can be hourly readings: either 168 columns (1 week)
            or 336 columns (2 consecutive weeks for week-on-week vacancy check).
        meter_id_col : str, optional
            Column name containing meter IDs. If None, df.index is used.
        has_two_weeks : bool
            If True, expects 336 hours (week 1 + week 2) to perform
            exact week-on-week vacancy comparison (>200% jump).

        Returns:
        --------
        cleaned_df : pd.DataFrame
            Cleaned matrix of shape (N_cleaned, 168) with meter IDs as index.
        audit_log : dict
            Step-by-step breakdown of dropped meters.
        """
        data = df.copy()

        # Extract meter IDs
        if meter_id_col and meter_id_col in data.columns:
            meter_ids = data[meter_id_col].astype(str).values
            data = data.drop(columns=[meter_id_col])
            data.index = meter_ids
        else:
            meter_ids = data.index.astype(str).values
            data.index = meter_ids

        initial_count = len(data)
        dropped_missing = []
        dropped_flat = []
        dropped_vacant = []

        # Convert all readings to numeric
        data = data.apply(pd.to_numeric, errors="coerce")

        # -------------------------------------------------------------
        # STAGE 1: Drop meters with missing recordings
        # -------------------------------------------------------------
        # A meter is dropped if it has any NaN or fewer columns than series_hours
        null_mask = data.isna().any(axis=1)
        dropped_missing = data.index[null_mask].tolist()
        data_stage1 = data.loc[~null_mask].copy()

        # Verify series length
        expected_cols = self.series_hours * (2 if has_two_weeks else 1)
        if data_stage1.shape[1] < expected_cols:
            raise ValueError(
                f"Data has {data_stage1.shape[1]} columns, but expected at least {expected_cols} columns."
            )

        # -------------------------------------------------------------
        # STAGE 2: Drop meters with zero mean, zero median, or zero variance
        # -------------------------------------------------------------
        # Evaluated on the first 168 hours of consumption
        week1_readings = data_stage1.iloc[:, : self.series_hours].values

        means = np.mean(week1_readings, axis=1)
        medians = np.median(week1_readings, axis=1)
        stds = np.std(week1_readings, axis=1)

        flat_mask = (means <= 1e-6) | (medians <= 1e-6) | (stds <= 1e-6)
        dropped_flat = data_stage1.index[flat_mask].tolist()
        data_stage2 = data_stage1.loc[~flat_mask].copy()

        # -------------------------------------------------------------
        # STAGE 3: Vacant homes filter (>200% week-on-week increase)
        # -------------------------------------------------------------
        if has_two_weeks and data_stage2.shape[1] >= self.series_hours * 2:
            # Full week 1 (0 to 167) vs week 2 (168 to 335)
            w1_sum = data_stage2.iloc[:, : self.series_hours].sum(axis=1).values
            w2_sum = data_stage2.iloc[:, self.series_hours : self.series_hours * 2].sum(axis=1).values

            # Percentage increase: (w2 - w1) / w1
            # Avoid division by zero by setting floor
            w1_safe = np.maximum(w1_sum, 1e-6)
            wow_increase = (w2_sum - w1_safe) / w1_safe
            vacant_mask = wow_increase > self.max_jump_ratio
            dropped_vacant = data_stage2.index[vacant_mask].tolist()
            data_stage3 = data_stage2.loc[~vacant_mask].iloc[:, : self.series_hours].copy()
        else:
            # Single week provided: detect anomalous extreme spikes (>200% jump over baseline)
            # or meters resuming from vacancy mid-week (e.g. second half > 200% first half)
            half_1 = data_stage2.iloc[:, : self.series_hours // 2].sum(axis=1).values
            half_2 = data_stage2.iloc[:, self.series_hours // 2 : self.series_hours].sum(axis=1).values
            half_1_safe = np.maximum(half_1, 1e-6)
            mid_jump = (half_2 - half_1_safe) / half_1_safe

            # Only drop if first half was near-zero vacant and second half spiked over 200%
            vacant_mask = (mid_jump > self.max_jump_ratio) & (half_1 < 5.0)
            dropped_vacant = data_stage2.index[vacant_mask].tolist()
            data_stage3 = data_stage2.loc[~vacant_mask].iloc[:, : self.series_hours].copy()

        # Ensure column names are standard: h_1, h_2, ..., h_168
        data_stage3.columns = [f"h_{i+1}" for i in range(self.series_hours)]

        final_count = len(data_stage3)
        self.audit_log = {
            "initial_count": initial_count,
            "dropped_missing_count": len(dropped_missing),
            "dropped_missing_sample": dropped_missing[:10],
            "after_missing_count": len(data_stage1),
            "dropped_flat_count": len(dropped_flat),
            "dropped_flat_sample": dropped_flat[:10],
            "after_flat_count": len(data_stage2),
            "dropped_vacant_count": len(dropped_vacant),
            "dropped_vacant_sample": dropped_vacant[:10],
            "final_count": final_count,
            "retention_rate_pct": round((final_count / max(1, initial_count)) * 100, 2),
            "series_length": self.series_hours,
        }

        return data_stage3, self.audit_log

    def resample_half_hourly_to_hourly(
        self,
        long_df: pd.DataFrame,
        meter_col: str = "LCLid",
        time_col: str = "DateTime",
        kwh_col: str = "KWH/hh",
    ) -> pd.DataFrame:
        """
        Utility for raw London SmartMeter CSV format:
        Aggregates half-hourly electricity readings into 1-hour intervals.
        """
        df = long_df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df[kwh_col] = pd.to_numeric(df[kwh_col], errors="coerce")

        # Group by meter and floor timestamp to hourly sum
        df_hourly = (
            df.groupby([meter_col, pd.Grouper(key=time_col, freq="1h")])[kwh_col]
            .sum()
            .reset_index()
        )
        return df_hourly
