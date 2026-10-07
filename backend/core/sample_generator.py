"""
Synthetic Smart Meter Data Generator for London SmartMeter Demonstration.
Generates realistic 168-hour (and 336-hour 2-week) consumption profiles
exhibiting diverse residential behavioral patterns along with intentional edge cases
to demonstrate the waterfall data cleaning pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from backend.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SERIES_HOURS


def generate_london_sample_dataset(
    n_households: int = 250,
    include_two_weeks: bool = True,
    random_seed: int = 42,
    save_files: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a realistic smart meter dataset with varied temporal signatures:
    - Standard Diurnal Evening Peak
    - Morning & Evening Dual Peak
    - Work-from-Home Daytime Continuous
    - Night-active / Shift Worker
    - High Autoregressive Heating/Cooling
    - Base-load Flat / Low Demand
    Plus deliberate edge cases:
    - Missing values (Stage 1 filter)
    - Zero variance / Dead meters (Stage 2 filter)
    - Vacant with sudden >200% jump (Stage 3 filter)

    Returns:
    --------
    raw_df : pd.DataFrame
        Raw dataset including edge cases and optional week 2.
    cleaned_df : pd.DataFrame
        Processed 168-hour matrix after waterfall filtering.
    """
    np.random.seed(random_seed)
    hours = SERIES_HOURS * 2 if include_two_weeks else SERIES_HOURS

    meter_ids = [f"MAC{i+1:06d}" for i in range(n_households)]
    data_matrix = np.zeros((n_households, hours))

    # Base hourly diurnal template (24 hours)
    # Typically low at night (0-6), morning spike (7-9), low midday (10-16), high evening (17-22), taper (23)
    diurnal_template = np.array([
        0.18, 0.15, 0.12, 0.12, 0.14, 0.22, 0.45, 0.85, 0.70, 0.40,
        0.35, 0.38, 0.42, 0.39, 0.37, 0.45, 0.65, 1.10, 1.45, 1.35,
        1.15, 0.85, 0.50, 0.30
    ])

    for i in range(n_households):
        # Assign household behavioral archetype
        archetype = i % 7
        t = np.arange(hours)
        hour_of_day = t % 24
        day_of_week = (t // 24) % 7

        noise = np.random.normal(0, 0.05, hours)

        if archetype == 0:
            # Archetype 0: Standard Evening Peak Working Family
            base = np.tile(diurnal_template, hours // 24 + 1)[:hours]
            # Weekend slightly higher midday
            weekend_boost = np.where((day_of_week >= 5) & (hour_of_day >= 11) & (hour_of_day <= 16), 0.35, 0.0)
            series = base * np.random.uniform(0.8, 1.3) + weekend_boost + noise

        elif archetype == 1:
            # Archetype 1: Dual-Peak Commuters (Strong 7am & 7pm spikes, deep mid-day drop)
            dual_curve = np.zeros(24)
            dual_curve[6:9] = [0.9, 1.4, 0.8]
            dual_curve[17:22] = [0.8, 1.5, 1.6, 1.2, 0.7]
            dual_curve[0:6] = 0.12
            dual_curve[9:17] = 0.20
            base = np.tile(dual_curve, hours // 24 + 1)[:hours]
            series = base * np.random.uniform(0.9, 1.4) + noise

        elif archetype == 2:
            # Archetype 2: Work-from-Home / Retiree (Sustained daytime active 9am-6pm)
            wfh_curve = np.zeros(24)
            wfh_curve[0:7] = 0.15
            wfh_curve[7:19] = np.random.uniform(0.7, 1.0, 12)
            wfh_curve[19:23] = np.random.uniform(0.9, 1.2, 4)
            wfh_curve[23] = 0.25
            base = np.tile(wfh_curve, hours // 24 + 1)[:hours]
            series = base + noise

        elif archetype == 3:
            # Archetype 3: Night Shift / Night Owl (Active 9pm-4am, sleeping daytime)
            night_curve = np.zeros(24)
            night_curve[0:5] = [0.9, 0.8, 0.7, 0.6, 0.3]
            night_curve[5:14] = 0.10
            night_curve[14:20] = 0.35
            night_curve[20:24] = [0.7, 1.1, 1.3, 1.0]
            base = np.tile(night_curve, hours // 24 + 1)[:hours]
            series = base * np.random.uniform(0.8, 1.2) + noise

        elif archetype == 4:
            # Archetype 4: Electric Heating / High Autoregressive Persistence
            # AR(1) process with high coefficient 0.8
            ar_series = np.zeros(hours)
            ar_series[0] = 0.5
            for step in range(1, hours):
                ar_series[step] = 0.75 * ar_series[step - 1] + np.random.normal(0, 0.15)
            series = np.abs(ar_series) + 0.3

        elif archetype == 5:
            # Archetype 5: Low Baserange / Flat Aperiodic (Single person / minimal appliance)
            series = np.random.uniform(0.08, 0.16, hours) + np.random.normal(0, 0.01, hours)

        else:
            # Archetype 6: Irregular Fluctuating / High Volatility
            series = np.random.gamma(shape=2.0, scale=0.3, size=hours)

        data_matrix[i] = np.clip(series, 0.01, 10.0)

    # -----------------------------------------------------------------
    # Inject Deliberate Edge Cases to Showcase the Waterfall Cleaner
    # -----------------------------------------------------------------
    # Edge Case 1: Missing values (Meters 0, 1, 2)
    data_matrix[0, 15:20] = np.nan
    data_matrix[1, 55] = np.nan
    data_matrix[2, 100:110] = np.nan

    # Edge Case 2: Zero variance / flat dead meters (Meters 3, 4, 5)
    data_matrix[3, :] = 0.0  # Zero mean & variance
    data_matrix[4, :] = 0.05  # Constant zero variance
    data_matrix[5, :] = 0.0

    # Edge Case 3: Vacant home with >200% sudden jump (Meters 6, 7, 8)
    if include_two_weeks:
        # Week 1 near vacant (0.05 kWh/h), Week 2 occupied (1.2 kWh/h) -> 2300% jump!
        data_matrix[6, :SERIES_HOURS] = 0.03
        data_matrix[6, SERIES_HOURS:] = 1.10
        data_matrix[7, :SERIES_HOURS] = 0.04
        data_matrix[7, SERIES_HOURS:] = 1.30
        data_matrix[8, :SERIES_HOURS] = 0.02
        data_matrix[8, SERIES_HOURS:] = 0.95

    # Assemble raw DataFrame
    col_names = [f"h_{t+1}" for t in range(hours)]
    raw_df = pd.DataFrame(data_matrix, index=meter_ids, columns=col_names)

    # Clean through WaterfallCleaner
    from backend.core.cleaner import WaterfallCleaner
    cleaner = WaterfallCleaner(series_hours=SERIES_HOURS)
    cleaned_df, _ = cleaner.clean_matrix(raw_df, has_two_weeks=include_two_weeks)

    if save_files:
        raw_path = DATA_RAW_DIR / "sample_london_smartmeter_raw.csv"
        processed_path = DATA_PROCESSED_DIR / "sample_london_smartmeter_cleaned.csv"
        raw_df.to_csv(raw_path, index_label="meter_id")
        cleaned_df.to_csv(processed_path, index_label="meter_id")

    return raw_df, cleaned_df


if __name__ == "__main__":
    raw_df, cleaned_df = generate_london_sample_dataset()
    print(f"Generated raw shape: {raw_df.shape}")
    print(f"Generated cleaned shape: {cleaned_df.shape}")
