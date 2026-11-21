"""Target: carbon intensity CI(t) = sum(gen_i * ef_i) / sum(gen_i)."""

import pandas as pd

from .emissions import LIFECYCLE_AR5_G_PER_KWH


def carbon_intensity(gen_wide, factors=LIFECYCLE_AR5_G_PER_KWH):
    """Compute hourly carbon intensity in gCO2eq/kWh.

    Parameters
    ----------
    gen_wide : DataFrame
        Index is a UTC DatetimeIndex. Columns are EIA-930 fuel codes.
        Values are hourly average generation in MW. NaN means missing.
    factors : dict
        Fuel code -> gCO2eq/kWh. Fuels not in this dict are excluded.

    Returns
    -------
    DataFrame with columns:
        ci_g_per_kwh : CI(t). NaN when no mapped, positive generation exists.
        mapped_share : share of positive generation covered by mapped fuels.

    Negative generation (storage charging or consumption) is not generation,
    so it is clipped to zero and drops out of both numerator and denominator.
    """
    mapped = [c for c in gen_wide.columns if c in factors]
    gen = gen_wide[mapped].clip(lower=0)
    ef = pd.Series({c: factors[c] for c in mapped})

    numerator = (gen * ef).sum(axis=1, min_count=1)
    denominator = gen.sum(axis=1, min_count=1)
    ci = (numerator / denominator).where(denominator > 0)

    total_pos = gen_wide.clip(lower=0).sum(axis=1, min_count=1)
    mapped_pos = gen.sum(axis=1, min_count=1)
    mapped_share = (mapped_pos / total_pos).where(total_pos > 0)

    return pd.DataFrame(
        {"ci_g_per_kwh": ci, "mapped_share": mapped_share},
        index=gen_wide.index,
    )
