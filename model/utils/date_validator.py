"""
Date validation utilities.

IMPORTANT: Always use calendar.isleap() for leap-year checks.
Years 1800, 1900, 2100, 2200 are NOT leap years (divisible by 100 but not 400).
Never implement custom leap-year logic.
"""

from __future__ import annotations

import calendar
from datetime import date


# Day-of-week abbreviations aligned with datetime.weekday() (0 = Monday)
_DOW: list[str] = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# Month abbreviations aligned with datetime month numbers (1-indexed)
_MONTHS: list[str] = [
    "", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


def is_valid_date(day: int, month: int, year: int) -> bool:
    """
    Return True if (day, month, year) is a valid Gregorian date
    within the assignment range [1-1-1800 .. 31-12-2200].

    Parameters
    ----------
    day, month, year : int

    Returns
    -------
    bool
    """
    if year < 1800 or year > 2200:
        return False
    try:
        date(year, month, day)
        return True
    except (ValueError, OverflowError):
        return False


def day_of_week(day: int, month: int, year: int) -> str:
    """
    Return the three-letter day-of-week abbreviation for a valid date.

    Parameters
    ----------
    day, month, year : int

    Returns
    -------
    str
        One of 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'.

    Raises
    ------
    ValueError
        If the date is invalid.
    """
    return _DOW[date(year, month, day).weekday()]


def month_abbr(month: int) -> str:
    """
    Return the three-letter month abbreviation for a month number 1-12.

    Parameters
    ----------
    month : int  (1 = January, 12 = December)

    Returns
    -------
    str
    """
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month number: {month}")
    return _MONTHS[month]


def decade_code(year: int) -> str:
    """
    Return the decade code string for a year.

    e.g. 1962 → '196', 2048 → '204'

    Parameters
    ----------
    year : int

    Returns
    -------
    str
    """
    return str(year // 10)


def check_conditions(
    day: int,
    month: int,
    year: int,
    day_cond: str,
    month_cond: str,
    leap_cond: str,
    decade_cond: str,
) -> dict[str, bool]:
    """
    Evaluate all four assignment conditions plus basic date validity.

    Parameters
    ----------
    day, month, year : int
        The generated date components.
    day_cond : str
        Expected day abbreviation, e.g. 'WED'.
    month_cond : str
        Expected month abbreviation, e.g. 'JAN'.
    leap_cond : str
        Expected leap-year status as string, 'True' or 'False'.
    decade_cond : str
        Expected decade code, e.g. '196' (means 1960-1969).

    Returns
    -------
    dict[str, bool]
        Keys: 'valid', 'day', 'month', 'leap', 'decade', 'all_pass'.
    """
    valid = is_valid_date(day, month, year)
    if not valid:
        return {
            "valid": False,
            "day": False,
            "month": False,
            "leap": False,
            "decade": False,
            "all_pass": False,
        }

    day_ok = day_of_week(day, month, year) == day_cond
    month_ok = month_abbr(month) == month_cond
    leap_ok = str(calendar.isleap(year)) == leap_cond
    decade_ok = decade_code(year) == decade_cond

    return {
        "valid": True,
        "day": day_ok,
        "month": month_ok,
        "leap": leap_ok,
        "decade": decade_ok,
        "all_pass": day_ok and month_ok and leap_ok and decade_ok,
    }


def parse_date_string(date_str: str) -> tuple[int, int, int] | None:
    """
    Parse a date string of format 'd-m-yyyy' into (day, month, year).

    Parameters
    ----------
    date_str : str
        e.g. '3-12-1962'

    Returns
    -------
    tuple[int, int, int] or None
        (day, month, year) or None if unparseable.
    """
    try:
        parts = date_str.strip().split("-")
        if len(parts) != 3:
            return None
        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        return (d, m, y)
    except (ValueError, AttributeError):
        return None


def fallback_date(
    day_cond: str,
    month_cond: str,
    leap_cond: str,
    decade_cond: str,
) -> str:
    """
    Rule-based fallback that always produces a valid, condition-satisfying date.
    Used when the neural model fails to generate a valid output.

    Strategy:
      1. Determine target decade → pick years in that range.
      2. Filter by leap condition using calendar.isleap().
      3. Pick the target month.
      4. Iterate over days 1-28 to find one that matches day-of-week.
         (Days 1-28 are always valid for any month.)

    Parameters
    ----------
    day_cond, month_cond, leap_cond, decade_cond : str
        The four condition strings (without brackets).

    Returns
    -------
    str
        A valid date string 'd-m-yyyy', or '1-1-1800' as ultimate fallback.
    """
    _DOW_MAP: dict[str, int] = {
        "MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6,
    }
    _MONTH_MAP: dict[str, int] = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    target_dow = _DOW_MAP.get(day_cond, 0)
    target_month = _MONTH_MAP.get(month_cond, 1)
    want_leap = leap_cond == "True"

    try:
        decade_num = int(decade_cond)
    except ValueError:
        return "1-1-1800"

    decade_start = decade_num * 10
    decade_end = min(decade_start + 9, 2200)

    for year in range(decade_start, decade_end + 1):
        if year < 1800 or year > 2200:
            continue
        if calendar.isleap(year) != want_leap:
            continue
        for day in range(1, 29):  # 1-28 always safe
            try:
                d = date(year, target_month, day)
                if d.weekday() == target_dow:
                    return f"{day}-{target_month}-{year}"
            except ValueError:
                continue

    return "1-1-1800"