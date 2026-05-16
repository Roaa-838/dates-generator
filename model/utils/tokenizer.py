"""
Custom tokenizer for conditional date generation.

Vocabulary design:
  - Input conditions: day-of-week (7), month (12), leap (2), decade (41) → each has its own embedding.
  - Output tokens:    digits 0-9 (10 tokens), separator '-' (1 token), special tokens PAD/SOS/EOS (3).

Token order for OUTPUT (year-first, most constrained first):
  SOS  Y1 Y2 Y3 Y4  -  M1 M2  -  D1 D2  EOS
  e.g. "3-12-1962" → [SOS, 1,9,6,2, SEP, 1,2, SEP, 3, EOS]
  This lets the model resolve the decade condition first (year), then month, then day-of-week.
"""

from __future__ import annotations

import re
from typing import Optional


# ────────────────────────────────────────────────
# Condition vocabularies
# ────────────────────────────────────────────────

DAY_TOKENS: list[str] = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_TOKENS: list[str] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]
LEAP_TOKENS: list[str] = ["False", "True"]
# Decade codes 180 → 1800-1809 … 220 → 2200-2209 (but we clamp to 2200)
DECADE_TOKENS: list[str] = [str(d) for d in range(180, 221)]  # 41 values

# Output-sequence special tokens
PAD_ID: int = 11
SOS_ID: int = 12
EOS_ID: int = 13
SEP_ID: int = 10   # the '-' character

# digit ids 0-9 map directly to their integer value
OUTPUT_VOCAB_SIZE: int = 14   # 0-9, SEP(10), PAD(11), SOS(12), EOS(13)

MAX_OUTPUT_LEN: int = 12   # SOS + Y1 Y2 Y3 Y4 SEP M1 M2 SEP D1 D2 + EOS


class DateTokenizer:
    """
    Encodes condition strings and date strings into integer token IDs,
    and decodes token ID sequences back into date strings.

    Attributes
    ----------
    day2id : dict mapping day abbreviation → int (0-6)
    month2id : dict mapping month abbreviation → int (0-11)
    leap2id : dict mapping 'False'/'True' → int (0-1)
    decade2id : dict mapping decade string (e.g. '196') → int (0-40)
    """

    # Input vocab sizes (used by embedding layers in models)
    DAY_VOCAB: int = len(DAY_TOKENS)       # 7
    MONTH_VOCAB: int = len(MONTH_TOKENS)   # 12
    LEAP_VOCAB: int = len(LEAP_TOKENS)     # 2
    DECADE_VOCAB: int = len(DECADE_TOKENS) # 41
    OUTPUT_VOCAB: int = OUTPUT_VOCAB_SIZE  # 14
    MAX_OUTPUT_LEN: int = MAX_OUTPUT_LEN   # 12

    # Special IDs (expose as class attributes for models to reference)
    PAD_ID: int = PAD_ID
    SOS_ID: int = SOS_ID
    EOS_ID: int = EOS_ID
    SEP_ID: int = SEP_ID

    def __init__(self) -> None:
        self.day2id: dict[str, int] = {tok: i for i, tok in enumerate(DAY_TOKENS)}
        self.id2day: dict[int, str] = {i: tok for tok, i in self.day2id.items()}

        self.month2id: dict[str, int] = {tok: i for i, tok in enumerate(MONTH_TOKENS)}
        self.id2month: dict[int, str] = {i: tok for tok, i in self.month2id.items()}

        self.leap2id: dict[str, int] = {tok: i for i, tok in enumerate(LEAP_TOKENS)}
        self.id2leap: dict[int, str] = {i: tok for tok, i in self.leap2id.items()}

        self.decade2id: dict[str, int] = {tok: i for i, tok in enumerate(DECADE_TOKENS)}
        self.id2decade: dict[int, str] = {i: tok for tok, i in self.decade2id.items()}

    # ─── Input encoding ──────────────────────────────────────────────────────

    def encode_input(self, line: str) -> tuple[int, int, int, int]:
        """
        Parse a condition line (with or without a trailing date) and return
        four integer IDs: (day_id, month_id, leap_id, decade_id).

        Parameters
        ----------
        line : str
            e.g. '[WED] [JAN] [False] [196]' or '[WED] [JAN] [False] [196] 3-12-1962'

        Returns
        -------
        tuple[int, int, int, int]
            (day_id, month_id, leap_id, decade_id)

        Raises
        ------
        ValueError
            If any condition token is unrecognised.
        """
        tokens = re.findall(r'\[([^\]]+)\]', line)
        if len(tokens) < 4:
            raise ValueError(f"Expected 4 bracketed tokens, got {len(tokens)} in: {line!r}")
        day_str, month_str, leap_str, decade_str = tokens[:4]

        if day_str not in self.day2id:
            raise ValueError(f"Unknown day token: {day_str!r}")
        if month_str not in self.month2id:
            raise ValueError(f"Unknown month token: {month_str!r}")
        if leap_str not in self.leap2id:
            raise ValueError(f"Unknown leap token: {leap_str!r}")
        if decade_str not in self.decade2id:
            raise ValueError(f"Unknown decade token: {decade_str!r}")

        return (
            self.day2id[day_str],
            self.month2id[month_str],
            self.leap2id[leap_str],
            self.decade2id[decade_str],
        )

    # ─── Output encoding ─────────────────────────────────────────────────────

    def encode_output(self, date_str: str) -> list[int]:
        """
        Encode a date string into a token-id sequence (YEAR-FIRST order).

        Token order: SOS Y1 Y2 Y3 Y4 SEP M1 M2 SEP D1 D2 EOS
        All digit characters → their integer value (0-9).
        '-' separator → SEP_ID (10).

        Parameters
        ----------
        date_str : str
            e.g. '3-12-1962'  (d-m-yyyy, no leading zeros)

        Returns
        -------
        list[int]
            Token ids including SOS and EOS.

        Raises
        ------
        ValueError
            If the date string is not in expected format.
        """
        parts = date_str.strip().split('-')
        if len(parts) != 3:
            raise ValueError(f"Bad date format (expected d-m-yyyy): {date_str!r}")
        day_s, month_s, year_s = parts

        if len(year_s) != 4:
            raise ValueError(f"Year must be 4 digits, got: {year_s!r}")

        ids: list[int] = [SOS_ID]
        # Year digits first
        for ch in year_s:
            ids.append(int(ch))
        ids.append(SEP_ID)
        # Month (zero-padded to 2 digits for consistent sequence length)
        month_padded = month_s.zfill(2)
        for ch in month_padded:
            ids.append(int(ch))
        ids.append(SEP_ID)
        # Day (zero-padded to 2 digits)
        day_padded = day_s.zfill(2)
        for ch in day_padded:
            ids.append(int(ch))
        ids.append(EOS_ID)
        return ids  # length = 12

    def decode_output(self, token_ids: list[int]) -> Optional[str]:
        """
        Decode a list of token ids back to a date string (d-m-yyyy, no leading zeros).

        Strips SOS/EOS/PAD tokens, then reconstructs year, month, day.

        Parameters
        ----------
        token_ids : list[int]
            Raw ids from model output (may include SOS, EOS, PAD).

        Returns
        -------
        str or None
            Decoded date string, or None if the sequence cannot be parsed.
        """
        # Remove special tokens
        clean: list[int] = [
            t for t in token_ids if t not in (SOS_ID, EOS_ID, PAD_ID)
        ]
        # Expected layout after stripping: Y1 Y2 Y3 Y4 SEP M1 M2 SEP D1 D2
        try:
            sep_positions = [i for i, t in enumerate(clean) if t == SEP_ID]
            if len(sep_positions) < 2:
                return None
            s1, s2 = sep_positions[0], sep_positions[1]
            year_digits = clean[:s1]
            month_digits = clean[s1 + 1:s2]
            day_digits = clean[s2 + 1:]

            year_str = "".join(str(d) for d in year_digits)
            month_str = "".join(str(d) for d in month_digits).lstrip("0") or "0"
            day_str = "".join(str(d) for d in day_digits).lstrip("0") or "0"

            if len(year_str) != 4:
                return None
            return f"{day_str}-{month_str}-{year_str}"
        except Exception:
            return None

    # ─── Condition string reconstruction ─────────────────────────────────────

    def decode_input(
        self,
        day_id: int,
        month_id: int,
        leap_id: int,
        decade_id: int,
    ) -> str:
        """
        Reconstruct the bracketed condition string from integer IDs.

        Parameters
        ----------
        day_id, month_id, leap_id, decade_id : int

        Returns
        -------
        str
            e.g. '[WED] [JAN] [False] [196]'
        """
        return (
            f"[{self.id2day[day_id]}] "
            f"[{self.id2month[month_id]}] "
            f"[{self.id2leap[leap_id]}] "
            f"[{self.id2decade[decade_id]}]"
        )