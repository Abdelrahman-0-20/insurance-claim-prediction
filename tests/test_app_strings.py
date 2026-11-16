"""Design rule: strictly zero emojis anywhere in the Streamlit app sources."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"

# Genuine emoji / pictograph blocks plus variation selector and ZWJ.
EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),   # emoji blocks (incl. flags 1F1E6-1F1FF)
    (0x1F200, 0x1F2FF),
    (0x2600, 0x27BF),     # misc symbols + dingbats
    (0x2B00, 0x2BFF),     # arrows, stars, shapes used as emoji
    (0x2300, 0x23FF),     # misc technical (watch, hourglass, media controls)
    (0x2934, 0x2935),
    (0xFE0F, 0xFE0F),     # variation selector-16
    (0x200D, 0x200D),     # zero-width joiner
    (0x2049, 0x2049),
    (0x2122, 0x2122),
    (0x2139, 0x2139),
    (0x2194, 0x21AA),
    (0x231A, 0x231B),
    (0x2328, 0x2328),
    (0x23CF, 0x23FA),
    (0x24C2, 0x24C2),
    (0x25AA, 0x25FE),
    (0x2614, 0x2615),
    (0x2648, 0x2653),
    (0x267F, 0x267F),
    (0x2693, 0x2693),
    (0x26A1, 0x26A1),
    (0x26AA, 0x26AB),
    (0x26BD, 0x26BE),
    (0x26C4, 0x26C5),
    (0x26CE, 0x26CE),
    (0x26D4, 0x26D4),
    (0x26EA, 0x26EA),
    (0x26F2, 0x26F3),
    (0x26F5, 0x26F5),
    (0x26FA, 0x26FA),
    (0x26FD, 0x26FD),
    (0x2705, 0x2705),
    (0x270A, 0x270B),
    (0x2728, 0x2728),
    (0x274C, 0x274C),
    (0x274E, 0x274E),
    (0x2753, 0x2755),
    (0x2757, 0x2757),
    (0x2795, 0x2797),
    (0x27B0, 0x27BF),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
)


def _emojis_in(text: str) -> list[str]:
    return sorted({
        ch for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in EMOJI_RANGES)
    })


def test_app_directory_exists():
    assert APP_DIR.exists(), "app/ is missing -- implement app/streamlit_app.py first"
    assert list(APP_DIR.rglob("*.py")), "no Python sources found under app/"


def test_no_emojis_in_app_sources():
    offenders = {}
    for f in sorted(APP_DIR.rglob("*.py")):
        found = _emojis_in(f.read_text())
        if found:
            offenders[str(f.relative_to(APP_DIR))] = found
    assert not offenders, (
        f"emoji characters found in app sources (design rule: zero emojis): {offenders}"
    )
