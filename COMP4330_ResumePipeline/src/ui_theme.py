"""
Runtime colour themes for the recruiter dashboard.

Streamlit's own theming is configured once in `.streamlit/config.toml` and
cannot change while the app is running. To let a user pick a colour scheme
at runtime, this module renders a small CSS override scoped to the app root.

Every theme is a set of hex colours. `build_theme_css` turns one into a
`<style>` block that targets stable Streamlit test IDs (never private class
names), so the overrides survive Streamlit upgrades.

Colours are chosen to keep WCAG AA contrast (body text >= 4.5:1) and to keep
white text on the primary button.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------
# Each palette carries:
#   bg        - page background
#   panel     - card / panel background
#   surface   - widget backgrounds (inputs, dropzones)
#   text      - body text
#   heading   - heading text
#   muted     - captions and secondary text
#   border    - card and widget borders
#   primary   - accent (buttons, links, progress)
#   onPrimary - text on top of the accent
#   success / warning / danger - status colours for badges and alerts


class ThemePalette:
    """One colour scheme."""

    def __init__(
        self,
        key: str,
        label: str,
        label_zh: str,
        bg: str,
        panel: str,
        surface: str,
        text: str,
        heading: str,
        muted: str,
        border: str,
        primary: str,
        on_primary: str,
        success: str,
        warning: str,
        danger: str,
        is_dark: bool = False,
    ) -> None:
        self.key = key
        self.label = label
        self.label_zh = label_zh
        self.bg = bg
        self.panel = panel
        self.surface = surface
        self.text = text
        self.heading = heading
        self.muted = muted
        self.border = border
        self.primary = primary
        self.on_primary = on_primary
        self.success = success
        self.warning = warning
        self.danger = danger
        self.is_dark = is_dark


THEMES: Dict[str, ThemePalette] = {
    "blue": ThemePalette(
        key="blue",
        label="Blue (default)",
        label_zh="藍色（預設）",
        bg="#FFFFFF",
        panel="#F6F7F9",
        surface="#FFFFFF",
        text="#1F2328",
        heading="#111418",
        muted="#5B6470",
        border="#E3E6EA",
        primary="#2563EB",
        on_primary="#FFFFFF",
        success="#15803D",
        warning="#B45309",
        danger="#DC2626",
    ),
    "teal": ThemePalette(
        key="teal",
        label="Teal",
        label_zh="青綠色",
        bg="#FFFFFF",
        panel="#F1F7F6",
        surface="#FFFFFF",
        text="#1B2A2E",
        heading="#0F1F23",
        muted="#52686D",
        border="#D6E6E4",
        primary="#0F766E",
        on_primary="#FFFFFF",
        success="#0F766E",
        warning="#B45309",
        danger="#DC2626",
    ),
    "purple": ThemePalette(
        key="purple",
        label="Purple",
        label_zh="紫色",
        bg="#FFFFFF",
        panel="#F7F5FC",
        surface="#FFFFFF",
        text="#241B33",
        heading="#160F22",
        muted="#635A75",
        border="#E4DEF2",
        primary="#6D28D9",
        on_primary="#FFFFFF",
        success="#15803D",
        warning="#B45309",
        danger="#DC2626",
    ),
    "forest": ThemePalette(
        key="forest",
        label="Forest",
        label_zh="墨綠色",
        bg="#FFFFFF",
        panel="#F3F8F4",
        surface="#FFFFFF",
        text="#16261C",
        heading="#0D1B13",
        muted="#4E6357",
        border="#D8E7DC",
        primary="#15803D",
        on_primary="#FFFFFF",
        success="#15803D",
        warning="#B45309",
        danger="#DC2626",
    ),
    "sunset": ThemePalette(
        key="sunset",
        label="Sunset",
        label_zh="夕陽色",
        bg="#FFFFFF",
        panel="#FFF8F3",
        surface="#FFFFFF",
        text="#2C1F18",
        heading="#1C120D",
        muted="#6E5A4E",
        border="#F2E0D4",
        primary="#C2410C",
        on_primary="#FFFFFF",
        success="#15803D",
        warning="#B45309",
        danger="#DC2626",
    ),
    "midnight": ThemePalette(
        key="midnight",
        label="Midnight (dark)",
        label_zh="午夜（深色）",
        bg="#12161C",
        panel="#1A2029",
        surface="#222932",
        text="#E6EAF0",
        heading="#F5F8FC",
        muted="#98A4B3",
        border="#2C3542",
        primary="#5B9DFF",
        on_primary="#0B1017",
        success="#4ADE80",
        warning="#FBBF24",
        danger="#F87171",
        is_dark=True,
    ),
}

DEFAULT_THEME = "blue"


def get_theme(key: str) -> ThemePalette:
    """Returns a palette, falling back to the default for unknown keys."""
    return THEMES.get(key, THEMES[DEFAULT_THEME])


def theme_options(lang: str = "en") -> List[str]:
    """Theme keys in display order, for a selectbox."""
    return list(THEMES.keys())


def theme_label(key: str, lang: str = "en") -> str:
    """Human-readable name of a theme in the chosen language."""
    palette = get_theme(key)
    if lang == "zh":
        return palette.label_zh
    return palette.label


# ---------------------------------------------------------------------------
# CSS generation
# ---------------------------------------------------------------------------

# The sidebar is toggled off on narrow viewports, so the same background is
# applied to both the sidebar and the main area to keep the page coherent.
_SIDEBAR_SELECTORS = (
    '[data-testid="stSidebar"]',
    '[data-testid="stSidebarContent"]',
    '[data-testid="stSidebarUserContent"]',
)

# Inputs that Streamlit renders with their own border. The visible control is
# the wrapper div, so both the wrapper and the inner field are recoloured.
_INPUT_SELECTORS = (
    '[data-testid="stTextInput"] input',
    '[data-testid="stTextArea"] textarea',
    '[data-testid="stFileUploaderDropzone"]',
    '[data-testid="stSelectbox"] div[data-rac="group"]',
    '[data-testid="stCheckbox"] label',
)


def build_theme_css(key: str) -> str:
    """
    Builds the CSS override for one theme.

    Only stable `data-testid` selectors and Streamlit's own `key`-derived
    `.st-key-*` classes are used, so the styling keeps working across
    Streamlit upgrades.
    """
    p = get_theme(key)

    rules: List[str] = [
        # Page and main surfaces
        f'[data-testid="stAppViewContainer"] {{ background-color: {p.bg}; }}',
        f'[data-testid="stMain"] {{ background-color: {p.bg}; }}',
        f'[data-testid="stMainBlockContainer"] {{ background-color: {p.bg}; }}',
        # Sidebar shares the page surface, tinted one step for separation
        *[f"{sel} {{ background-color: {p.panel}; }}" for sel in _SIDEBAR_SELECTORS],
        # Cards, panels and expanders
        f'.stVerticalBlock.st-key-card_job, .stVerticalBlock.st-key-card_upload, '
        f'.stVerticalBlock.st-key-card_shortlist {{ background-color: {p.panel}; border-color: {p.border}; }}',
        f'[data-testid="stExpander"] {{ background-color: {p.panel}; border-color: {p.border}; }}',
        f'[data-testid="stExpander"] summary {{ background-color: {p.panel}; }}',
        f'[data-testid="stExpander"] summary:hover {{ background-color: {p.surface}; }}',
        # Primary button
        f'[data-testid="stBaseButton-primary"] {{ background-color: {p.primary} !important; '
        f"border-color: {p.primary} !important; color: {p.on_primary} !important; }}",
        f'[data-testid="stBaseButton-primary"]:hover {{ filter: brightness(0.93); }}',
        # Secondary buttons and links
        f'[data-testid="stBaseButton-secondary"] {{ background-color: {p.surface} !important; '
        f"border-color: {p.border} !important; color: {p.text} !important; }}",
        f'a, [data-testid="stLink"] {{ color: {p.primary}; }}',
        # Typography.
        # Streamlit's emotion classes set heading colour with a more specific
        # selector, so `!important` is needed for the override to win.
        f'h1, h2, h3, h4, h5, h6 {{ color: {p.heading} !important; }}',
        f'[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {{ color: {p.text} !important; }}',
        f'[data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {{ color: {p.muted} !important; }}',
        # Widgets
        *[f"{sel} {{ background-color: {p.surface}; color: {p.text}; border-color: {p.border}; }}"
          for sel in _INPUT_SELECTORS],
        f'[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{ color: {p.text} !important; }}',
        f'[data-testid="stSelectbox"] input {{ color: {p.text} !important; }}',
        # Dividers and dataframes
        f'hr {{ border-color: {p.border}; }}',
        f'[data-testid="stDataFrame"] {{ border-color: {p.border}; }}',
        f'[data-testid="stDataFrame"] * {{ color: {p.text}; }}',
        # Status colours
        f'[data-testid="stException"] {{ border-color: {p.danger}; }}',
    ]

    if p.is_dark:
        # Streamlit's markdown helpers and alerts are styled for a light
        # surface, so they need explicit overrides on a dark background.
        rules.append(
            f'[data-testid="stMarkdownContainer"] p {{ color: {p.text} !important; }}'
        )
        rules.append(
            f'[data-testid="stAlert"] {{ background-color: {p.surface} !important; color: {p.text} !important; }}'
        )
        rules.append(
            f'[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {p.text} !important; }}'
        )
        rules.append(
            f'[data-testid="stDataFrame"] * {{ color: {p.text} !important; }}'
        )

    body = "\n".join(rules)
    return f"<style>\n{body}\n</style>"


def build_theme_css_all() -> str:
    """CSS that keeps Streamlit's default look when no theme override applies."""
    return build_theme_css(DEFAULT_THEME)
