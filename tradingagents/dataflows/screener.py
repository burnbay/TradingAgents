"""
Stock screener that identifies promising candidates across major world exchanges.

Uses yfinance's EquityQuery screener with five strategy presets and configurable
region/exchange filters. Returns ranked candidates ready to feed into TradingAgents.
"""

from typing import Annotated, Optional
import time

from yfinance.screener import screen, EquityQuery

# ---------------------------------------------------------------------------
# Exchange registry — grouped by geographic region
# ---------------------------------------------------------------------------

EXCHANGE_GROUPS: dict[str, dict] = {
    "us": {
        "label": "United States (NYSE / NASDAQ)",
        "codes": ["NMS", "NYQ", "ASE"],
    },
    "canada": {
        "label": "Canada (TSX / NEO)",
        "codes": ["TOR", "NEO"],
    },
    "uk": {
        "label": "United Kingdom (LSE)",
        "codes": ["LSE"],
    },
    "germany": {
        "label": "Germany (Frankfurt / Xetra)",
        "codes": ["FRA", "GER"],
    },
    "france": {
        "label": "France (Euronext Paris)",
        "codes": ["PAR"],
    },
    "oslo": {
        "label": "Norway (Oslo Stock Exchange)",
        "codes": ["OSL"],
    },
    "stockholm": {
        "label": "Sweden (Nasdaq Stockholm)",
        "codes": ["STO"],
    },
    "copenhagen": {
        "label": "Denmark (Nasdaq Copenhagen)",
        "codes": ["CPH"],
    },
    "helsinki": {
        "label": "Finland (Nasdaq Helsinki)",
        "codes": ["HEL"],
    },
    "japan": {
        "label": "Japan (JPX / Tokyo)",
        "codes": ["JPX"],
    },
    "hongkong": {
        "label": "Hong Kong (HKEX)",
        "codes": ["HKG"],
    },
    "australia": {
        "label": "Australia (ASX)",
        "codes": ["ASX"],
    },
    "india": {
        "label": "India (NSE / BSE)",
        "codes": ["NSI", "BSE"],
    },
    "korea": {
        "label": "South Korea (KRX)",
        "codes": ["KSC"],
    },
    "singapore": {
        "label": "Singapore (SGX)",
        "codes": ["SES"],
    },
    "brazil": {
        "label": "Brazil (B3)",
        "codes": ["SAO"],
    },
}

# Convenience bundles
REGION_BUNDLES: dict[str, list[str]] = {
    "americas": ["us", "canada", "brazil"],
    "europe": ["uk", "germany", "france", "oslo", "stockholm", "copenhagen", "helsinki"],
    "nordics": ["oslo", "stockholm", "copenhagen", "helsinki"],
    "asia_pacific": ["japan", "hongkong", "australia", "india", "korea", "singapore"],
    "global": list(EXCHANGE_GROUPS.keys()),
}

# ---------------------------------------------------------------------------
# Screening presets
# ---------------------------------------------------------------------------

def _growth_query(exchanges: list[str]) -> EquityQuery:
    """High revenue and EPS growth at a reasonable valuation."""
    return EquityQuery("and", [
        EquityQuery("is-in", ["exchange"] + exchanges),
        EquityQuery("gt", ["totalrevenues1yrgrowth.lasttwelvemonths", 15]),
        EquityQuery("gt", ["epsgrowth.lasttwelvemonths", 10]),
        EquityQuery("gt", ["peratio.lasttwelvemonths", 0]),
        EquityQuery("lt", ["peratio.lasttwelvemonths", 50]),
        EquityQuery("gt", ["intradaymarketcap", 500_000_000]),
    ])


def _value_query(exchanges: list[str]) -> EquityQuery:
    """Undervalued profitable companies with low P/E and solid ROE."""
    return EquityQuery("and", [
        EquityQuery("is-in", ["exchange"] + exchanges),
        EquityQuery("gt", ["peratio.lasttwelvemonths", 3]),
        EquityQuery("lt", ["peratio.lasttwelvemonths", 18]),
        EquityQuery("lt", ["pricebookratio.quarterly", 3]),
        EquityQuery("gt", ["returnonequity.lasttwelvemonths", 10]),
        EquityQuery("gt", ["intradaymarketcap", 1_000_000_000]),
    ])


def _momentum_query(exchanges: list[str]) -> EquityQuery:
    """Strong 52-week price momentum — stocks near recent highs."""
    return EquityQuery("and", [
        EquityQuery("is-in", ["exchange"] + exchanges),
        EquityQuery("gt", ["fiftytwowkpercentchange", 20]),
        EquityQuery("gt", ["intradaymarketcap", 1_000_000_000]),
        EquityQuery("gt", ["avgdailyvol3m", 100_000]),
    ])


def _quality_query(exchanges: list[str]) -> EquityQuery:
    """Financially strong companies: high ROE, fat margins, low leverage."""
    return EquityQuery("and", [
        EquityQuery("is-in", ["exchange"] + exchanges),
        EquityQuery("gt", ["returnonequity.lasttwelvemonths", 20]),
        EquityQuery("gt", ["netincomemargin.lasttwelvemonths", 15]),
        EquityQuery("lt", ["totaldebtequity.lasttwelvemonths", 100]),
        EquityQuery("gt", ["intradaymarketcap", 2_000_000_000]),
    ])


def _dividend_query(exchanges: list[str]) -> EquityQuery:
    """Dividend growers with attractive yield and market cap stability."""
    return EquityQuery("and", [
        EquityQuery("is-in", ["exchange"] + exchanges),
        EquityQuery("gt", ["forward_dividend_yield", 2.5]),
        EquityQuery("gt", ["consecutive_years_of_dividend_growth_count", 3]),
        EquityQuery("gt", ["intradaymarketcap", 1_000_000_000]),
    ])


SCREENING_PRESETS: dict[str, dict] = {
    "growth": {
        "label": "Growth — high revenue & EPS growth, reasonable P/E",
        "query_fn": _growth_query,
        "sort_field": "totalrevenues1yrgrowth.lasttwelvemonths",
    },
    "value": {
        "label": "Value — low P/E, low P/B, strong ROE",
        "query_fn": _value_query,
        "sort_field": "peratio.lasttwelvemonths",
        "sort_asc": True,
    },
    "momentum": {
        "label": "Momentum — strongest 52-week price performance",
        "query_fn": _momentum_query,
        "sort_field": "fiftytwowkpercentchange",
    },
    "quality": {
        "label": "Quality — high ROE & margins, low leverage",
        "query_fn": _quality_query,
        "sort_field": "returnonequity.lasttwelvemonths",
    },
    "dividend": {
        "label": "Dividend — attractive yield with consecutive growth",
        "query_fn": _dividend_query,
        "sort_field": "forward_dividend_yield",
    },
}

# ---------------------------------------------------------------------------
# Core screening function
# ---------------------------------------------------------------------------

def _resolve_exchanges(region_keys: list[str]) -> list[str]:
    """Flatten region keys (groups or bundles) into exchange codes."""
    codes: list[str] = []
    seen: set[str] = set()
    for key in region_keys:
        if key in REGION_BUNDLES:
            for sub in REGION_BUNDLES[key]:
                for code in EXCHANGE_GROUPS[sub]["codes"]:
                    if code not in seen:
                        codes.append(code)
                        seen.add(code)
        elif key in EXCHANGE_GROUPS:
            for code in EXCHANGE_GROUPS[key]["codes"]:
                if code not in seen:
                    codes.append(code)
                    seen.add(code)
    return codes


def screen_stocks(
    regions: Annotated[list[str], "region keys, exchange group keys, or bundle names"],
    criteria: Annotated[str, "screening preset: growth|value|momentum|quality|dividend"] = "growth",
    count: Annotated[int, "max results to return"] = 20,
    min_market_cap: Annotated[float, "minimum market cap in USD"] = 500_000_000,
) -> list[dict]:
    """
    Screen stocks from major world exchanges using the selected strategy preset.

    Returns a list of candidate dicts sorted by the preset's primary metric.
    Each dict includes: symbol, name, exchange, market_cap, price, change_pct,
    pe_ratio, revenue_growth, roe, fiftytwo_wk_change.
    """
    if criteria not in SCREENING_PRESETS:
        raise ValueError(f"Unknown criteria '{criteria}'. Choose from: {list(SCREENING_PRESETS)}")

    exchanges = _resolve_exchanges(regions)
    if not exchanges:
        raise ValueError(f"No exchanges resolved from regions: {regions}")

    preset = SCREENING_PRESETS[criteria]
    query = preset["query_fn"](exchanges)
    sort_field = preset["sort_field"]
    sort_asc = preset.get("sort_asc", False)

    # yfinance screener caps at 250 per call; fetch in one shot up to that limit
    fetch_count = min(count, 250)

    result = screen(
        query,
        count=fetch_count,
        sortField=sort_field,
        sortAsc=sort_asc,
    )

    quotes = result.get("quotes", [])
    candidates = []
    seen_symbols: set[str] = set()

    for q in quotes:
        symbol = q.get("symbol", "")
        if not symbol or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)

        market_cap = q.get("marketCap") or q.get("intradayMarketCap", 0)
        if market_cap and market_cap < min_market_cap:
            continue

        candidates.append({
            "symbol": symbol,
            "name": q.get("shortName") or q.get("longName", ""),
            "exchange": q.get("exchange", ""),
            "market_cap": market_cap,
            "price": q.get("regularMarketPrice"),
            "currency": q.get("currency", ""),
            "change_pct": q.get("regularMarketChangePercent"),
            "pe_ratio": q.get("trailingPE") or q.get("forwardPE"),
            "revenue_growth": q.get("revenueGrowth"),
            "roe": q.get("returnOnEquity"),
            "fiftytwo_wk_change": q.get("52WeekChange") or q.get("fiftyTwoWeekChangePercent"),
        })

        if len(candidates) >= count:
            break

    return candidates


def get_available_regions() -> dict[str, str]:
    """Return all valid region keys with their human-readable labels."""
    result = {}
    for key, info in EXCHANGE_GROUPS.items():
        result[key] = info["label"]
    for key, sub_keys in REGION_BUNDLES.items():
        labels = ", ".join(EXCHANGE_GROUPS[k]["label"].split("(")[0].strip() for k in sub_keys)
        result[key] = f"Bundle: {labels}"
    return result


def get_available_criteria() -> dict[str, str]:
    """Return all valid criteria keys with their human-readable labels."""
    return {key: info["label"] for key, info in SCREENING_PRESETS.items()}
