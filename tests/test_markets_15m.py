from kalshi_bot.data.kalshi_client import normalize_market
from kalshi_bot.data.markets_15m import (
    default_crypto_15m_tickers,
    get_series_spec,
    parse_series_ticker,
    resolve_enabled_series,
)


def test_parse_series_ticker_eth():
    assert parse_series_ticker("KXETH15M-26AUG100345") == "KXETH15M"
    assert parse_series_ticker("KXBTC15M-26AUG070730") == "KXBTC15M"
    assert parse_series_ticker("KXBTCD-26AUG0714") is None


def test_normalize_eth_15m():
    raw = {
        "ticker": "KXETH15M-26AUG100345-45",
        "event_ticker": "KXETH15M-26AUG100345",
        "floor_strike": 1927.89,
        "close_time": "2026-08-10T07:45:00Z",
        "yes_bid_dollars": "0.5600",
        "yes_ask_dollars": "0.5700",
    }
    m = normalize_market(raw)
    assert m["series_ticker"] == "KXETH15M"
    assert m["strike"] == 1927.89


def test_series_spec_eth():
    spec = get_series_spec("KXETH15M")
    assert spec is not None
    assert spec.asset == "ETH"
    assert spec.cf_benchmarks_index == "ETHUSD_RTI"
    assert spec.kraken_pair == "ETHUSD"


def test_default_crypto_series():
    tickers = default_crypto_15m_tickers()
    assert "KXBTC15M" in tickers
    assert "KXETH15M" in tickers
    assert "KXSOL15M" in tickers
    assert len(tickers) >= 7


def test_resolve_enabled_series_configured():
    specs = resolve_enabled_series(configured=["KXETH15M", "KXSOL15M"])
    assert len(specs) == 2
    assert {s.ticker for s in specs} == {"KXETH15M", "KXSOL15M"}


def test_resolve_enabled_series_auto_discover():
    specs = resolve_enabled_series(auto_discover=True, include_head_to_head=False)
    tickers = {s.ticker for s in specs}
    assert "KXBTC15M" in tickers
    assert "KXGOLD15M" in tickers
    assert "KXCRYPTOCOMP15M" not in tickers
