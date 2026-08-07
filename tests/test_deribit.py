"""Unit tests for Deribit smile loader (network)."""

import pytest

from kalshi_bot.data.deribit_options import load_deribit_btc_smile


@pytest.mark.integration
def test_deribit_smile_live():
    smile = load_deribit_btc_smile()
    assert len(smile.points) >= 4
    assert 0.2 < smile.atm_iv < 2.0
    assert smile.spot_btc > 1000
