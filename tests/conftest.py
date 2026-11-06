"""Общие фикстуры прогона на подделках."""

from __future__ import annotations

import pytest

from tests.support import RunHarness, build_harness


@pytest.fixture
def harness() -> RunHarness:
    """Собирает стандартный харнесс.

    Returns:
        RunHarness.
    """
    return build_harness()
