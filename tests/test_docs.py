"""Run the examples in the docstrings of the installed package."""
import doctest

import pytest

import wageningen_bseries
from wageningen_bseries import polynomials, validity


@pytest.mark.parametrize("module", [wageningen_bseries, polynomials, validity])
def test_docstring_examples(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0
