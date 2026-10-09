from __future__ import annotations

import pytest

from finagent.schemas import FinQAExample


@pytest.fixture
def finqa_example() -> FinQAExample:
    return FinQAExample(
        id="ETR/2016/page_23.pdf-2",
        question="what is the net change in net revenue during 2015?",
        answer="94",
        exe_ans=94.0,
        program="subtract(5829, 5735)",
        pre_text=["net revenue increased in 2015."],
        post_text=["see note 2."],
        table=[["", "amount"], ["2014 net revenue", "$ 5735"], ["2015 net revenue", "$ 5829"]],
    )
