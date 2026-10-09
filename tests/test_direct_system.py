from __future__ import annotations

import json

from finagent.llm.fake import FakeClient
from finagent.schemas import FinQAExample
from finagent.systems.direct import DirectFinQASystem


def test_direct_system_parses_reply_and_sees_context(finqa_example: FinQAExample) -> None:
    client = FakeClient(json.dumps({"reasoning": "5829 - 5735 = 94", "answer": " 94 "}))
    answer = DirectFinQASystem(client).answer(finqa_example)
    assert answer.answer == "94"
    assert answer.evidence[0].doc_id == finqa_example.id
    assert "2015 net revenue" in client.calls[0]
    assert finqa_example.question in client.calls[0]


def test_fake_embeddings_are_deterministic_unit_vectors() -> None:
    client = FakeClient()
    a, b = client.embed(["x", "x"])
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9
