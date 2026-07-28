import json

import pytest

from robonix_openvla_skill.vla_client import VLAClient


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, body):
        self.body = body
        self.last_post = None

    def post(self, url, **kwargs):
        self.last_post = (url, kwargs)
        return FakeResponse(self.body)

    def get(self, url, **kwargs):
        return FakeResponse({"ok": True})

    def close(self):
        pass


def test_predict_parses_action_chunk() -> None:
    session = FakeSession(
        {
            "action_chunk": [[0, 0, 0, 0, 0, 0, 0.5]],
            "done": True,
            "success": True,
        }
    )
    client = VLAClient(
        server_url="http://localhost:8001/act",
        timeout_s=2,
        session=session,
    )
    result = client.predict(
        task_id="pick",
        instruction="pick",
        image_jpeg=b"jpeg",
        state=[0] * 7,
    )
    assert result.done is True
    assert result.actions[0][6] == 0.5
    payload = json.loads(session.last_post[1]["data"]["payload"])
    assert payload["task_id"] == "pick"


def test_predict_rejects_wrong_action_shape() -> None:
    client = VLAClient(
        server_url="http://localhost:8001/act",
        timeout_s=2,
        session=FakeSession({"action": [0, 1]}),
    )
    with pytest.raises(RuntimeError, match="seven"):
        client.predict(
            task_id="pick",
            instruction="pick",
            image_jpeg=b"jpeg",
            state=[0] * 7,
        )
