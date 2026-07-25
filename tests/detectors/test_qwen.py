import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ai_guardrail.detectors.qwen import QwenDetector, resolve_occurrence
from ai_guardrail.domain import EntityType

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "qwen-entity-schema.json"
)


def make_detector(
    transport: httpx.MockTransport,
    *,
    timeout_seconds: float = 2,
) -> QwenDetector:
    return QwenDetector(
        base_url="http://qwen.test/",
        model_version="qwen3-0.6b-q4_k_m",
        timeout_seconds=timeout_seconds,
        schema_path=SCHEMA_PATH,
        transport=transport,
    )


def test_repeated_text_requires_explicit_occurrence() -> None:
    text = "Project Falcon replaced Project Falcon."
    assert resolve_occurrence(text, "Project Falcon", None) is None
    assert resolve_occurrence(text, "Project Falcon", 2) == (24, 38)


def test_occurrence_requires_a_real_integer_and_non_empty_value() -> None:
    assert resolve_occurrence("Jane", "", 1) is None
    assert resolve_occurrence("Jane", "Jane", True) is None
    assert resolve_occurrence("Jane", "Jane", 1.0) is None  # type: ignore[arg-type]


def test_occurrence_uses_unicode_code_point_offsets() -> None:
    assert resolve_occurrence("😀 Jane Cooper", "Jane Cooper", None) == (2, 13)


@pytest.mark.asyncio
async def test_qwen_uses_explicit_schema_path_outside_repository_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entities":[]}'}}]},
        )

    monkeypatch.chdir(tmp_path)
    detector = make_detector(httpx.MockTransport(handler))

    output = await detector.detect("Jane")

    assert output.status == "success"


@pytest.mark.parametrize("schema_contents", ["not-json secret", '{"type": 123}'])
def test_qwen_rejects_invalid_schema_without_leaking_details(
    tmp_path: Path,
    schema_contents: str,
) -> None:
    schema_path = tmp_path / "secret-schema.json"
    schema_path.write_text(schema_contents, encoding="utf-8")

    with pytest.raises(ValueError) as error:
        QwenDetector(
            base_url="http://qwen.test/",
            model_version="qwen3-0.6b-q4_k_m",
            timeout_seconds=2,
            schema_path=schema_path,
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )

    assert str(error.value) == "invalid Qwen detector schema"
    assert error.value.__cause__ is None


def test_qwen_rejects_missing_schema_without_leaking_path(tmp_path: Path) -> None:
    schema_path = tmp_path / "secret-missing-schema.json"

    with pytest.raises(ValueError) as error:
        QwenDetector(
            base_url="http://qwen.test/",
            model_version="qwen3-0.6b-q4_k_m",
            timeout_seconds=2,
            schema_path=schema_path,
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )

    assert str(error.value) == "invalid Qwen detector schema"
    assert error.value.__cause__ is None


@pytest.mark.asyncio
async def test_qwen_calls_exact_llama_cpp_paths_with_trailing_base_slash() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"entities":[]}'}}]},
            )
        return httpx.Response(404)

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "success"
    assert requested_paths == ["/tokenize", "/v1/chat/completions"]


@pytest.mark.asyncio
async def test_qwen_enforces_one_deadline_across_both_http_calls() -> None:
    requested_paths: list[str] = []
    completion_finished = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal completion_finished
        requested_paths.append(request.url.path)
        await asyncio.sleep(0.25)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        completion_finished = True
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entities":[]}'}}]},
        )

    output = await make_detector(
        httpx.MockTransport(handler),
        timeout_seconds=0.4,
    ).detect("Jane")

    assert output.status == "timeout"
    assert output.error_code == "detector_timeout"
    assert requested_paths == ["/tokenize", "/v1/chat/completions"]
    assert completion_finished is False


@pytest.mark.asyncio
async def test_qwen_accepts_only_validated_source_substrings() -> None:
    response_content = {
        "entities": [
            {"type": "PERSON", "text": "Jane Cooper"},
            {"type": "INTERNAL_PROJECT", "text": "Project Mirage"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(12))})
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["max_tokens"] == 96
        assert body["messages"][-1]["content"].endswith("\n/no_think")
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["schema"]["additionalProperties"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(response_content)}}
                ]
            },
        )

    detector = make_detector(httpx.MockTransport(handler))

    output = await detector.detect("Send Jane Cooper's record.")

    assert len(output.candidates) == 1
    assert output.candidates[0].type == EntityType.PERSON
    assert output.invalid_candidate_count == 1


@pytest.mark.asyncio
async def test_qwen_returns_safe_error_for_malformed_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-json"}}]},
        )

    detector = make_detector(httpx.MockTransport(handler))

    output = await detector.detect("Contact Jane Cooper.")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"
    assert output.candidates == []


@pytest.mark.asyncio
async def test_qwen_rejects_non_list_token_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": "123"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entities":[]}'}}]},
        )

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"


@pytest.mark.asyncio
async def test_qwen_returns_safe_error_for_empty_choices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(200, json={"choices": []})

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"


@pytest.mark.asyncio
async def test_qwen_rejects_non_integer_occurrence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"entities":[{"type":"PERSON","text":"Jane",'
                                '"occurrence":1.0}]}'
                            )
                        }
                    }
                ]
            },
        )

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"


@pytest.mark.asyncio
async def test_qwen_rejects_oversized_model_content() -> None:
    oversized_content = '{"entities":[]}' + (" " * 20_000)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": oversized_content}}]},
        )

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"


@pytest.mark.asyncio
async def test_qwen_enforces_input_token_cap_without_calling_completion() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, json={"tokens": list(range(513))})

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "input_too_long"
    assert requested_paths == ["/tokenize"]


@pytest.mark.asyncio
async def test_qwen_returns_timeout_without_exposing_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret prompt details", request=request)

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "timeout"
    assert output.error_code == "detector_timeout"
    assert output.candidates == []
    assert "secret" not in output.model_dump_json()
    assert "Jane" not in output.model_dump_json()


@pytest.mark.asyncio
async def test_qwen_returns_safe_error_for_http_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="secret raw model response")

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"
    assert "secret" not in output.model_dump_json()
    assert "Jane" not in output.model_dump_json()


@pytest.mark.asyncio
async def test_qwen_rejects_unknown_entity_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"entities":[{"type":"UNKNOWN","text":"Jane"}]}'
                            )
                        }
                    }
                ]
            },
        )

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"


@pytest.mark.asyncio
async def test_qwen_marks_ambiguous_repeat_invalid_and_accepts_unicode_span() -> None:
    content = json.dumps(
        {
            "entities": [
                {"type": "PERSON", "text": "Jane"},
                {"type": "PERSON", "text": "Zoë 😀"},
            ]
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    output = await make_detector(httpx.MockTransport(handler)).detect(
        "Jane met Jane and Zoë 😀."
    )

    assert output.status == "success"
    assert output.invalid_candidate_count == 1
    assert len(output.candidates) == 1
    assert (output.candidates[0].start, output.candidates[0].end) == (18, 23)


@pytest.mark.asyncio
async def test_qwen_rejects_more_than_schema_maximum_entities() -> None:
    content = json.dumps(
        {"entities": [{"type": "PERSON", "text": "Jane"}] * 33}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    output = await make_detector(httpx.MockTransport(handler)).detect("Jane")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"
