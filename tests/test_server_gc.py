from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAKE_PROJECT_ROOT = Path(tempfile.gettempdir()) / "cosyvoice_win_jobs_test_root"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if "uvicorn" not in sys.modules:
    sys.modules["uvicorn"] = types.SimpleNamespace(run=lambda *args, **kwargs: None)

if "fastapi" not in sys.modules:
    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def get(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

    def _dummy_file(default=None, **kwargs):
        return default

    def _dummy_form(default=None, **kwargs):
        return default

    sys.modules["fastapi"] = types.SimpleNamespace(
        FastAPI=_DummyFastAPI,
        File=_dummy_file,
        Form=_dummy_form,
        HTTPException=Exception,
        UploadFile=object,
        status=types.SimpleNamespace(
            HTTP_202_ACCEPTED=202,
            HTTP_400_BAD_REQUEST=400,
            HTTP_404_NOT_FOUND=404,
            HTTP_409_CONFLICT=409,
            HTTP_500_INTERNAL_SERVER_ERROR=500,
            HTTP_503_SERVICE_UNAVAILABLE=503,
        ),
    )

if "fastapi.responses" not in sys.modules:
    sys.modules["fastapi.responses"] = types.SimpleNamespace(
        FileResponse=object,
        Response=object,
        StreamingResponse=object,
    )

if "pydantic" not in sys.modules:
    class _DummyBaseModel:
        pass

    def _dummy_field(default=None, **kwargs):
        return default

    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_DummyBaseModel, Field=_dummy_field)

if "cosyvoice_win.cli" not in sys.modules:
    fake_cli = types.SimpleNamespace(
        DEFAULT_FP16=True,
        DEFAULT_FIX_QUESTION_INTONATION=False,
        DEFAULT_MODEL_DIR=FAKE_PROJECT_ROOT / "pretrained_models" / "CosyVoice2-0.5B",
        DEFAULT_MODEL_ID="CosyVoice2-0.5B",
        DEFAULT_MODE="zero_shot",
        DEFAULT_SHARED_DIR=FAKE_PROJECT_ROOT / "shared",
        DEFAULT_SPEED=1.0,
        DEFAULT_TEXT_FRONTEND=False,
        PROJECT_ROOT=FAKE_PROJECT_ROOT,
        CosyVoiceModelOptions=lambda **kwargs: kwargs,
        CosyVoiceSynthesisOptions=lambda **kwargs: kwargs,
        ResolvedReference=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ensure_zero_shot_speaker=lambda *args, **kwargs: None,
        estimate_audio_duration_seconds=lambda text, speed=1.0: float(len(text.split())),
        find_reference_audio_in_shared=lambda *args, **kwargs: FAKE_PROJECT_ROOT / "shared" / "reference.wav",
        find_reference_text_for_audio=lambda *args, **kwargs: ("prompt", str(FAKE_PROJECT_ROOT / "shared" / "reference.txt")),
        format_duration=lambda seconds: f"{seconds:.1f}s",
        build_runtime_instruction_text=lambda **kwargs: None,
        load_model=lambda *args, **kwargs: object(),
        load_prompt_audio_16k=lambda *args, **kwargs: object(),
        parse_on_off=lambda value: bool(value) if isinstance(value, bool) else str(value).lower() == "on",
        resolve_dir=lambda value: Path(value),
        resolve_effective_mode=lambda mode, **kwargs: mode,
        resolve_model_dir=lambda value: Path(value),
        iter_synthesis=lambda *args, **kwargs: iter(()),
        synthesize_to_file=lambda *args, **kwargs: 1,
    )
    sys.modules["cosyvoice_win.cli"] = fake_cli

from cosyvoice_win.server import (
    SUPPORTED_DIRECT_RESPONSE_FORMATS,
    SUPPORTED_JOB_RESPONSE_FORMATS,
    JobStore,
    LoadedTranscriptionModel,
    ServerSettings,
    TranscriptionService,
    VoiceStore,
    audio_media_type,
    create_app,
    wav_header,
)


class DummyRequest:
    def __init__(
        self,
        *,
        input: str,
        model: str = "CosyVoice2-0.5B",
        voice: str = "reference",
        response_format: str = "wav",
        mode: str | None = None,
        text_frontend: bool | None = None,
        speed: float | None = None,
        stream: bool | None = None,
        instructions: str | None = None,
        instruct_text: str | None = None,
        reference_audio_base64: str | None = None,
        reference_audio_filename: str | None = None,
        reference_text: str | None = None,
        force_rebuild_voice: bool | None = None,
        metadata: dict | None = None,
    ):
        self.input = input
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.mode = mode
        self.text_frontend = text_frontend
        self.speed = speed
        self.stream = stream
        self.instructions = instructions
        self.instruct_text = instruct_text
        self.reference_audio_base64 = reference_audio_base64
        self.reference_audio_filename = reference_audio_filename
        self.reference_text = reference_text
        self.force_rebuild_voice = force_rebuild_voice
        self.metadata = metadata


def iso_utc(hours_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


class TestJobStoreCleanup(unittest.TestCase):
    def test_mark_downloaded_updates_job_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(DummyRequest(input="hello"))
            store.update_job(job["id"], status="completed", completed_at=iso_utc(1), audio_ready=True)
            store.audio_path(job["id"]).write_bytes(b"RIFF")

            updated = store.mark_downloaded(job["id"])

            self.assertEqual(updated["download_count"], 1)
            self.assertIsNotNone(updated["first_downloaded_at"])
            self.assertIsNotNone(updated["last_downloaded_at"])

    def test_cleanup_removes_old_completed_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(DummyRequest(input="hello"))
            store.update_job(job["id"], status="completed", completed_at=iso_utc(30), audio_ready=True)
            store.audio_path(job["id"]).write_bytes(b"RIFF")

            removed = store.cleanup_expired(
                job_retention=timedelta(hours=24),
                downloaded_job_retention=timedelta(hours=6),
            )

            self.assertEqual(removed, [job["id"]])
            self.assertFalse(store.job_dir(job["id"]).exists())
            self.assertIsNone(store.get_job(job["id"]))

    def test_request_payload_keeps_cosyvoice_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create_job(
                DummyRequest(
                    input="hello",
                    voice="reference_long",
                    mode="zero_shot",
                    text_frontend=False,
                    speed=1.0,
                    reference_text="exact transcript",
                )
            )

            payload_path = store.job_dir(job["id"]) / "request.json"
            payload = payload_path.read_text(encoding="utf-8")

            self.assertIn('"voice": "reference_long"', payload)
            self.assertIn('"mode": "zero_shot"', payload)
            self.assertIn('"reference_text": "exact transcript"', payload)


class TestVoiceStore(unittest.TestCase):
    def test_put_and_get_voice_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VoiceStore(Path(temp_dir))
            path = store.put(
                "reference_long",
                {
                    "voice": "reference_long",
                    "mode": "zero_shot",
                    "reference_text_present": True,
                },
            )

            self.assertTrue(path.exists())
            loaded = store.get("reference_long")
            self.assertEqual(loaded["voice"], "reference_long")
            self.assertTrue(loaded["reference_text_present"])


class TestDirectAudioHelpers(unittest.TestCase):
    def test_direct_endpoint_supports_pcm_without_changing_job_formats(self):
        self.assertEqual(SUPPORTED_JOB_RESPONSE_FORMATS, {"wav"})
        self.assertIn("wav", SUPPORTED_DIRECT_RESPONSE_FORMATS)
        self.assertIn("pcm", SUPPORTED_DIRECT_RESPONSE_FORMATS)
        self.assertEqual(audio_media_type("pcm"), "audio/pcm; codecs=pcm_s16le")
        self.assertEqual(audio_media_type("wav"), "audio/wav")

    def test_streaming_wav_header_uses_unknown_length_placeholder(self):
        header = wav_header(24000)

        self.assertEqual(header[:4], b"RIFF")
        self.assertEqual(header[8:12], b"WAVE")
        self.assertEqual(int.from_bytes(header[4:8], "little"), 0xFFFFFFFF)
        self.assertEqual(int.from_bytes(header[40:44], "little"), 0xFFFFFFFF)


class TestTranscriptionEndpoint(unittest.TestCase):
    def test_transcription_service_unloads_model_after_request_by_default(self):
        class Segment:
            id = 0
            seek = 0
            start = 0.0
            end = 1.0
            text = " hello"
            tokens = []
            temperature = 0.0
            avg_logprob = -0.1
            compression_ratio = 1.0
            no_speech_prob = 0.0

        class Info:
            language = "en"
            duration = 1.0

        class DummyWhisper:
            def transcribe(self, *args, **kwargs):
                return iter([Segment()]), Info()

        service = TranscriptionService(ServerSettings(), threading.Lock())
        service._loaded_model = LoadedTranscriptionModel(DummyWhisper(), 0.01)

        result = service.transcribe(Path("dummy.wav"))

        self.assertEqual(result["text"], "hello")
        self.assertTrue(result["unloaded_after_request"])
        self.assertIsNone(service._loaded_model)

    def test_transcription_service_can_keep_model_loaded(self):
        class Segment:
            id = 0
            seek = 0
            start = 0.0
            end = 1.0
            text = " hello"
            tokens = []
            temperature = 0.0
            avg_logprob = -0.1
            compression_ratio = 1.0
            no_speech_prob = 0.0

        class Info:
            language = "en"
            duration = 1.0

        class DummyWhisper:
            def transcribe(self, *args, **kwargs):
                return iter([Segment()]), Info()

        service = TranscriptionService(
            ServerSettings(unload_transcription_after_request=False),
            threading.Lock(),
        )
        loaded_model = LoadedTranscriptionModel(DummyWhisper(), 0.01)
        service._loaded_model = loaded_model

        result = service.transcribe(Path("dummy.wav"))

        self.assertEqual(result["text"], "hello")
        self.assertFalse(result["unloaded_after_request"])
        self.assertIs(service._loaded_model, loaded_model)

    def test_tts_and_transcription_share_gpu_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                ServerSettings(
                    shared_dir=root / "shared",
                    jobs_dir=root / "jobs",
                    voices_dir=root / "voices",
                    model_dir=root / "model",
                )
            )

            self.assertIs(app.state.worker._gpu_lock, app.state.gpu_lock)
            self.assertIs(app.state.transcriber._gpu_lock, app.state.gpu_lock)

    def test_transcription_endpoint_returns_openai_json_shape(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app(
                ServerSettings(
                    shared_dir=root / "shared",
                    jobs_dir=root / "jobs",
                    voices_dir=root / "voices",
                    model_dir=root / "model",
                )
            )

            def fake_transcribe(audio_path, **kwargs):
                self.assertTrue(Path(audio_path).is_file())
                self.assertEqual(kwargs["language"], "ru")
                return {"text": "hello from whisper"}

            app.state.transcriber.transcribe = fake_transcribe

            with TestClient(app) as client:
                response = client.post(
                    "/v1/audio/transcriptions",
                    data={
                        "model": "whisper-1",
                        "language": "ru",
                        "response_format": "json",
                    },
                    files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"text": "hello from whisper"})


if __name__ == "__main__":
    unittest.main()
