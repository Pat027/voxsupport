"""
Browser WebRTC transport.

The browser opens a WebSocket to /web/stream; audio flows in both directions
as raw PCM frames. The simplest production-grade setup on top of Pipecat.

For a true WebRTC SFU experience (better NAT traversal, adaptive bitrate),
swap this for DailyTransport (requires a Daily.co API key) or LiveKit.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
except ImportError:
    from pipecat.vad.silero import SileroVADAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web", tags=["web"])


@router.websocket("/stream")
async def web_stream(websocket: WebSocket) -> None:
    from src.voice.pipeline import run_conversation  # lazy import

    await websocket.accept()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    try:
        await run_conversation(transport)
    except Exception:  # noqa: BLE001
        logger.exception("Web call failed")
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
