"""
Twilio phone transport.

Twilio Media Streams pushes 8 kHz mulaw audio over a WebSocket to our server.
Pipecat's TwilioFrameSerializer handles the framing, so we hand the WebSocket
to the pipeline and it speaks back through the same socket.

Flow:
1. Caller dials the Twilio number.
2. Twilio hits our /twilio/voice endpoint -> we return TwiML with <Stream>.
3. Twilio opens a WebSocket to /twilio/stream -> we attach the Pipecat pipeline.
4. Conversation happens over the socket until hangup.

Setup checklist (one-time):
- Buy a Twilio number (~$1/month in most countries).
- In the Twilio Console, set the number's Voice webhook to:
    https://{your-domain}/twilio/voice
- Deploy voxsupport to a public URL (Modal, Fly, Render, ...).
- Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER env vars.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response

from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.vad.silero import SileroVADAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])


@router.post("/voice")
async def twilio_voice_webhook(request: Request) -> Response:
    """Handle the inbound call webhook from Twilio by returning TwiML.

    The TwiML tells Twilio to open a bi-directional media stream to our
    WebSocket endpoint. The voxsupport pipeline lives on the other end.
    """
    host = request.headers.get("host") or os.environ.get("PUBLIC_HOST", "")
    stream_url = f"wss://{host}/twilio/stream"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/stream")
async def twilio_stream(websocket: WebSocket) -> None:
    """Per-call WebSocket. Attaches the voice pipeline and runs it until hangup."""
    from src.voice.pipeline import run_conversation  # lazy to avoid import cycles

    await websocket.accept()

    # Twilio's first WS message is a "start" event with stream metadata.
    # Pipecat's TwilioFrameSerializer reads this automatically.
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=TwilioFrameSerializer(),
        ),
    )

    try:
        await run_conversation(transport)
    except Exception:  # noqa: BLE001
        logger.exception("Twilio call failed")
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
