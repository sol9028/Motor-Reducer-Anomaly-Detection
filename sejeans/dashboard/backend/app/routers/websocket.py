"""WebSocket router.

WS /ws/stream  – pushes a new inference result every 2 seconds.

The WebSocket broadcasts to all connected clients, making it suitable for a
real-time dashboard. Each message is a JSON-serialised InferenceResponse.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services import alert_store
from app.services.inference_engine import build_inference_response

logger = logging.getLogger(__name__)

router = APIRouter()

# Active WebSocket connections
_connections: Set[WebSocket] = set()

_PUSH_INTERVAL_SECONDS = 2.0


async def _broadcast(payload: str) -> None:
    """Send a JSON string to all connected clients, removing stale ones."""
    dead: Set[WebSocket] = set()
    for ws in list(_connections):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


@router.websocket("/stream")
async def ws_stream(websocket: WebSocket):
    """Stream inference results to the connected client every 2 seconds."""
    await websocket.accept()
    _connections.add(websocket)
    logger.info("WebSocket client connected. Total: %d", len(_connections))

    try:
        while True:
            response = build_inference_response()
            if response is not None:
                alert_store.ingest_inference(response)
                payload = response.model_dump_json()
                try:
                    await websocket.send_text(payload)
                except Exception:
                    break
            await asyncio.sleep(_PUSH_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected cleanly")
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        _connections.discard(websocket)
        logger.info("WebSocket client removed. Remaining: %d", len(_connections))
