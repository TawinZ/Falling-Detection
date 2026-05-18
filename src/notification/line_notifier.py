"""LINE Bot notification for fall detection alerts"""

import requests
import base64
import threading
from datetime import datetime
from pathlib import Path

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"

# Cooldown: only one notification per this many seconds
ALERT_COOLDOWN_SEC = 30

_last_alert_time = 0.0
_lock = threading.Lock()


def _upload_to_imgbb(image_path: str, api_key: str) -> str | None:
    """Upload image to imgbb, return public URL or None on failure."""
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        resp = requests.post(
            IMGBB_UPLOAD_URL,
            data={"key": api_key, "image": encoded},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()["data"]["url"]
        print(f"[LINE] imgbb upload failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"[LINE] imgbb upload error: {e}")
    return None


def _push_messages(messages: list, token: str, user_id: str) -> bool:
    """Send messages via LINE Messaging API."""
    try:
        resp = requests.post(
            LINE_PUSH_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"to": user_id, "messages": messages},
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        print(f"[LINE] Push failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print(f"[LINE] Push error: {e}")
    return False


def _do_send(screenshot_path, confidence: float, timestamp: datetime,
             token: str, user_id: str, imgbb_key: str | None):
    messages = []

    text = (
        f"FALL DETECTED!\n"
        f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Confidence: {confidence * 100:.1f}%\n"
        f"Please check on the person immediately."
    )
    messages.append({"type": "text", "text": text})

    if imgbb_key and screenshot_path and Path(screenshot_path).exists():
        image_url = _upload_to_imgbb(str(screenshot_path), imgbb_key)
        if image_url:
            messages.append({
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url,
            })

    success = _push_messages(messages, token, user_id)
    if success:
        print("[LINE] Alert sent successfully.")
    else:
        print("[LINE] Alert failed.")


def send_fall_alert(
    screenshot_path,
    confidence: float,
    timestamp: datetime,
    token: str,
    user_id: str,
    imgbb_key: str | None = None,
) -> None:
    """Send fall alert non-blocking. Respects ALERT_COOLDOWN_SEC between alerts."""
    import time

    global _last_alert_time

    with _lock:
        now = time.time()
        if now - _last_alert_time < ALERT_COOLDOWN_SEC:
            return
        _last_alert_time = now

    t = threading.Thread(
        target=_do_send,
        args=(screenshot_path, confidence, timestamp, token, user_id, imgbb_key),
        daemon=True,
    )
    t.start()
