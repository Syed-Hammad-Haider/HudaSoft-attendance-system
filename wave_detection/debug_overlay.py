"""
DEBUG/ANALYSIS overlay for wave detection -- toggle with 'D' while
main.py is running.

WHY A SEPARATE MODULE, NOT PART OF display/renderer.py: this overlay
is a diagnostic tool, not part of the normal product UI (the renderer
covers everything a real user of the attendance system should see).
Keeping it separate means it can show raw internal numbers (hand
scale, swing magnitude, regularity CV, sample counts) that would be
noise in the normal display, without touching or risking the
already-tested renderer code at all.

Draws, per tracked person with an active hand: a compact panel showing
- hand tracking status (tracked / lost)
- samples analyzed this streak
- current swing magnitude vs. required threshold (in hand-widths)
- reversal count vs. required count
- regularity (coefficient of variation) and whether it currently passes
- body-stillness displacement vs. the allowed cap
- overall detector state (idle / tracking / ready_pending_regularity / confirmed)

This is intentionally verbose/raw -- it's meant to answer "why is (or
isn't) a wave being detected right now," not to look polished.
"""

from typing import Optional

import cv2

from wave_detection.wave_result import WaveResult


def draw_wave_debug_panel(frame, person_bbox, wave_result: WaveResult, panel_index: int = 0) -> None:
    """
    Draws a debug panel near the given person's bbox. `panel_index`
    offsets multiple people's panels vertically so they don't overlap
    when several people are tracked at once.
    """
    if not wave_result.hands:
        return

    x1, y1, x2, y2 = person_bbox
    frame_h, frame_w = frame.shape[:2]

    panel_w = 300
    line_h = 16
    lines = []

    for i, hand in enumerate(wave_result.hands):
        dbg = hand.debug
        if dbg is None:
            continue
        lines.append(f"--- Hand {i} ---")
        lines.append(f"state: {dbg.state}")
        lines.append(f"samples analyzed: {dbg.samples_analyzed}")
        if dbg.hand_scale_px is not None:
            lines.append(f"hand scale: {dbg.hand_scale_px:.1f}px")
        lines.append(
            f"swing: {dbg.last_swing_magnitude_hand_widths:.2f} hw "
            f"(need >= threshold)"
        )
        lines.append(f"reversals: {dbg.reversal_count} / {dbg.reversals_required}")
        if dbg.regularity_cv is not None:
            ok = "OK" if dbg.regularity_ok else "TOO IRREGULAR"
            lines.append(f"regularity CV: {dbg.regularity_cv:.2f} ({ok})")
        else:
            lines.append("regularity CV: n/a (need 3+ reversals)")
        if dbg.seconds_since_last_reversal is not None:
            lines.append(f"since last reversal: {dbg.seconds_since_last_reversal:.2f}s")

    if wave_result.body_debug is not None:
        bd = wave_result.body_debug
        lines.append("--- Body ---")
        if bd.speed_fraction_of_height_per_sec is not None:
            status = "STILL" if bd.is_stationary else "MOVING (blocks wave)"
            lines.append(f"speed: {bd.speed_fraction_of_height_per_sec:.2f} height/s ({status})")
        else:
            lines.append("displacement: n/a (gathering history)")
        lines.append(f"body samples: {bd.samples}")

    if not lines:
        return

    panel_h = line_h * len(lines) + 10
    panel_x = min(max(0, x2 + 8), frame_w - panel_w - 1)
    panel_y = min(max(0, y1 + panel_index * (panel_h + 10)), frame_h - panel_h - 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 230), 1)

    for i, text in enumerate(lines):
        color = (255, 255, 255)
        if text.startswith("---"):
            color = (0, 200, 230)
        elif "TOO IRREGULAR" in text or "MOVING (blocks wave)" in text:
            color = (60, 60, 240)
        elif "OK" in text or "STILL" in text:
            color = (80, 200, 120)
        cv2.putText(
            frame, text, (panel_x + 8, panel_y + 16 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )
