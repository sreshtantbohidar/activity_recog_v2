"""Generate a sci-fi military terminal style PDF for detailed video analysis."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle)


def _text(value) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value)) or "None"
    return str(value)


def unique_frame_groups(frame_analysis: list[dict]) -> list[dict]:
    """Collapse consecutive frames with identical observations.

    Frames are compared on (activities, scene summary, key objects): a run of
    repeats becomes one entry whose timestamp window spans the whole run, so
    the PDF shows each distinct observation exactly once.
    """
    groups: list[dict] = []
    for frame in frame_analysis or []:
        signature = (tuple(sorted(frame.get("detected_activity", []) or [])),
            str(frame.get("scene_summary", "") or ""),
            tuple(sorted(str(item) for item in frame.get("key_objects", []) or [])))
        if groups and groups[-1]["signature"] == signature:
            groups[-1]["last_sec"] = float(frame.get("timestamp_sec", 0))
            groups[-1]["count"] += 1
        else:
            groups.append({"signature": signature,
                "first_sec": float(frame.get("timestamp_sec", 0)),
                "last_sec": float(frame.get("timestamp_sec", 0)),
                "count": 1})
    # Rebuild indices correctly (position within the original frame list).
    position = 0
    for group in groups:
        group["indices"] = list(range(position, position + group["count"]))
        position += group["count"]
        group.pop("next_index", None)
    return groups


def _overall_section_data(analysis: dict, overview: dict) -> dict:
    """Collect the headline findings for the report's opening section."""
    overall = analysis.get("overall_analysis") or {}
    executive = (analysis.get("detailed_report") or {}).get("executive_overview") or overview or {}
    audio = analysis.get("audio_analysis", {}) or {}
    cadence = audio.get("firing_cadence") or {}
    motion = analysis.get("motion_burstiness") or {}
    return {
        "dominant_activity": overall.get("dominant_activity") or executive.get("dominant_activity"),
        "basis": overall.get("dominant_activity_basis") or executive.get("dominant_activity_basis"),
        "activities": (overall.get("detected_activities") or executive.get("activities") or []),
        "evidence_biases": overall.get("evidence_biases") or [],
        "cadence_classification": cadence.get("classification"),
        "cadence_bias": cadence.get("suggested_activity_bias"),
        "cadence_median_gap": cadence.get("median_gap_sec"),
        "cadence_coverage": cadence.get("firing_coverage_pct"),
        "gunfire_events": (cadence.get("ast_gunfire_support") or {}).get("event_count"),
        "motion_classification": motion.get("classification"),
        "motion_bias": motion.get("suggested_activity_bias"),
        "audio_available": audio.get("has_audio", executive.get("audio_available", False)),
        "transcript_excerpt": " ".join(
            segment.get("text", "") for segment in (audio.get("transcript") or [])[:6]).strip(),
        "sound_cues": sorted({event.get("label") for event in (audio.get("sound_events") or [])
            if event.get("label")})[:12],
        "summary": overall.get("narrative") or executive.get("summary") or "",
    }


def create_video_analysis_pdf(analysis: dict, frame_paths: list[str], filename: str) -> dict:
    reports_dir = Path(os.getenv("REPORTS_DIR", "/app/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"video-analysis-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.pdf"
    output_path = reports_dir / report_name

    # ---- sci-fi terminal palette -------------------------------------------
    # Hex strings are kept alongside the Colour objects so Paragraph markup can
    # reuse the exact same values.
    hex_bg, hex_panel = "#070C11", "#0D151E"
    hex_panel_alt, hex_grid = "#0A1219", "#1E3444"
    hex_hairline, hex_phosphor = "#152734", "#00FF9C"
    hex_phosphor_dim, hex_cyan = "#0E6B4A", "#35E0FF"
    hex_amber, hex_danger = "#FFC857", "#FF4D5E"
    hex_dim, hex_light = "#8FA9B8", "#D7E8EF"

    bg = colors.HexColor(hex_bg)
    panel = colors.HexColor(hex_panel)
    panel_alt = colors.HexColor(hex_panel_alt)
    grid = colors.HexColor(hex_grid)
    hairline = colors.HexColor(hex_hairline)
    phosphor = colors.HexColor(hex_phosphor)
    phosphor_dim = colors.HexColor(hex_phosphor_dim)
    cyan = colors.HexColor(hex_cyan)
    amber = colors.HexColor(hex_amber)
    danger = colors.HexColor(hex_danger)
    dim_text = colors.HexColor(hex_dim)
    text_light = colors.HexColor(hex_light)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Courier-Bold",
        fontSize=22, leading=25, textColor=phosphor, alignment=TA_CENTER, spaceAfter=2))
    styles.add(ParagraphStyle(name="Banner", parent=styles["BodyText"], fontName="Courier-Bold",
        fontSize=7.5, leading=10, textColor=amber, alignment=TA_CENTER, spaceAfter=2))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["BodyText"], fontName="Courier",
        fontSize=7.5, leading=10, textColor=dim_text, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Courier-Bold",
        fontSize=11.5, leading=14, textColor=cyan, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="SectionMeta", parent=styles["BodyText"], fontName="Courier",
        fontSize=6.8, leading=9, textColor=dim_text, alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Courier",
        fontSize=7.8, leading=10.5, textColor=dim_text))
    styles.add(ParagraphStyle(name="BodySafe", parent=styles["BodyText"], fontName="Courier",
        fontSize=9, leading=12.5, spaceAfter=5, textColor=text_light))
    styles.add(ParagraphStyle(name="Mono", parent=styles["BodyText"], fontName="Courier-Bold",
        fontSize=10.5, leading=14, textColor=phosphor))
    styles.add(ParagraphStyle(name="Chip", parent=styles["BodyText"], fontName="Courier",
        fontSize=7.5, leading=10.5, textColor=text_light, alignment=TA_LEFT))

    def paragraph(value, style="BodySafe"):
        return Paragraph(escape(_text(value)).replace("\n", "<br/>"), styles[style])

    def status_chips(entries):
        """Reverse-video readout strip: label above, value beneath."""
        cells = []
        for label, value, accent in entries:
            cells.append(Paragraph(
                f"<font size=6.2 color=\"{hex_dim}\">{escape(label.upper())}</font><br/>"
                f"<font size=9.5 color=\"{accent}\"><b>{escape(_text(value)).upper()}</b></font>",
                styles["Chip"]))
        width = 174*mm / max(1, len(cells))
        return Table([cells], colWidths=[width] * len(cells), style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), panel),
            ("BOX", (0, 0), (-1, -1), 0.7, phosphor_dim),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, grid),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))

    def section(index, title):
        """Section header bar with a phosphor rule and a right-hand status tag."""
        label = Paragraph(f"<font color=\"{hex_phosphor}\">[ {index:02d} ]</font>  "
            f"{escape(_text(title)).upper()}", styles["Section"])
        meta = Paragraph(f"SECTION {index:02d} // VERIFIED", styles["SectionMeta"])
        return Table([[label, meta]], colWidths=[132*mm, 42*mm], style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ("LINEBELOW", (0, 0), (-1, -1), 1.2, phosphor),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 13), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))

    def table(rows, widths=None):
        value = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        value.setStyle(TableStyle([
            # Reverse-video header: phosphor bar with dark terminal text.
            ("BACKGROUND", (0, 0), (-1, 0), phosphor),
            ("TEXTCOLOR", (0, 0), (-1, 0), bg),
            ("FONTNAME", (0, 0), (-1, 0), "Courier-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("FONTSIZE", (0, 0), (-1, 0), 7.4),
            ("FONTSIZE", (0, 1), (-1, -1), 7.8),
            ("TEXTCOLOR", (0, 1), (-1, -1), text_light),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, grid),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [panel, panel_alt]),
            ("BOX", (0, 0), (-1, -1), 0.6, phosphor_dim),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return value

    def page_chrome(canvas, doc):
        """Dark terminal page: scanlines, HUD brackets, telemetry rail, stamps."""
        canvas.saveState()
        canvas.setFillColor(bg)
        canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)

        # Faint horizontal scanlines across the whole sheet.
        canvas.setStrokeColor(hairline)
        canvas.setLineWidth(0.3)
        for y in range(14, 292, 6):
            canvas.line(9*mm, y*mm, A4[0] - 9*mm, y*mm)

        # Outer and inner frames.
        canvas.setStrokeColor(grid)
        canvas.setLineWidth(1)
        canvas.rect(8*mm, 8*mm, A4[0] - 16*mm, A4[1] - 16*mm, stroke=1, fill=0)
        canvas.setStrokeColor(hairline)
        canvas.setLineWidth(0.4)
        canvas.rect(10.5*mm, 10.5*mm, A4[0] - 21*mm, A4[1] - 21*mm, stroke=1, fill=0)

        # HUD corner brackets.
        canvas.setStrokeColor(phosphor)
        canvas.setLineWidth(1.7)
        for x, y, dx, dy in ((12*mm, 12*mm, 1, 1), (A4[0] - 12*mm, 12*mm, -1, 1),
                             (12*mm, A4[1] - 12*mm, 1, -1), (A4[0] - 12*mm, A4[1] - 12*mm, -1, -1)):
            canvas.line(x, y, x + dx*7*mm, y)
            canvas.line(x, y, x, y + dy*7*mm)

        # Right-edge telemetry rail with tick marks.
        canvas.setStrokeColor(phosphor_dim)
        canvas.setLineWidth(0.5)
        canvas.line(A4[0] - 12.5*mm, 26*mm, A4[0] - 12.5*mm, A4[1] - 26*mm)
        for tick in range(46):
            y = (28 + tick * 5.1) * mm
            canvas.line(A4[0] - 14.3*mm, y, A4[0] - 10.7*mm, y)

        # Reverse-video classification bar.
        canvas.setFillColor(phosphor)
        canvas.rect(18*mm, A4[1] - 15.4*mm, 174*mm, 5.4*mm, stroke=0, fill=1)
        canvas.setFillColor(bg)
        canvas.setFont("Courier-Bold", 6.8)
        canvas.drawCentredString(A4[0]/2, A4[1] - 13.8*mm,
            "RESTRICTED // ACTIVITY RECOGNITION ANALYTICS // SYNTHESISED REPORT // HANDLE VIA SECURE CHANNEL")

        # Diagonal watermark, drawn beneath the story content.
        canvas.saveState()
        canvas.translate(A4[0]/2, A4[1]/2)
        canvas.rotate(32)
        canvas.setFillColor(colors.HexColor("#0E1E2A"))
        canvas.setFont("Courier-Bold", 54)
        canvas.drawCentredString(0, 0, "RESTRICTED")
        canvas.restoreState()

        # Footer channel line and pagination.
        canvas.setStrokeColor(phosphor)
        canvas.setLineWidth(1.1)
        canvas.line(18*mm, 14*mm, 192*mm, 14*mm)
        canvas.setFont("Courier", 6.8)
        canvas.setFillColor(dim_text)
        canvas.drawString(18*mm, 9.6*mm, "NODE // ACTIVITY-RECOGNITION // TACTICAL ANALYTICS ENGINE")
        canvas.drawCentredString(A4[0]/2, 9.6*mm,
            f"BUILD {datetime.now(timezone.utc):%Y%m%d.%H%M}Z")
        canvas.setFillColor(phosphor)
        canvas.drawRightString(192*mm, 9.6*mm, f"[ PAGE {doc.page:03d} ]")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
        topMargin=18*mm, bottomMargin=18*mm, title="Detailed Video Analysis Report")

    metadata = analysis.get("metadata", {})
    report = analysis.get("detailed_report", {})
    overview = report.get("executive_overview", {})
    overall = _overall_section_data(analysis, overview)

    story = [
        Paragraph("// RESTRICTED // ACTIVITY RECOGNITION ANALYTICS // AUTONOMOUS SYNTHESIS //", styles["Banner"]),
        Paragraph("TACTICAL VIDEO ANALYSIS", styles["ReportTitle"]),
        Paragraph("MACHINE-ASSISTED MULTIMODAL INTERPRETATION // EYES ONLY // NOT FOR DISTRIBUTION",
            styles["SubTitle"]),
        paragraph(f"SOURCE FILE: {filename}   //   REPORT ID: {report_name}", "Small"),
        paragraph(f"GENERATED (UTC): {datetime.now(timezone.utc).isoformat()}", "Small"),
        Spacer(1, 10),
    ]

    # ---- [01] overall analysis ---------------------------------------------
    basis_label = {
        "multimodal_evidence_override": "MULTIMODAL OVERRIDE",
        "multimodal_evidence_consensus": "MULTIMODAL CONSENSUS",
        "multimodal_evidence_agreement": "MULTIMODAL AGREEMENT",
        "audio_firing_cadence_override": "AUDIO CADENCE OVERRIDE",
        "visual_frame_votes": "VISUAL FRAME VOTES",
    }.get(overall["basis"], _text(overall["basis"]).upper())
    story += [status_chips([
            ("Dominant activity", overall["dominant_activity"] or "undetermined", hex_phosphor),
            ("Decision basis", basis_label, hex_cyan),
            ("Audio stream", "available" if overall["audio_available"] else "absent",
                hex_phosphor if overall["audio_available"] else hex_amber),
            ("Firing cadence", overall["cadence_classification"] or "unknown", hex_danger),
        ]),
        section(1, "OVERALL ANALYSIS"),
        paragraph(f"DOMINANT ACTIVITY: {overall['dominant_activity'] or 'Not available'}", "Mono"),
        paragraph(f"DECISION BASIS: {basis_label}   //   ACTIVITIES: {_text(overall['activities'])}", "Small"),
        paragraph(f"AUDIO: {'AVAILABLE' if overall['audio_available'] else 'NOT AVAILABLE'}"
            f"   //   FUSION VOTES: {_text(overall['evidence_biases']) or 'none beyond visual'}", "Small")]
    cadence_line = "Not available"
    if overall["cadence_classification"]:
        cadence_line = (f"{overall['cadence_classification'].upper()} (bias: {overall['cadence_bias'] or 'none'};"
            f" median gap: {overall['cadence_median_gap']}s; coverage: {overall['cadence_coverage']}%)")
    motion_line = "Not available"
    if overall["motion_classification"]:
        motion_line = f"{overall['motion_classification'].upper()} (bias: {overall['motion_bias'] or 'none'})"
    story += [paragraph(f"FIRING CADENCE: {cadence_line}", "Small"),
        paragraph(f"MOTION RHYTHM: {motion_line}", "Small"),
        paragraph(f"DETECTED SOUND CUES: {_text(overall['sound_cues']) or 'none'}", "Small")]
    if overall["transcript_excerpt"]:
        story.append(paragraph(f"SPEECH EXCERPT: {overall['transcript_excerpt']}", "Small"))
    if overall["summary"]:
        story.append(paragraph(f"SYNTHESIS: {overall['summary']}", "Small"))

    # ---- [02] executive overview -------------------------------------------
    story += [section(2, "EXECUTIVE OVERVIEW"), table([
        ["Duration", "Frames", "Tracks", "Audio", "Activities"],
        [f"{metadata.get('duration_sec', 0):.2f} sec", overview.get("frames_analyzed", 0),
         overview.get("persistent_tracks", 0),
         "YES" if overview.get("audio_available") else "NO", paragraph(overview.get("activities", []), "Small")]
    ], [25*mm, 20*mm, 20*mm, 17*mm, 64*mm])]

    story += [section(3, "ACTIVITY EVIDENCE")]
    evidence = report.get("activity_evidence", [])
    story.append(table([["Activity", "First seen", "Last seen", "Frames"]] +
        [[paragraph(x.get("activity"), "Small"), x.get("first_seen_sec"), x.get("last_seen_sec"),
          x.get("supporting_frames")] for x in evidence], [55*mm, 34*mm, 34*mm, 43*mm]) if evidence
        else paragraph("No classified activity evidence was produced."))

    # ---- [04] frame evidence, consecutive duplicates removed ----------------
    groups = unique_frame_groups(analysis.get("frame_analysis", []))
    story += [section(4, "FRAME-BY-FRAME VISUAL EVIDENCE"),
        paragraph(f"{len(analysis.get('frame_analysis', []))} frames analyzed -> "
            f"{len(groups)} unique observation(s) shown; repeated consecutive observations are collapsed.", "Small")]
    for group in groups:
        frame = analysis["frame_analysis"][group["indices"][0]]
        image = Image(frame_paths[group["indices"][0]], width=62*mm, height=42*mm, kind="proportional")
        stamp = (f"{group['first_sec']:.2f} sec" if group["count"] == 1
            else f"{group['first_sec']:.2f} - {group['last_sec']:.2f} sec ({group['count']} frames)")
        confidence = frame.get("confidence", 0.0)
        confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        details = [paragraph(f"TIMESTAMP: {stamp}", "Small"),
            paragraph(f"SELECTED FOR: {_text(frame.get('frame_selection',{}).get('reasons',[]))};"
                f" change score: {frame.get('frame_selection',{}).get('change_score',0)}", "Small"),
            paragraph(f"SCENE: {frame.get('scene_summary', '')}", "Small"),
            paragraph(f"ACTIVITIES: {_text(frame.get('detected_activity', []))}", "Small"),
            paragraph(f"OBJECTS: {_text(frame.get('key_objects', []))}", "Small"),
            paragraph(f"CONFIDENCE: {confidence * 100:.0f}%", "Small"),
            paragraph(f"REASONING: {frame.get('reasoning', '')}", "Small")]
        story.append(KeepTogether([Table([[image, details]], colWidths=[67*mm, 99*mm],
            style=[("VALIGN",(0,0),(-1,-1),"TOP"),
                   ("BOX",(0,0),(-1,-1),0.5,grid),
                   ("BACKGROUND",(0,0),(-1,-1),panel),
                   ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
                   ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]), Spacer(1, 6)]))

    story += [section(5, "TRACKED PERSONNEL AND OBJECTS")]
    tracks = analysis.get("object_tracking", {}).get("tracks", [])
    story.append(table([["Track ID", "Class", "First seen", "Last seen", "Obs", "Max conf"]] +
        [[x.get("track_id"), paragraph(x.get("class"), "Small"), x.get("first_seen_sec"), x.get("last_seen_sec"),
          x.get("observations"), x.get("max_confidence")] for x in tracks],
        [17*mm, 44*mm, 27*mm, 27*mm, 15*mm, 36*mm]) if tracks else paragraph("No persistent tracks were detected."))

    audio = analysis.get("audio_analysis", {})
    story += [section(6, "AUDIO ANALYSIS"), paragraph(
        ">> [ OK ]  AUDIO STREAM DETECTED // UNIFIED SPEECH + ENVIRONMENTAL SOUND PASS"
        if audio.get("has_audio") else ">> [ -- ]  NO AUDIO STREAM DETECTED")]
    transcripts = audio.get("transcript", [])
    if transcripts:
        story.append(table([["Time", "Transcript"]] + [[f"{x.get('start_sec')} - {x.get('end_sec')} sec",
            paragraph(x.get("text"), "Small")] for x in transcripts], [38*mm, 128*mm]))
    events = audio.get("sound_events", [])
    if events:
        story += [Spacer(1, 5), table([["Time", "Sound event", "Confidence"]] + [[
            f"{x.get('start_sec')} - {x.get('end_sec')} sec", paragraph(x.get("label"), "Small"),
            x.get("confidence")] for x in events], [42*mm, 92*mm, 32*mm])]

    story += [section(7, "INTERPRETATION AND LIMITATIONS")]
    for note in report.get("cross_modal_notes", []) + report.get("limitations", []):
        story.append(paragraph(f">> {note}"))
    story += [section(8, "INFERENCE PROVENANCE"),
        paragraph(f"VISION BACKENDS: {_text(analysis.get('backends_used', []))}", "Small"),
        paragraph("VISION ENGINE: in-container vLLM // Qwen3-VL-8B-Instruct int4 (AWQ) // enforce_eager // sleeps for audio", "Small"),
        paragraph("AUDIO ENGINE: Qwen2-Audio-7B-Instruct (bfloat16) // runs STRICTLY AFTER vision, while vLLM sleeps", "Small")]
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    host_dir = Path(os.getenv("REPORTS_HOST_PATH", str(reports_dir)))
    # `path` is the host-visible location (handy for operators), `local_path` is
    # where the file actually lives inside the container that serves the API.
    return {"filename": report_name, "path": str(host_dir / report_name),
            "local_path": str(output_path), "download_url": f"/reports/{report_name}"}
