import json
import os
import tempfile
import shutil
from io import BytesIO
from datetime import datetime
from typing import Optional

from pypdf import PdfReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
)
from reportlab.platypus.flowables import KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

DB_PATH = "data/applications.json"
RESUME_PATH = "data/default_resume.txt"
MAX_URL_CONTENT_TOKENS = 25_000  # ~100k chars; truncate before passing to agent


# =========================
# DATABASE (applications)
# =========================

def load_db() -> list[dict]:
    """Load all saved applications from disk.

    Returns an empty list if the file does not exist or is corrupt,
    rather than raising, so callers never need to handle file-not-found.
    """
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("DB root must be a JSON array")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            f"applications.json is corrupt and could not be loaded ({e}). "
            "Fix or delete the file at: " + DB_PATH
        ) from e


def save_db(db: list[dict]) -> None:
    """Persist the full applications list to disk atomically.

    Writes to a temp file first, then renames, so a crash mid-write
    never corrupts the existing database.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    dir_name = os.path.dirname(os.path.abspath(DB_PATH))
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=dir_name, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(db, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, DB_PATH)


def save_application(job_description: str, resume: str, result: dict) -> dict:
    """Append one application run to the persistent database.

    Args:
        job_description: Raw job posting text.
        resume: Resume text used for this run.
        result: Structured output dict returned by the agent.

    Returns:
        The saved entry, including its generated timestamp.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "job_description": job_description,
        "resume_snapshot": resume,
        "result": result,
    }
    db = load_db()
    db.append(entry)
    save_db(db)
    return entry


# =========================
# RESUME MEMORY
# =========================

def save_default_resume(text: str, name: str = "Saved Resume") -> None:
    """Save a resume as the default for future sessions.

    Args:
        text: Plain-text resume content.
        name: Human-readable label shown in the UI.
    """
    if not text or not text.strip():
        raise ValueError("Resume text must not be empty.")
    os.makedirs(os.path.dirname(RESUME_PATH), exist_ok=True)
    with open(RESUME_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": name, "text": text}, f, ensure_ascii=False)


def load_default_resume() -> Optional[dict]:
    """Load the saved default resume from disk.

    Returns:
        Dict with keys ``name`` and ``text``, or ``None`` if no resume is saved.
    """
    if not os.path.exists(RESUME_PATH):
        return None
    with open(RESUME_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "text" in data:
            return data
        raise ValueError("Unexpected format")
    except (json.JSONDecodeError, ValueError):
        # Backwards-compat: file was plain text before JSON wrapper was added
        return {"name": "Imported Resume", "text": raw}


# =========================
# PDF PARSING & STYLE EXTRACTION
# =========================

def extract_pdf_style(file_bytes: bytes) -> dict:
    """
    Extract the accent color and dominant font sizing from a PDF's first page.

    Uses PyMuPDF (fitz) when available. Falls back to empty dict so callers
    always get a valid (possibly empty) style dict without needing to handle errors.

    Returns dict with optional keys:
        accent_color: hex string e.g. "#4f46e5"
        name_font_size: float, largest font size seen (capped at 24)
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]

        color_weight: dict[int, float] = {}
        max_font_size = 0.0

        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if size > max_font_size:
                        max_font_size = size

                    color_int = span.get("color", 0)
                    # Skip pure black (0x000000) and pure white (0xFFFFFF)
                    if color_int in (0, 0xFFFFFF):
                        continue
                    # Weight by font size so larger/heading text dominates
                    color_weight[color_int] = color_weight.get(color_int, 0.0) + size

        doc.close()

        style: dict = {}
        if color_weight:
            dominant = max(color_weight, key=color_weight.get)  # type: ignore[arg-type]
            r = (dominant >> 16) & 0xFF
            g = (dominant >> 8) & 0xFF
            b = dominant & 0xFF
            style["accent_color"] = f"#{r:02x}{g:02x}{b:02x}"

        if max_font_size:
            style["name_font_size"] = min(float(max_font_size), 24.0)

        return style
    except Exception:
        return {}


def extract_text_from_pdf(file) -> str:
    """Extract all text from a PDF file object.

    Args:
        file: File-like object (e.g. from st.file_uploader or open()).

    Returns:
        Concatenated plain text from all pages.

    Raises:
        ValueError: If the PDF contains no extractable text (e.g. scanned image).
    """
    reader = PdfReader(file)
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            pages_text.append(text)
        except Exception as e:
            # Skip unreadable pages rather than aborting the whole document
            pages_text.append(f"[Page {i + 1} could not be read: {e}]")

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image — try copying and pasting the text manually."
        )
    return full_text


# =========================
# URL PARSING
# =========================

def extract_job_from_url(url: str, timeout_ms: int = 60_000) -> str:
    """Scrape the visible text of a job posting page using a headless browser.

    Uses Playwright so JavaScript-rendered pages (Greenhouse, Lever, Workday)
    load fully before extraction.

    Args:
        url: Public URL of the job posting.
        timeout_ms: Page load timeout in milliseconds (default 60s).

    Returns:
        Visible body text of the page, truncated to ``MAX_URL_CONTENT_TOKENS`` chars.

    Raises:
        ValueError: If the URL is empty or the page fails to load.
        RuntimeError: If Playwright is not installed.
    """
    if not url or not url.strip():
        raise ValueError("A non-empty URL is required.")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, timeout=timeout_ms)
                page.wait_for_timeout(3_000)  # allow JS to finish rendering
                content = page.inner_text("body")
            except PlaywrightTimeoutError:
                raise ValueError(
                    f"The page at {url!r} took too long to load. "
                    "Check the URL or try again."
                )
            finally:
                browser.close()
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(
            f"Playwright failed to launch. Make sure it is installed: "
            "`playwright install chromium`"
        ) from e

    return content[:MAX_URL_CONTENT_TOKENS]


def search_linkedin_jobs(search_queries: list[str], location: str = "") -> list[dict]:
    """Search LinkedIn jobs using a headless browser.

    Returns one result dict per query with keys: query, text (page body text,
    truncated to 12000 chars), blocked (bool), url (str).

    Args:
        search_queries: List of job search queries (only the first 3 are used).
        location: Optional location string to filter results.

    Returns:
        List of dicts, one per query, each with keys:
        ``query``, ``text``, ``blocked``, ``url``.
    """
    from urllib.parse import quote_plus

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )

        for query in search_queries[:3]:
            encoded_q = quote_plus(query)
            encoded_loc = quote_plus(location)
            url = (
                f"https://www.linkedin.com/jobs/search/"
                f"?keywords={encoded_q}&location={encoded_loc}&f_TP=1%2C2%2C3%2C4"
            )
            content = ""
            blocked = False
            page = context.new_page()
            try:
                page.goto(url, timeout=40000, wait_until="domcontentloaded")
                page.wait_for_timeout(2500)
                content = page.inner_text("body")
                if (
                    "Join now" in content
                    and "Sign in" in content
                    and len(content) < 3000
                ):
                    blocked = True
            except PlaywrightTimeoutError:
                blocked = True
                content = ""
            except Exception:
                blocked = True
                content = ""
            finally:
                page.close()

            results.append({
                "query": query,
                "text": content[:12000],
                "blocked": blocked,
                "url": url,
            })

        browser.close()

    return results


# =========================
# PDF EXPORT
# =========================

def export_cover_letter_pdf(text: str) -> BytesIO:
    """Render a cover letter string as a downloadable PDF."""
    if not text or not text.strip():
        raise ValueError("Cover letter text must not be empty.")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]

    story = []
    for line in text.split("\n"):
        story.append(Paragraph(line or "&nbsp;", normal))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer


def _normalize_experience(experience: list[dict]) -> list[dict]:
    """
    Normalize experience entries to the grouped-roles schema.

    Accepts both the new schema (each entry has a 'roles' list) and the
    legacy flat schema (each entry has 'title', 'dates', 'bullets' directly)
    and always returns the grouped-roles format.
    """
    normalized = []
    for exp in experience:
        if "roles" in exp:
            normalized.append(exp)
        else:
            # Legacy flat schema — wrap in a single-role list
            normalized.append({
                "company":  exp.get("company", ""),
                "location": exp.get("location", ""),
                "roles": [{
                    "title":   exp.get("title", ""),
                    "dates":   exp.get("dates", ""),
                    "bullets": exp.get("bullets") or [],
                }],
            })
    return normalized


def export_resume_pdf(resume_data: dict, style_params: dict | None = None) -> BytesIO:
    """Render a structured resume dict as a polished, one-page PDF.

    Accepts both the new grouped-roles schema (experience[].roles[]) and the
    legacy flat schema (experience[].title / dates / bullets) via normalization.

    Uses KeepInFrame with mode='shrink' so content always fits on exactly
    one page regardless of how much the agent generated.

    Args:
        resume_data:  Structured resume dict from _generate_tailored_resume().
        style_params: Optional dict from extract_pdf_style() with keys
                      'accent_color' (hex str) and/or 'name_font_size' (float).
                      When provided, the output PDF will mirror the input's palette.

    Returns:
        In-memory BytesIO buffer positioned at the start, ready for download.
    """
    style_params = style_params or {}
    MARGIN = 0.50 * inch
    PAGE_W, PAGE_H = letter
    USABLE_W = PAGE_W - 2 * MARGIN
    USABLE_H = PAGE_H - 2 * MARGIN

    # ── Colour palette ────────────────────────────────────────────────────────
    # Use extracted accent color from the uploaded resume when available;
    # otherwise fall back to the default indigo palette.
    accent_hex = style_params.get("accent_color", "#4f46e5")
    name_size  = style_params.get("name_font_size", 17.0)

    NAVY   = colors.HexColor("#0f172a")
    SLATE  = colors.HexColor("#475569")
    DARK   = colors.HexColor("#1e293b")
    ACCENT = colors.HexColor(accent_hex)
    MID    = colors.HexColor("#64748b")

    # ── Paragraph styles ──────────────────────────────────────────────────────
    def S(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    name_s    = S("name",    fontName="Helvetica-Bold",    fontSize=name_size, leading=name_size + 4, textColor=NAVY,   spaceAfter=2)
    contact_s = S("contact", fontName="Helvetica",         fontSize=8.5,  leading=12, textColor=SLATE,  spaceAfter=2)
    sec_s     = S("sec",     fontName="Helvetica-Bold",    fontSize=7.5,  leading=10, textColor=ACCENT, spaceBefore=5, spaceAfter=3)
    co_s      = S("co",      fontName="Helvetica-Bold",    fontSize=9.5,  leading=12, textColor=NAVY,   spaceBefore=6, spaceAfter=1)
    role_s    = S("role",    fontName="Helvetica-Oblique", fontSize=8.5,  leading=11, textColor=SLATE,  spaceAfter=1)
    dates_s   = S("dates",   fontName="Helvetica",         fontSize=8.5,  leading=11, textColor=MID,    alignment=TA_RIGHT)
    bullet_s  = S("bullet",  fontName="Helvetica",         fontSize=8.5,  leading=11, textColor=DARK,   leftIndent=10, firstLineIndent=-8, spaceAfter=1)
    body_s    = S("body",    fontName="Helvetica",         fontSize=9,    leading=12, textColor=DARK,   spaceAfter=3)
    skills_s  = S("skills",  fontName="Helvetica",         fontSize=8.5,  leading=12, textColor=DARK,   spaceAfter=2)

    def divider():
        return HRFlowable(width="100%", thickness=0.4, color=ACCENT, spaceAfter=3, spaceBefore=2)

    def section(title: str) -> list:
        return [divider(), Paragraph(title.upper(), sec_s)]

    def company_banner(company: str, location: str) -> list:
        """Company name + location as a single-row banner (no dates — dates live on role rows)."""
        return [Paragraph(
            f"<b>{company}</b>&nbsp;&nbsp;<font color='#64748b' size='8'>{location}</font>",
            co_s,
        )]

    def role_row(title: str, dates: str) -> list:
        """Role title (left) and date range (right) on one line."""
        title_para = Paragraph(title, role_s)
        dates_para = Paragraph(dates, dates_s)
        tbl = Table([[title_para, dates_para]], colWidths=[USABLE_W * 0.65, USABLE_W * 0.35])
        tbl.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (0, 0), "LEFT"),
            ("ALIGN",         (1, 0), (1, 0), "RIGHT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        return [tbl]

    # ── Build story ───────────────────────────────────────────────────────────
    story: list = []

    # Header
    story.append(Paragraph(resume_data.get("candidate_name", ""), name_s))
    story.append(Paragraph(resume_data.get("contact_line", ""), contact_s))

    # Summary
    summary = (resume_data.get("summary") or "").strip()
    if summary:
        story += section("Professional Summary")
        story.append(Paragraph(summary, body_s))

    # Experience — grouped by company, multiple roles per banner
    raw_experience = resume_data.get("experience") or []
    experience = _normalize_experience(raw_experience)
    if experience:
        story += section("Experience")
        for exp in experience:
            story += company_banner(exp.get("company", ""), exp.get("location", ""))
            for role in exp.get("roles") or []:
                story += role_row(role.get("title", ""), role.get("dates", ""))
                for b in role.get("bullets") or []:
                    story.append(Paragraph(f"• {b}", bullet_s))

    # Skills
    skills = [s for s in (resume_data.get("skills") or []) if s]
    if skills:
        story += section("Skills")
        story.append(Paragraph(" &nbsp;·&nbsp; ".join(skills), skills_s))

    # Education
    education = resume_data.get("education") or []
    if education:
        story += section("Education")
        for edu in education:
            line = "  |  ".join(filter(None, [
                edu.get("school"), edu.get("degree"), edu.get("dates"),
            ]))
            story.append(Paragraph(line, body_s))

    # Certifications
    certs = [c for c in (resume_data.get("certifications") or []) if c]
    if certs:
        story += section("Certifications")
        story.append(Paragraph(" &nbsp;·&nbsp; ".join(certs), skills_s))

    # ── One-page enforcement via KeepInFrame(mode='shrink') ───────────────────
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
    )
    frame = KeepInFrame(USABLE_W, USABLE_H, story, mode="shrink")
    doc.build([frame])
    buffer.seek(0)
    return buffer
