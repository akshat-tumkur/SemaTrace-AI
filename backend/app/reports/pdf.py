"""PDF from saved evidence only. No retrieval, providers, or model execution."""

from io import BytesIO
from pathlib import Path
from functools import lru_cache
from html import escape
from threading import Lock
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    LongTable,
    TableStyle,
    KeepTogether,
)

ASSETS = Path(__file__).with_name("assets")
_font_lock = Lock()


@lru_cache(maxsize=1)
def fonts():
    with _font_lock:
        for name, file in [
            ("Noto", "NotoSans-Regular.ttf"),
            ("NotoBold", "NotoSans-Bold.ttf"),
        ]:
            pdfmetrics.registerFont(TTFont(name, str(ASSETS / file)))
        pdfmetrics.registerFontFamily(
            "Noto", normal="Noto", bold="NotoBold", italic="Noto", boldItalic="NotoBold"
        )
    return pdfmetrics.getFont("Noto").face.charToGlyph


def render_pdf(saved: dict) -> bytes:
    report = saved.get("similarity_report")
    if not isinstance(report, dict):
        raise ValueError(
            "This legacy record has no saved textual similarity report. Recalculate it before PDF export."
        )
    entries = report.get("ledger", [])
    if len(entries) > 1000:
        raise ValueError(
            "PDF export is limited to 1000 evidence entries; use the complete JSON report."
        )
    glyphs = fonts()
    fallback = set()

    def clean(value):
        value = str(value if value is not None else "Not recorded")
        out = []
        for c in value:
            if c in "\n\r\t":
                out.append(" ")
            elif ord(c) < 32:
                out.append(" ")
            elif ord(c) in glyphs:
                out.append(c)
            else:
                fallback.add(ord(c))
                out.append("[U+" + format(ord(c), "04X") + "]")
        return escape("".join(out))

    style = ParagraphStyle(
        "body",
        fontName="Noto",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#23344B"),
        spaceAfter=6,
        splitLongWords=True,
    )
    title = ParagraphStyle(
        "title",
        parent=style,
        fontName="NotoBold",
        fontSize=23,
        leading=29,
        spaceAfter=14,
    )
    heading = ParagraphStyle(
        "heading",
        parent=style,
        fontName="NotoBold",
        fontSize=12,
        leading=17,
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True,
    )
    small = ParagraphStyle("small", parent=style, fontSize=8, leading=11, spaceAfter=4)
    label = ParagraphStyle("label", parent=small, keepWithNext=True)
    excerpt = ParagraphStyle(
        "excerpt",
        parent=style,
        backColor=colors.HexColor("#FFF3CE"),
        borderPadding=7,
        spaceAfter=12,
    )

    def p(value, kind=style):
        return Paragraph(clean(value), kind)

    story = [
        p("SEMATRACE AI", small),
        p("Similarity report", title),
        p(saved.get("filename", "Untitled document")),
        p(
            "Analysis "
            + str(saved.get("analysis_id", "Not recorded"))
            + "  |  Created "
            + str(saved.get("created_at") or "Not recorded in this legacy report"),
            small,
        ),
        p(
            "Saved matching-word evidence and semantic source comparisons for human review. No semantic percentage or misconduct verdict.",
            small,
        ),
    ]
    assessment = saved.get("semantic_assessment") or {}
    ai_findings = assessment.get("findings", [])
    if assessment:
        story += [
            p("AI meaning and paraphrase review", heading),
            p(
                f"Status: {assessment.get('status', 'not requested')} | {len(ai_findings)} advisory correspondences | {len(assessment.get('reviewed_unit_starts', []))} of {assessment.get('eligible_units', 0)} eligible units reviewed; {assessment.get('remaining_units', 0)} remaining."
            ),
            p(
                "Provider/model: "
                + str(assessment.get("model", "none"))
                + ". Literal quotes are grounded; AI interpretations remain unverified and do not prove copying.",
                small,
            ),
        ]
        for warning in assessment.get("warnings", []):
            story.append(p(warning, small))
    nominations = saved.get("related_sources", [])
    first = {}
    for row in nominations:
        at = row.get("submission_start")
        if row.get("minilm_rank", 1) == 1 and at not in first:
            first[at] = row
    counts = {}
    source_names = {}
    source_scores = {}
    for row in first.values():
        sid = row["source_document_id"]
        counts[sid] = counts.get(sid, 0) + 1
        source_names[sid] = row.get("source_title", sid)
        source_scores[sid] = max(
            source_scores.get(sid, float("-inf")), row.get("score", 0)
        )
    if counts:
        story += [p("Retrieved sources to inspect", heading)]
        for sid in sorted(counts, key=lambda s: (-counts[s], -source_scores[s], s))[:4]:
            story.append(
                p(
                    f"{source_names[sid]}: first-ranked for {counts[sid]} of {len(first)} submitted text units."
                )
            )
        story.append(
            p(
                "These are retrieval counts, not confirmed reused sentences or semantic coverage. Zero matching-word coverage does not rule out paraphrasing. Source passage comparisons follow below.",
                small,
            )
        )
    stats = [
        ["MATCHING-WORD COVERAGE", "MATCHED WORDS", "ELIGIBLE WORDS"],
        [
            str(report.get("textual_similarity_percent", 0)) + "%",
            str(report.get("unique_matched_words", 0)),
            str(report.get("total_eligible_words", 0)),
        ],
    ]
    table = LongTable([[p(v) for v in row] for row in stats], colWidths=[165, 165, 165])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EDF5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story += [Spacer(1, 12), table]
    corpus = saved.get("corpus") or {}
    scope = (
        f"{corpus.get('source_origins', 'Not recorded')} source documents | {corpus.get('unique_passages', 'Not recorded')} indexed passages"
        if corpus
        else "Corpus identity was not recorded in this legacy report."
    )
    story += [
        p("Corpus and scope", heading),
        p(scope),
        p("Snapshot: " + str(corpus.get("version", "Not recorded")), small),
        p(
            "Only accessible, indexed sources and retained candidates were examined. Missing matches do not prove originality. A zero textual score does not guarantee originality. Similarity does not establish copying history or misconduct."
        ),
    ]
    story += [p("Source contributions", heading)]
    sources = report.get("sources", [])
    for source in sources:
        attribution = source.get("attribution") or {}
        attribution_text = (
            "; ".join(
                str(v)
                for v in (
                    attribution.get("attribution_text"),
                    attribution.get("license_name"),
                    attribution.get("license_url"),
                )
                if v
            )
            or "See original source record; license not recorded here"
        )
        story += [
            p(str(source.get("name") or source.get("source_id")), heading),
            p(
                "Primary contribution: "
                + str(source.get("attributed_percent", 0))
                + "% | Overlapping raw contribution: "
                + str(source.get("raw_overlap_percent", 0))
                + "% | "
                + ("Excluded" if source.get("excluded") else "Included"),
                small,
            ),
            p(source.get("url", ""), small),
            p("Source version: " + str(source.get("source_id")), small),
            p("Attribution: " + attribution_text, small),
        ]
    if not sources:
        story.append(p("No source has recorded textual overlap."))
    options = report.get("options", {})
    excluded = report.get("exclusions", {})
    story += [
        p("Applied exclusions", heading),
        p(
            f"Minimum contiguous match: {options.get('minimum_match_words', 'Not recorded')} words. Excluded document words: {excluded.get('excluded_document_words', 0)}. Excluded source versions: {len(options.get('excluded_source_ids', []))}.",
            small,
        ),
        p(
            "Source and minimum-length exclusions affect matched words. Explicit document ranges remove intersecting words from both numerator and denominator. Exclusions remain visible in the evidence below.",
            small,
        ),
        p("Matched passage evidence", heading),
    ]
    for row in options.get("excluded_ranges", []):
        story.append(
            p(
                f"Document exclusion [{row.get('start')}, {row.get('end')}): {row.get('reason', 'User selection')}",
                small,
            )
        )
    source_names = {
        r.get("source_id"): r.get("name") or r.get("source_id") for r in sources
    }
    for i, row in enumerate(entries, 1):
        status = (
            "INCLUDED"
            if row.get("score_eligible")
            else "EXCLUDED: " + str(row.get("exclusion_reason") or "Not eligible")
        )
        story += [
            p(
                f"{i}. {status} | {source_names.get(row.get('source_document_id'), 'Unknown source')}",
                heading,
            ),
            p(
                f"Submission [{row.get('document_start')}, {row.get('document_end')}) | Source [{row.get('source_start')}, {row.get('source_end')})",
                small,
            ),
            p("Submitted text", label),
            p(row.get("submission_excerpt", ""), excerpt),
            p("Source text", label),
            p(row.get("source_excerpt", ""), excerpt),
        ]
    if not entries:
        story.append(p("No grounded textual passages are recorded."))
    if ai_findings:
        story += [
            p("AI-identified meaning correspondences", heading),
            p(
                f"Showing {min(40, len(ai_findings))} of {len(ai_findings)} saved findings. The complete findings and candidate pools are in JSON. These findings receive no matching-word coverage credit."
            ),
        ]
        ai_style = ParagraphStyle(
            "ai", parent=excerpt, backColor=colors.HexColor("#EDE4FA")
        )
        for i, finding in enumerate(ai_findings[:40], 1):
            card = [
                p(
                    f"{i}. {finding.get('source_title', 'Source')} | {finding.get('relation', 'Potential correspondence')}",
                    heading,
                ),
                p("Submitted evidence", label),
                p(finding.get("submission_excerpt", ""), ai_style),
                p("Literal source evidence", label),
                p(finding.get("source_excerpt", ""), ai_style),
                p("AI explanation: " + str(finding.get("explanation", "")), small),
            ]
            for difference in finding.get("differences", []):
                card.append(p("Changed/limited detail: " + difference, small))
            card += [
                p(
                    f"Submission [{finding.get('submission_start')}, {finding.get('submission_end')}); original extracted [{finding.get('original_submission_start')}, {finding.get('original_submission_end')}). Source [{finding.get('source_start')}, {finding.get('source_end')}).",
                    small,
                ),
                p(
                    "Source version: "
                    + str(finding.get("source_document_id"))
                    + " | revision: "
                    + str(finding.get("source_revision")),
                    small,
                ),
                p(finding.get("source_url", ""), small),
            ]
            attribution = finding.get("source_attribution") or {}
            card.append(
                p(
                    "Attribution: "
                    + "; ".join(
                        str(v)
                        for v in (
                            ", ".join(attribution.get("authors", [])),
                            attribution.get("license_name"),
                            attribution.get("attribution_text"),
                        )
                        if v
                    ),
                    small,
                )
            )
            story.append(KeepTogether(card))
    story += [
        p("Potential semantic correspondences", heading),
        p(
            "Automatically retrieved MiniLM context for human exploration. Purple excerpts are context, not proven paraphrase boundaries. Ranking scores are not plagiarism probabilities. Semantic candidates receive no textual credit. No semantic or combined percentage is calculated."
        ),
    ]
    semantic_style = ParagraphStyle(
        "semantic", parent=excerpt, backColor=colors.HexColor("#EDE4FA")
    )
    review = saved.get("semantic_review", {})
    investigations = review.get("investigations", {})

    # Bounded PDF rendering; complete candidate pools remain in saved JSON.
    def investigation_for(row):
        return investigations.get(
            "openai:" + str(row.get("candidate_id"))
        ) or investigations.get(row.get("candidate_id"))

    shown = [r for r in nominations if investigation_for(r)]
    if not ai_findings:
        shown += [
            r
            for r in sorted(first.values(), key=lambda r: -r.get("score", 0))
            if r not in shown
        ]
    shown = shown[:20]
    story.append(
        p(
            f"{len(nominations)} retrieved candidates saved. This PDF includes {len(shown)} selected/top-ranked contexts, capped at20. Complete alternatives and tool context are in JSON.",
            small,
        )
    )
    for r in shown:
        rank = (
            review.get("rankings", {})
            .get(str(r.get("submission_start")), {})
            .get("ranking", [])
        )
        position = next(
            (
                i
                for i, v in enumerate(rank, 1)
                if v["candidate_id"] == r.get("candidate_id")
            ),
            None,
        )
        story += [
            p(r.get("source_title", "Source context"), heading),
            p(
                f"MiniLM rank {r.get('minilm_rank', 1)} | cosine {r.get('score', 0):.3f}"
                + (f" | experimental RoBERTa rank {position}" if position else ""),
                small,
            ),
            p("Submitted context", label),
            p(r.get("submission_excerpt", ""), semantic_style),
            p("Indexed source context", label),
            p(r.get("source_excerpt", ""), semantic_style),
            p(
                f"Submission normalized [{r.get('submission_start')}, {r.get('submission_end')}); original extracted [{r.get('original_submission_start', 'not recorded')}, {r.get('original_submission_end', 'not recorded')}). Source indexed [{r.get('source_start')}, {r.get('source_end')}).",
                small,
            ),
            p(
                "Source ID: "
                + str(r.get("source_document_id"))
                + " | revision: "
                + str(r.get("source_revision")),
                small,
            ),
            p("SHA256: " + str(r.get("source_document_sha256")), small),
            p(r.get("source_url", ""), small),
        ]
        attribution = r.get("source_attribution") or {}
        story.append(
            p(
                "Attribution: "
                + "; ".join(
                    str(v)
                    for v in (
                        ", ".join(attribution.get("authors", [])),
                        attribution.get("license_name"),
                        attribution.get("attribution_text"),
                    )
                    if v
                ),
                small,
            )
        )
        story.append(
            p(
                "Citation marker: "
                + str(r.get("citation_context") or "not detected")
                + ". Claim support not assessed.",
                small,
            )
        )
        investigation = investigation_for(r)
        if investigation:
            e = investigation["explanation"]
            story += [
                p("Optional evidence investigation", heading),
                p(e.get("warning", "Unverified model explanation.")),
            ]
            if e.get("submission_quote"):
                story += [
                    p("Grounded submission quote: " + e["submission_quote"]),
                    p("Grounded source quote: " + e["source_quote"]),
                    p(
                        "Explanation source: "
                        + str(e.get("source_document_id"))
                        + " | revision: "
                        + str(e.get("source_revision")),
                        small,
                    ),
                    p("Model-reported similarity: " + e.get("similarities", "")),
                    p("Differences: " + e.get("differences", "")),
                    p("Uncertainty: " + e.get("uncertainty", "")),
                ]
            story.append(
                p(
                    "Tool trace: "
                    + " -> ".join(
                        t.get("tool") or t.get("action") or t["node"]
                        for t in investigation.get("trace", [])
                    ),
                    small,
                )
            )
    story += [
        p("Unresolved findings", heading),
        p(
            str(len(report.get("unresolved_findings", [])))
            + " unresolved or excluded candidates. These receive no textual credit. Detailed statuses remain available in the saved JSON.",
            small,
        ),
    ]
    methodology = [
        p("Methodology and limitations", heading),
        p(report.get("methodology", "Saved word-overlap accounting."), small),
    ]
    for line in report.get("limitations", []):
        methodology.append(p(line, small))
    methodology.append(
        p(
            "Highlights refer to extracted Unicode text, not original PDF page coordinates. Yellow excerpts mark grounded word runs; larger passage context does not imply every word was copied.",
            small,
        )
    )
    story.append(KeepTogether(methodology))
    if fallback:
        story.append(
            p(
                "Font coverage: unsupported characters are explicitly represented as Unicode code points: "
                + ", ".join("U+" + format(c, "04X") for c in sorted(fallback))
                + ". Original text remains intact in the saved JSON.",
                small,
            )
        )
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=48,
        bottomMargin=48,
        title="SemaTrace similarity report",
        author="SemaTrace AI",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Noto", 8)
        canvas.setFillColor(colors.HexColor("#65758B"))
        canvas.drawString(
            50, 26, "SemaTrace AI | Similarity evidence, not a misconduct determination"
        )
        canvas.drawRightString(A4[0] - 50, 26, str(doc.page))
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
