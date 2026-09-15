from __future__ import annotations

from app.agents.state import InvestigationState


def judge(state: InvestigationState) -> InvestigationState:
    analyzable_tokens = sum(len(unit.text.split()) for unit in state.units if len(unit.text.split()) >= 8)
    affected_tokens = sum(len(match["submission_text"].split()) for match in state.verified_matches)
    coverage = round(min(100.0, affected_tokens / max(analyzable_tokens, 1) * 100), 1)
    uncited = [match for match in state.verified_matches if not match.get("citation_present")]
    average_score = sum(match["verification_score"] for match in uncited) / max(len(uncited), 1)
    risk_score = round(min(1.0, coverage / 100 * 0.6 + average_score * 0.4), 4)
    risk_level = "VERY HIGH" if risk_score >= 0.75 else "HIGH" if risk_score >= 0.5 else "MODERATE" if risk_score >= 0.25 else "LOW"
    state.final_verdict = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "coverage_percent": coverage,
        "strong_matches": sum(match["match_type"] == "EXACT_OR_LEXICAL_OVERLAP" for match in state.verified_matches),
        "possible_paraphrases": sum(match["match_type"] == "POSSIBLE_PARAPHRASE" for match in state.verified_matches),
        "false_positives": len(state.false_positives),
        "candidate_count": len(state.candidates),
        "matches": state.verified_matches[:10],
        "summary": f"{risk_level} similarity risk based on {len(state.verified_matches)} verified source matches covering {coverage}% of analyzable text.",
        "recommendations": [
            "Review each uncredited match and add source attribution." if uncited else "Confirm that cited matches accurately represent the referenced sources.",
            "Rewrite high-overlap passages in original language where attribution is insufficient." if state.verified_matches else "No immediate source-overlap action is recommended.",
        ],
    }
    if state.reasoner is not None:
        reasoning = state.reasoner.synthesize(state.final_verdict)
        if reasoning:
            state.final_verdict.update(reasoning)
            state.event("OpenAI Reasoning Agent", "synthesized_report", "Structured report summary generated from retrieved evidence")
        else:
            details = "OpenAI request failed; deterministic report retained" if state.reasoner.last_error else "OpenAI not configured; deterministic report retained"
            if state.reasoner.last_error:
                details += f" ({state.reasoner.last_error})"
            state.event("OpenAI Reasoning Agent", "fallback", details)
    state.event("Forensic Judge Agent", "issued_verdict", f"Risk {risk_level} from {len(state.verified_matches)} verified matches")
    return state
