from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.citations import check_citations
from app.agents.decomposer import decompose
from app.agents.filters import filter_false_positives
from app.agents.investigator import investigate
from app.agents.judge import judge
from app.agents.state import InvestigationState
from app.agents.verifier import verify_candidates


class GraphState(TypedDict):
    investigation: InvestigationState


def _node(agent):
    def run(data: GraphState) -> GraphState:
        return {"investigation": agent(data["investigation"])}

    return run


def build_workflow():
    graph = StateGraph(GraphState)
    graph.add_node("decomposer", _node(decompose))
    graph.add_node("investigator", _node(investigate))
    graph.add_node("verifier", _node(verify_candidates))
    graph.add_node("false_positive_filter", _node(filter_false_positives))
    graph.add_node("citation_checker", _node(check_citations))
    graph.add_node("judge", _node(judge))
    graph.add_edge(START, "decomposer")
    graph.add_edge("decomposer", "investigator")
    graph.add_edge("investigator", "verifier")
    graph.add_edge("verifier", "false_positive_filter")
    graph.add_edge("false_positive_filter", "citation_checker")
    graph.add_edge("citation_checker", "judge")
    graph.add_edge("judge", END)
    return graph.compile()
