"""
Agentic AI Context Engineering - Main Agent
Author: Vinit Metange | AI Product Leader
GitHub: https://github.com/VinitMetange
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from loguru import logger

from context_engine import ContextEngine, ContextWindow

load_dotenv()


class AgentState(TypedDict):
    messages: List[BaseMessage]
    memory: str
    retrieved_knowledge: str
    current_input: str
    iteration: int
    final_answer: Optional[str]


class ContextAwareAgent:
    """
    Multi-turn agent using 5-layer context architecture.
    Maintains persistent memory across conversation turns.
    """

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 8000):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.context_engine = ContextEngine(llm=self.llm, max_context_tokens=max_tokens)
        self.memory_store: List[str] = []
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("think", self._think_node)
        workflow.add_node("respond", self._respond_node)
        workflow.add_node("memorize", self._memorize_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "think")
        workflow.add_edge("think", "respond")
        workflow.add_edge("respond", "memorize")
        workflow.add_edge("memorize", END)

        return workflow.compile()

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant knowledge for current query."""
        # In production: integrate with vector store (Chroma, pgvector, etc.)
        logger.info(f"Retrieving context for: {state['current_input'][:50]}")
        state["retrieved_knowledge"] = "[Knowledge retrieved from vector store]"
        return state

    def _think_node(self, state: AgentState) -> AgentState:
        """Build context and reason about the query."""
        memory_summary = "\n".join(self.memory_store[-3:]) if self.memory_store else ""

        window = ContextWindow(
            system_prompt=(
                "You are an expert AI assistant with deep knowledge of cloud architecture, "
                "agentic AI systems, and enterprise software. Be precise and technical."
            ),
            memory_context=memory_summary,
            knowledge_context=state["retrieved_knowledge"],
            history=state["messages"][-10:],
            user_input=state["current_input"],
        )

        context_messages = self.context_engine.build_context(window)
        response = self.llm.invoke(context_messages)
        state["final_answer"] = response.content
        return state

    def _respond_node(self, state: AgentState) -> AgentState:
        """Append response to message history."""
        state["messages"].append(HumanMessage(content=state["current_input"]))
        state["messages"].append(AIMessage(content=state["final_answer"]))
        state["iteration"] += 1
        return state

    def _memorize_node(self, state: AgentState) -> AgentState:
        """Extract and store key information in memory."""
        if state["iteration"] % 3 == 0:  # Memorize every 3 turns
            summary = f"Turn {state['iteration']}: {state['current_input'][:100]}"
            self.memory_store.append(summary)
            logger.debug(f"Memory updated: {len(self.memory_store)} entries")
        return state

    def chat(self, user_input: str, state: Optional[AgentState] = None) -> str:
        """Process a user message and return the agent's response."""
        if state is None:
            state = AgentState(
                messages=[],
                memory="",
                retrieved_knowledge="",
                current_input=user_input,
                iteration=0,
                final_answer=None,
            )
        else:
            state["current_input"] = user_input

        result = self.graph.invoke(state)
        return result["final_answer"]


if __name__ == "__main__":
    agent = ContextAwareAgent()
    print("Context-Aware Agent initialized. Type 'quit' to exit.")
    state = None
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        response = agent.chat(user_input, state)
        print(f"\nAgent: {response}")
