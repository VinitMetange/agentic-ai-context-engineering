"""
Context Engine - 5-Layer Context Architecture for Agentic AI
Author: Vinit Metange | AI Product Leader
GitHub: https://github.com/VinitMetange
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger


class ContextLayer(Enum):
    SYSTEM = "system"        # Layer 1: System instructions & persona
    MEMORY = "memory"        # Layer 2: Long-term memory / episodic
    KNOWLEDGE = "knowledge"  # Layer 3: Retrieved knowledge (RAG)
    HISTORY = "history"      # Layer 4: Conversation history
    USER = "user"            # Layer 5: Current user input


@dataclass
class ContextWindow:
    """Represents a structured context window with 5-layer architecture."""
    system_prompt: str = ""
    memory_context: str = ""
    knowledge_context: str = ""
    history: List[BaseMessage] = field(default_factory=list)
    user_input: str = ""
    max_tokens: int = 8000
    token_budget: Dict[ContextLayer, float] = field(default_factory=lambda: {
        ContextLayer.SYSTEM: 0.15,
        ContextLayer.MEMORY: 0.10,
        ContextLayer.KNOWLEDGE: 0.40,
        ContextLayer.HISTORY: 0.25,
        ContextLayer.USER: 0.10,
    })

    def get_token_limits(self) -> Dict[str, int]:
        return {
            layer.value: int(self.max_tokens * budget)
            for layer, budget in self.token_budget.items()
        }


class ContextCompressor:
    """Handles context compression to fit within token limits."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def compress_history(self, messages: List[BaseMessage], max_tokens: int) -> List[BaseMessage]:
        """Compress conversation history using rolling summary."""
        if not messages:
            return []

        # Estimate token count (rough: 4 chars = 1 token)
        total_chars = sum(len(m.content) for m in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= max_tokens:
            return messages

        # Keep last N messages and summarize the rest
        keep_recent = 6
        to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []
        recent = messages[-keep_recent:] if len(messages) > keep_recent else messages

        if to_summarize:
            summary = self._summarize_messages(to_summarize)
            summary_msg = SystemMessage(content=f"[CONVERSATION SUMMARY]: {summary}")
            return [summary_msg] + recent

        return recent

    def _summarize_messages(self, messages: List[BaseMessage]) -> str:
        """Use LLM to summarize a list of messages."""
        conversation_text = "\n".join([
            f"{type(m).__name__}: {m.content[:200]}" for m in messages
        ])
        prompt = f"Summarize this conversation in 2-3 sentences, preserving key decisions and context:\n\n{conversation_text}"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def compress_knowledge(self, knowledge: str, max_tokens: int, query: str) -> str:
        """Compress retrieved knowledge chunks to most relevant content."""
        if len(knowledge) // 4 <= max_tokens:
            return knowledge

        prompt = f"""Extract the most relevant information for this query from the context below.
Keep only what directly answers the query. Be concise.

Query: {query}

Context:
{knowledge[:3000]}

Relevant excerpts:"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class ContextEngine:
    """
    Production-grade 5-layer context management engine.
    Manages context assembly, compression, and token budgeting.
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_context_tokens: int = 8000,
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.compressor = ContextCompressor(self.llm)
        self.max_tokens = max_context_tokens
        self._stats: Dict[str, Any] = {"compressions": 0, "total_builds": 0}

    def build_context(self, window: ContextWindow) -> List[BaseMessage]:
        """
        Build a token-budget-aware message list from the context window.
        Returns ordered messages ready for LLM consumption.
        """
        limits = window.get_token_limits()
        messages: List[BaseMessage] = []

        # Layer 1: System prompt
        system_parts = [window.system_prompt]
        if window.memory_context:
            system_parts.append(f"\n\n## Relevant Memory\n{window.memory_context}")
        messages.append(SystemMessage(content="\n".join(system_parts)))

        # Layer 3: Knowledge context as system context injection
        if window.knowledge_context:
            compressed_knowledge = self.compressor.compress_knowledge(
                window.knowledge_context,
                limits[ContextLayer.KNOWLEDGE.value],
                window.user_input
            )
            messages.append(SystemMessage(
                content=f"## Retrieved Knowledge\n{compressed_knowledge}"
            ))

        # Layer 4: Compressed conversation history
        compressed_history = self.compressor.compress_history(
            window.history, limits[ContextLayer.HISTORY.value] * 4  # tokens to chars
        )
        messages.extend(compressed_history)

        # Layer 5: Current user input
        messages.append(HumanMessage(content=window.user_input))

        self._stats["total_builds"] += 1
        logger.debug(f"Context built: {len(messages)} messages")
        return messages

    def get_stats(self) -> Dict[str, Any]:
        return self._stats


if __name__ == "__main__":
    # Demo usage
    engine = ContextEngine(max_context_tokens=4000)
    window = ContextWindow(
        system_prompt="You are a helpful AI assistant specializing in cloud architecture.",
        memory_context="User previously asked about Kubernetes deployments.",
        knowledge_context="LangGraph is a framework for building stateful multi-actor applications...",
        history=[
            HumanMessage(content="What is LangGraph?"),
            AIMessage(content="LangGraph is a library for building stateful LLM applications."),
        ],
        user_input="How do I implement a multi-agent workflow?"
    )
    messages = engine.build_context(window)
    print(f"Built context with {len(messages)} messages")
    for i, msg in enumerate(messages):
        print(f"  [{i}] {type(msg).__name__}: {msg.content[:80]}...")
