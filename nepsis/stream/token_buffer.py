"""Token buffering and semantic chunking for stream processing.

Implements hybrid chunking strategy:
- Primary: Detect semantic boundaries (sentence/clause end)
- Fallback: Max buffer size to prevent blocking
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterator
import re


@dataclass
class TokenChunk:
    """A semantic chunk of tokens ready for processing."""
    text: str
    tokens: List[str]
    token_count: int
    is_complete: bool  # True if ended on semantic boundary
    start_idx: int  # Global token index of first token
    end_idx: int  # Global token index of last token

    @property
    def is_sentence_end(self) -> bool:
        """Check if chunk ends with sentence boundary."""
        return self.text.rstrip().endswith(('.', '!', '?', '...'))


class TokenBuffer:
    """Buffers tokens and emits semantic chunks.

    Chunking Modes:
        - 'hybrid': Semantic boundaries OR max buffer (recommended)
        - 'semantic': Only emit on sentence/clause boundaries
        - 'fixed': Emit every N tokens regardless of semantics
    """

    # Semantic boundary patterns
    SENTENCE_END = re.compile(r'[.!?]+[\s\n]*$')
    CLAUSE_END = re.compile(r'[,;:—][\s\n]*$')

    def __init__(
        self,
        mode: str = "hybrid",
        max_buffer_tokens: int = 100,
        emit_clause_boundaries: bool = False
    ):
        """Initialize token buffer.

        Args:
            mode: Chunking mode ('hybrid', 'semantic', 'fixed')
            max_buffer_tokens: Max tokens before forced emit
            emit_clause_boundaries: Also emit on clause (comma, semicolon)
        """
        self.mode = mode
        self.max_buffer_tokens = max_buffer_tokens
        self.emit_clause_boundaries = emit_clause_boundaries

        self.buffer: List[str] = []
        self.global_token_idx = 0
        self.chunk_start_idx = 0

    def add_token(self, token: str) -> Optional[TokenChunk]:
        """Add token to buffer, return chunk if ready.

        Args:
            token: New token from LLM stream

        Returns:
            TokenChunk if ready to emit, None otherwise
        """
        self.buffer.append(token)
        current_text = ''.join(self.buffer)

        # Check emission conditions
        should_emit = False
        is_complete = False

        if self.mode == 'fixed':
            # Emit every N tokens
            should_emit = len(self.buffer) >= self.max_buffer_tokens
            is_complete = False

        elif self.mode == 'semantic':
            # Only emit on semantic boundaries
            if self._is_semantic_boundary(current_text):
                should_emit = True
                is_complete = True
            # Force emit if buffer exceeds max (fallback)
            elif len(self.buffer) >= self.max_buffer_tokens * 2:
                should_emit = True
                is_complete = False

        elif self.mode == 'hybrid':
            # Prefer semantic, fallback to max buffer
            if self._is_semantic_boundary(current_text):
                should_emit = True
                is_complete = True
            elif len(self.buffer) >= self.max_buffer_tokens:
                should_emit = True
                is_complete = False

        if should_emit:
            chunk = self._emit_chunk(is_complete)
            self.global_token_idx += 1
            return chunk

        self.global_token_idx += 1
        return None

    def flush(self) -> Optional[TokenChunk]:
        """Flush remaining buffer as final chunk.

        Returns:
            Final TokenChunk if buffer not empty
        """
        if not self.buffer:
            return None
        return self._emit_chunk(is_complete=True)

    def _is_semantic_boundary(self, text: str) -> bool:
        """Check if text ends with semantic boundary.

        Args:
            text: Current buffer text

        Returns:
            True if ends with sentence or clause boundary
        """
        # Always emit on sentence end
        if self.SENTENCE_END.search(text):
            return True

        # Optionally emit on clause end
        if self.emit_clause_boundaries and self.CLAUSE_END.search(text):
            return True

        return False

    def _emit_chunk(self, is_complete: bool) -> TokenChunk:
        """Emit current buffer as chunk and reset.

        Args:
            is_complete: Whether chunk ended on semantic boundary

        Returns:
            TokenChunk from current buffer
        """
        chunk = TokenChunk(
            text=''.join(self.buffer),
            tokens=self.buffer.copy(),
            token_count=len(self.buffer),
            is_complete=is_complete,
            start_idx=self.chunk_start_idx,
            end_idx=self.global_token_idx
        )

        # Reset buffer
        self.buffer = []
        self.chunk_start_idx = self.global_token_idx + 1

        return chunk


def chunk_tokens(
    tokens: Iterator[str],
    mode: str = "hybrid",
    max_buffer_tokens: int = 100
) -> Iterator[TokenChunk]:
    """Convenience function to chunk token stream.

    Args:
        tokens: Iterator of token strings
        mode: Chunking mode
        max_buffer_tokens: Max buffer size

    Yields:
        TokenChunk objects

    Example:
        >>> tokens = ["The", " quick", " brown", " fox", ".", " Jumps", "."]
        >>> for chunk in chunk_tokens(iter(tokens), mode="hybrid"):
        ...     print(f"Chunk: '{chunk.text}' (complete={chunk.is_complete})")
        Chunk: 'The quick brown fox.' (complete=True)
        Chunk: ' Jumps.' (complete=True)
    """
    buffer = TokenBuffer(mode=mode, max_buffer_tokens=max_buffer_tokens)

    for token in tokens:
        chunk = buffer.add_token(token)
        if chunk:
            yield chunk

    # Flush final chunk
    final = buffer.flush()
    if final:
        yield final
