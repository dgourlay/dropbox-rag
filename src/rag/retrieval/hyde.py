from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.config import SummarizationConfig
    from rag.protocols import Embedder

logger = logging.getLogger(__name__)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

_HYDE_PROMPT = """\
Answer the following question in 2-3 sentences. \
Write a direct, factual answer as if you had full knowledge of the topic. \
Do not hedge or say you don't know.

Question: {query}
"""


def generate_hypothetical_answer(
    query: str,
    config: SummarizationConfig,
) -> str | None:
    """Generate a hypothetical answer using the LLM CLI.

    Returns the generated text, or None if the LLM is unavailable or fails.
    Reuses the same CLI tool and preset pattern as CliSummarizer.
    """
    if not config.enabled:
        return None

    command = config.command
    if shutil.which(command) is None:
        logger.debug("HyDE: CLI command %s not found, skipping", command)
        return None

    prompt = _HYDE_PROMPT.format(query=query)
    args = config.args or []
    input_mode = config.input_mode or "stdin"

    if input_mode == "arg":
        cmd = [command, *args, prompt]
        stdin_text = None
    else:
        cmd = [command, *args]
        stdin_text = prompt

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            env=env,
        )
        if result.returncode != 0:
            logger.debug(
                "HyDE: CLI exited with code %d, skipping. stderr: %s",
                result.returncode,
                result.stderr[:200],
            )
            return None

        output = _ANSI_ESCAPE_RE.sub("", result.stdout).strip()
        if not output:
            return None
        return output

    except subprocess.TimeoutExpired:
        logger.debug("HyDE: CLI timed out after %ds", config.timeout_seconds)
        return None
    except FileNotFoundError:
        logger.debug("HyDE: CLI command %s not found", command)
        return None


def hyde_embed(
    query: str,
    embedder: Embedder,
    config: SummarizationConfig,
) -> list[float] | None:
    """Generate a hypothetical answer and embed it.

    Returns the embedding vector of the hypothetical answer, or None if
    generation fails (falls back silently to raw query embedding).
    """
    answer = generate_hypothetical_answer(query, config)
    if answer is None:
        return None

    logger.debug("HyDE: generated hypothetical answer (%d chars)", len(answer))
    return embedder.embed_query(answer)
