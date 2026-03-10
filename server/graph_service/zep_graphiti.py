import logging
import os
from typing import Annotated, Any

from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.llm_client.openai_client import OpenAIClient  # type: ignore
from graphiti_core.llm_client.openai_base_client import DEFAULT_MAX_TOKENS  # type: ignore
from graphiti_core.llm_client.config import LLMConfig, ModelSize  # type: ignore
from graphiti_core.prompts.models import Message as LLMMessage  # type: ignore
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)

# nomic-embed-text dimension
NOMIC_DIM = 768


class NoThinkOpenAIClient(OpenAIClient):
    """OpenAIClient for local thinking models (Qwen3.5, DeepSeek-R1, etc.).

    Root cause of ingestion failures:
      1. response_format=json_object (grammar constraint) + chat_template_kwargs
         triggers llama-server error 500 "context does not logits computation".
      2. _create_structured_completion calls responses.parse (OpenAI Responses API)
         which llama-server does not support at all.

    Fix: override _generate_response to:
      - use chat.completions.create directly (no grammar, no Responses API)
      - inject enable_thinking=false via extra_body
      - always use _handle_json_response (works with ChatCompletion)
    """

    _NO_THINK_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}

    # These are required by the abstract base but are never reached when
    # _generate_response is overridden — implemented as thin pass-throughs.
    async def _create_completion(self, model, messages, temperature, max_tokens,
                                 response_model=None, reasoning=None, verbosity=None) -> Any:
        return await self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, extra_body=self._NO_THINK_EXTRA,
        )

    async def _create_structured_completion(self, model, messages, temperature,
                                            max_tokens, response_model,
                                            reasoning=None, verbosity=None) -> Any:
        return await self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, extra_body=self._NO_THINK_EXTRA,
        )

    async def _generate_response(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> tuple[dict[str, Any], int, int]:
        """Always use plain chat.completions — no grammar, no Responses API."""
        openai_messages = self._convert_messages_to_openai_format(messages)
        model = self._get_model_for_size(model_size)
        response = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            extra_body=self._NO_THINK_EXTRA,
        )
        # Strip markdown code fences before JSON parsing.
        # Local models (Qwen3.5, Llama, etc.) often wrap output in ```json ... ```
        # which causes json.loads to fail in _handle_json_response.
        content = response.choices[0].message.content or ''
        content = _strip_markdown_json(content)
        response.choices[0].message.content = content

        return self._handle_json_response(response)


def _strip_markdown_json(text: str) -> str:
    """Remove markdown code fences from JSON output.

    Converts:  ```json\\n{...}\\n```   →   {...}
    Also handles: ```\\n{...}\\n```
    """
    import re
    text = text.strip()
    # Remove opening fence: ```json or ```
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def _make_llm_client(settings) -> LLMClient:
    """Create LLM client — NoThinkOpenAIClient when DISABLE_LLM_THINKING=true."""
    disable_thinking = os.environ.get('DISABLE_LLM_THINKING', 'false').lower() == 'true'
    config = LLMConfig(
        api_key=settings.openai_api_key or 'ollama',
        base_url=settings.openai_base_url,
        model=settings.model_name,
    )
    client_cls = NoThinkOpenAIClient if disable_thinking else OpenAIClient
    logger.info(f'LLM client: {client_cls.__name__} model={settings.model_name} url={settings.openai_base_url}')
    return client_cls(config=config)


def _make_embedder(settings):
    """Create an OpenAI-compatible embedder using EMBEDDING_BASE_URL (or OPENAI_BASE_URL as fallback)."""
    base_url = settings.embedding_base_url or settings.openai_base_url
    api_key = settings.openai_api_key or 'ollama'
    model = settings.embedding_model_name or 'nomic-embed-text'

    if base_url:
        cfg = OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=base_url,
            embedding_model=model,
            embedding_dim=NOMIC_DIM,
        )
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        logger.info(f'Embedder: model={model} dim={NOMIC_DIM} url={base_url}')
        return OpenAIEmbedder(config=cfg, client=client)
    return None


class ZepGraphiti(Graphiti):
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
        super().__init__(uri, user, password, llm_client)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


async def get_graphiti(settings: ZepEnvDep):
    llm_client = _make_llm_client(settings)
    embedder = _make_embedder(settings)
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client,
    )
    if embedder is not None:
        client.embedder = embedder

    try:
        yield client
    finally:
        await client.close()


async def initialize_graphiti(settings: ZepEnvDep):
    llm_client = _make_llm_client(settings)
    embedder = _make_embedder(settings)
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client,
    )
    if embedder is not None:
        client.embedder = embedder
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
