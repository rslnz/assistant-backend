import logging
from enum import Flag, auto, Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class TagProcessingMode(Flag):
    STREAM = auto()
    BUFFER = auto()
    STREAM_AND_BUFFER = STREAM | BUFFER

class TagConfig(BaseModel):
    mode: TagProcessingMode = Field(default=TagProcessingMode.BUFFER)
    buffer_tag: Optional[str] = Field(default=None)

    @field_validator('buffer_tag')
    @classmethod
    def validate_buffer_tag(cls, v: Optional[str], info):
        if TagProcessingMode.STREAM_AND_BUFFER in info.data.get('mode', TagProcessingMode.BUFFER) and not v:
            raise ValueError("If mode is STREAM_AND_BUFFER, buffer_tag must be specified.")
        return v

class ProcessingState(Enum):
    LOOKING_FOR_TAG = 1
    READING_TAG = 2
    READING_CONTENT = 3

class TagProcessor:
    def __init__(self, tag_config: Dict[str, TagConfig]):
        self.tag_config = tag_config
        self.state = ProcessingState.LOOKING_FOR_TAG
        self.current_tag = ""
        self.content_buffer = ""
        self.tag_buffer = ""
        self.debug_content = ""

    async def process_stream(self, token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            async for token in token_stream:
                self.debug_content += token
                for result in self._process_token(token):
                    if result:
                        yield result

            for result in self._flush_buffers():
                if result:
                    yield result

            if self.debug_content:
                yield {"tag": "debug", "content": self.debug_content}
        finally:
            self.reset()

    def _process_token(self, token: str) -> List[Dict[str, Any]]:
        results = []
        if self.state == ProcessingState.LOOKING_FOR_TAG:
            if token.startswith('§'):
                results.extend(self._flush_buffers())
                self.state = ProcessingState.READING_TAG
                self.tag_buffer = token[1:]
            else:
                self.content_buffer += token
        elif self.state == ProcessingState.READING_TAG:
            if '\n' in token:
                parts = token.split('\n', 1)
                self.tag_buffer += parts[0]
                self.current_tag = self.tag_buffer.strip()
                self.tag_buffer = ""
                self.state = ProcessingState.READING_CONTENT
                if len(parts) > 1:
                    self.content_buffer += parts[1]
                config = self.tag_config.get(self.current_tag, TagConfig())
                if self._is_streaming_config(config):
                    results.append(self._create_result(self.current_tag, ""))
            else:
                self.tag_buffer += token
        elif self.state == ProcessingState.READING_CONTENT:
            if '§' in token:
                parts = token.split('§', 1)
                self.content_buffer += parts[0]
                results.extend(self._flush_buffers())
                self.state = ProcessingState.READING_TAG
                self.tag_buffer = parts[1] if len(parts) > 1 else ""
            else:
                self.content_buffer += token
                config = self.tag_config.get(self.current_tag, TagConfig())
                if self._is_streaming_config(config):
                    results.append(self._create_result(self.current_tag, token))
        return results

    def _flush_buffers(self) -> List[Dict[str, Any]]:
        results = []
        if self.content_buffer or self.current_tag:
            config = self.tag_config.get(self.current_tag, TagConfig())
            content = self.content_buffer.strip()
            
            if content or self.current_tag:
                if self._is_buffering_config(config):
                    buffer_tag = config.buffer_tag or self.current_tag or "unknown"
                    results.append(self._create_result(buffer_tag, content))
        
        self.content_buffer = ""
        self.current_tag = ""
        return results

    def _create_result(self, tag: str, content: str) -> Dict[str, Any]:
        return {"tag": tag, "content": content}

    def _is_streaming_config(self, config: TagConfig) -> bool:
        return bool(config.mode & TagProcessingMode.STREAM)

    def _is_buffering_config(self, config: TagConfig) -> bool:
        return bool(config.mode & TagProcessingMode.BUFFER)
    def reset(self):
        self.state = ProcessingState.LOOKING_FOR_TAG
        self.current_tag = ""
        self.content_buffer = ""
        self.tag_buffer = ""
        self.debug_content = ""
