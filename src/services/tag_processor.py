from typing import Dict, Any, List, AsyncGenerator
import logging
import re
import json

logger = logging.getLogger(__name__)

class TagProcessor:
    def __init__(self, stream_tags: List[str], buffer_tags: List[str]):
        self.stream_tags = stream_tags
        self.buffer_tags = buffer_tags
        self.current_tag = None
        self.content_buffer = ""
        self.tag_buffer = ""
        self.debug_content = ""
        self.tag_pattern = re.compile(r'\[/?([^\]]+)\]')

    async def process_stream(self, token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            async for token in token_stream:
                self.debug_content += token
                
                for result in self._process_token(token):
                    yield result

            for result in self._flush_buffer():
                yield result

            yield {"tag": "debug", "content": self.debug_content}
        finally:
            self._reset()

    def _process_token(self, token: str) -> List[Dict[str, Any]]:
        results = []
        self.tag_buffer += token

        while self.tag_buffer:
            if '[' not in self.tag_buffer:
                results.extend(self._process_content(self.tag_buffer))
                self.tag_buffer = ""
                continue

            parts = self.tag_buffer.split('[', 1)
            if parts[0]:
                results.extend(self._process_content(parts[0]))
            self.tag_buffer = '[' + parts[1]

            match = self.tag_pattern.match(self.tag_buffer)
            if not match:
                break

            tag = match.group(1).lower()
            is_closing_tag = self.tag_buffer.startswith('[/')
            
            results.extend(self._process_tag(tag, is_closing_tag))
            self.tag_buffer = self.tag_buffer[match.end():]

        return results

    def _process_tag(self, tag: str, is_closing_tag: bool) -> List[Dict[str, Any]]:
        if tag not in self.stream_tags and tag not in self.buffer_tags:
            return self._process_content(self.tag_pattern.match(self.tag_buffer).group(0))

        results = []
        if is_closing_tag and tag == self.current_tag:
            results.extend(self._flush_buffer())
            self.current_tag = None
        elif not is_closing_tag:
            if self.current_tag:
                results.extend(self._flush_buffer())
            self.current_tag = tag
        return results

    def _process_content(self, content: str | Dict[str, Any]) -> List[Dict[str, Any]]:
        if not content:
            return []
        
        if isinstance(content, str):
            self.content_buffer += content
            if self.current_tag == 'content':
                return [{"tag": "content", "content": content}]
            return []
        
        if isinstance(content, dict):
            self.content_buffer = content
            return []

    def _flush_buffer(self) -> List[Dict[str, Any]]:
        results = []

        if not self.content_buffer:
            return []

        if self.current_tag == 'content':
            results.append({"tag": "content_full", "content": self.content_buffer})
        elif self.current_tag not in self.buffer_tags:
            results.append({"tag": self.current_tag, "content": self.content_buffer})
        elif self.current_tag == "tool":
            if isinstance(self.content_buffer, dict):
                results.append({"tag": "tool", "content": self.content_buffer})
            else:
                try:
                    tool_data = json.loads(self.content_buffer)
                    results.append({"tag": "tool", "content": tool_data})
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool JSON: {self.content_buffer}")
                    results.append({"tag": self.current_tag, "content": self.content_buffer})
        else:
            results.append({"tag": self.current_tag, "content": self.content_buffer})

        self.content_buffer = ""

        return results

    def _reset(self):
        self.current_tag = None
        self.content_buffer = ""
        self.tag_buffer = ""
        self.debug_content = ""