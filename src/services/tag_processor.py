import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

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

            result = self._flush_buffer()
            if result:
                yield result

            if self.current_tag:
                self.debug_content += f"[/{self.current_tag}]"

            yield {"tag": "debug", "content": self.debug_content}
        finally:
            self._reset()

    def _process_token(self, token: str) -> List[Dict[str, Any]]:
        results = []
        self.tag_buffer += token

        while self.tag_buffer:
            if '[' not in self.tag_buffer:
                content_result = self._process_content(self.tag_buffer)
                if content_result:
                    results.append(content_result)
                self.tag_buffer = ""
                continue

            parts = self.tag_buffer.split('[', 1)
            if parts[0]:
                content_result = self._process_content(parts[0])
                if content_result:
                    results.append(content_result)
            self.tag_buffer = '[' + parts[1]

            match = self.tag_pattern.match(self.tag_buffer)
            if not match:
                break

            tag = match.group(1).lower()
            is_closing_tag = self.tag_buffer.startswith('[/')
            
            result = self._process_tag(tag, is_closing_tag)
            if result:
                results.append(result)
            self.tag_buffer = self.tag_buffer[match.end():]

        return results

    def _process_tag(self, tag: str, is_closing_tag: bool) -> Optional[Dict[str, Any]]:
        if tag not in self.stream_tags and tag not in self.buffer_tags:
            return self._process_content(self.tag_pattern.match(self.tag_buffer).group(0))

        if is_closing_tag and tag == self.current_tag:
            result = self._flush_buffer()
            self.current_tag = None
            return result
        elif not is_closing_tag:
            if self.current_tag:
                result = self._flush_buffer()
                self.current_tag = tag
                return result
            self.current_tag = tag
    
        return None

    def _process_content(self, content: str) -> Optional[Dict[str, Any]]:
        if not content:
            return None
        
        self.content_buffer += content
        if self.current_tag in self.stream_tags:
            return {"tag": self.current_tag, "content": content}
        return None

    def _flush_buffer(self) -> Optional[Dict[str, Any]]:
        if not self.content_buffer:
            return None

        if self.current_tag == 'text':
            result = {"tag": "full_text", "content": self.content_buffer}
        else:
            result = {"tag": self.current_tag, "content": self.content_buffer}

        self.content_buffer = ""
        return result

    def _reset(self):
        self.current_tag = None
        self.content_buffer = ""
        self.tag_buffer = ""
        self.debug_content = ""