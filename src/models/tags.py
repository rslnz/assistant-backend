from enum import Enum


class ResponseType(str, Enum):
    REASONING = "reasoning"
    TEXT = "text"
    FULL_TEXT = "full_text"
    DEBUG = "debug"
    ERROR = "error"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    UPDATED_CONTEXT = "updated_context"