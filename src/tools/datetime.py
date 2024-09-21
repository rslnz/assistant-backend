from datetime import datetime
from typing import ClassVar

from langchain_core.tools import BaseTool

class DateTimeTool(BaseTool):
    name: ClassVar[str] = "get_current_datetime"
    description: ClassVar[str] = "Returns the current date and time."

    def _run(self) -> str:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current date and time: {current_datetime}"

    async def _arun(self) -> str:
        return self._run()