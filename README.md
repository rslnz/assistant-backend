# assistant-backend

This project is a stateless backend service for a chat application using Large Language Models (LLM) with OpenAI integration.

## Features

- Streaming chat responses
- Integration with OpenAI's language models
- Web search and web parsing tools
- Stateless conversation context management (updated context is returned to the client)
- Docker support for easy deployment

## Prerequisites

- Python 3.12+
- Docker (optional)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/rslnz/assistant-backend.git
cd assistant-backend
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:

```bash
API_V1=/api/v1
HOST=0.0.0.0
PORT=8000

OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o

MAX_HISTORY_MESSAGES=10
```

## Running the Application

### Using run.sh (Linux/macOS)

The easiest way to run the application on Linux or macOS is by using the provided `run.sh` script. This script sets up the virtual environment, installs dependencies, and starts the application.

1. Make sure the script is executable:

```bash
chmod +x run.sh
```

2. Run the script:

```bash
./run.sh
```

This will set up everything and start the server.

### Using run.ps1 (Windows)

For Windows users, we provide a PowerShell script `run.ps1` that performs the same functions as `run.sh`.

1. Open PowerShell and navigate to the project directory.

2. You may need to change the execution policy to run the script. You can do this for the current PowerShell session with:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

3. Run the script:

```powershell
.\run.ps1
```

This will set up the virtual environment, install dependencies, and start the server.

### Running manually

If you prefer to run the application manually:

1. Ensure you're in the virtual environment:

For Linux/macOS:

```bash
source venv/bin/activate
```

For Windows:

```powershell
venv\Scripts\activate
```

2. Run the application:

```bash
python src/main.py
```

### Using Docker

1. Build the Docker image:

```bash
docker build -t assistant-backend .
```

2. Run the Docker container:

```bash
docker run -p 8000:8000 assistant-backend
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Handling Streaming Responses

The chat endpoint returns a streaming response. Here's a general algorithm for handling the stream:

1. Send a POST request to the chat endpoint (e.g., `http://localhost:8000/api/chat`) with the appropriate headers and JSON body containing the message, system prompt, and context (if available).

2. Set up to receive a stream of Server-Sent Events (SSE).

3. For each event in the stream:
   a. Check if the event data starts with "data: ".
   b. If it does, remove the "data: " prefix.
   c. If the remaining content is "[DONE]", this signals the end of the stream. Exit the loop.
   d. Otherwise, parse the content as JSON.
   e. Handle the parsed JSON based on its "type" field:
      - "reasoning": Show the AI's user notification.
      - "text": Accumulate or display the text content.
      - "tool_start": Indicate that a tool is being used.
      - "tool_end": Process and display the tool result.
      - "context_update": Store this updated context for the next request.
      - "error": Handle or display the error message.

4. After the stream ends, use the most recent "context_update" in your next request to maintain conversation continuity.

The `content` field for each event type contains the following:

- "reasoning": A string containing a short notification for the user (1-5 words).
- "text": A string containing a token of the generated text from the AI.
- "tool_start": An object with the structure:
  ```json
  {
    "id": "unique_tool_call_id",
    "name": "tool_name",
    "description": "Tool description",
    "user_notification": "Short notification for the user (1-5 words)"
  }
  ```
- "tool_end": An object with the structure:
  ```json
  {
    "id": "unique_tool_call_id",
    "name": "tool_name",
    "result": "Result of the tool execution"
  }
  ```
- "context_update": The full updated ConversationContext object, containing the entire conversation history. This updated context must be sent with the next request to maintain conversation continuity.
- "error": A string describing the error that occurred.

Remember that the server is stateless, so it's crucial to send the updated context (received in the "context_update" event) with each new request to preserve the conversation history and maintain continuity.

This approach allows for real-time processing and display of the AI's response, including its reasoning process and any tool usage, while maintaining the conversation state on the client side. The `id` field in `tool_start` and `tool_end` events represents a unique identifier for each specific tool call, allowing clients to match the start and end of individual tool executions and track their progress.

## Stateless Backend

This backend is designed to be stateless. The conversation context is not stored on the server but is instead returned to the client as part of the response. Clients should send the updated context with each new request to maintain conversation continuity. This design allows for easy scaling and deployment of the backend service.
