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

### Using run.sh (Recommended)

The easiest way to run the application is by using the provided `run.sh` script. This script sets up the virtual environment, installs dependencies, and starts the application using uvicorn.

1. Make sure the script is executable:

```bash
chmod +x run.sh
```

2. Run the script:

```bash
./run.sh
```

This will set up everything and start the server using uvicorn.

### Using uvicorn directly

If you prefer to run the application directly with uvicorn:

1. Ensure you're in the virtual environment:

```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Set the necessary environment variables:

```bash
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

3. Run the application with uvicorn:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
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
      - "text": Accumulate or display the text content (see note below).
      - "tool_start": Indicate that a tool is being used.
      - "tool_end": Process and display the tool result.
      - "context_update": Store this updated context for the next request.
      - "error": Handle or display the error message.

4. After the stream ends, use the most recent "context_update" in your next request to maintain conversation continuity.

The `content` field for each event type contains the following:

- "text": A string containing a token of the generated text from the AI. Note that these are individual tokens and should be concatenated to form the complete response. Clients should accumulate these tokens and update the display in real-time as they are received.
- "tool_start": An object with the following structure:
  ```json
  {
    "name": "tool_name",
    "description": "Tool description",
    "action": "Description of the current action"
  }
  ```
- "tool_end": An object with the following structure:
  ```json
  {
    "name": "tool_name",
    "result": "Result of the tool execution"
  }
  ```
- "context_update": The full updated ConversationContext object, which should be sent with the next request.
- "error": A string describing the error that occurred.

Remember that the server is stateless, so it's crucial to send the updated context with each new request to preserve the conversation history.

This approach allows for real-time processing and display of the AI's response, including any tool usage, while maintaining the conversation state on the client side. The streaming of individual tokens for "text" events enables a more responsive and dynamic user interface, where the AI's response can be displayed as it's being generated.

## Stateless Backend

This backend is designed to be stateless. The conversation context is not stored on the server but is instead returned to the client as part of the response. Clients should send the updated context with each new request to maintain conversation continuity. This design allows for easy scaling and deployment of the backend service.
