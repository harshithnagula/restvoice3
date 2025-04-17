import os
import json
import base64
import asyncio
import websockets
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
SYSTEM_MESSAGE = (
    "You are a friendly and bubbly AI assistant for Rollover Pizzas, a restaurant specializing in delicious pizzas, burgers, salads, and daily specials. "
    "Your primary role is to answer queries about the menu, including item descriptions, prices, and recommendations, and assist with customer questions. "
    "You have a penchant for dad jokes, owl jokes, and subtly rickrolling customers while keeping interactions positive and engaging. "
    "Here is the menu:\n\n"
    "**Pizzas**:\n"
    "1. Margherita Pizza - $12.99: Classic tomato sauce, fresh mozzarella, basil, and a drizzle of olive oil.\n"
    "2. Pepperoni Feast - $14.99: Tomato sauce, mozzarella, and loaded with pepperoni slices.\n"
    "3. Veggie Delight - $13.99: Tomato sauce, mozzarella, bell peppers, mushrooms, onions, and olives.\n\n"
    "**Burgers**:\n"
    "1. Classic Cheeseburger - $10.99: Beef patty, cheddar cheese, lettuce, tomato, pickles, and house sauce.\n"
    "2. BBQ Bacon Burger - $12.99: Beef patty, bacon, cheddar, onion rings, and BBQ sauce.\n"
    "3. Mushroom Swiss Burger - $11.99: Beef patty, Swiss cheese, sautéed mushrooms, and garlic aioli.\n\n"
    "**Salads**:\n"
    "1. Caesar Salad - $8.99: Romaine lettuce, croutons, parmesan, and Caesar dressing.\n"
    "2. Greek Salad - $9.99: Mixed greens, feta, olives, cucumbers, tomatoes, and vinaigrette.\n"
    "3. Spinach & Berry Salad - $9.99: Spinach, strawberries, blueberries, walnuts, and balsamic dressing.\n\n"
    "**Daily Specials**:\n"
    "1. Hawaiian Pizza - $15.99: Tomato sauce, mozzarella, ham, and pineapple (available Monday-Wednesday).\n"
    "2. Spicy Chicken Burger - $12.99: Grilled chicken, spicy mayo, jalapeños, and pepper jack cheese (available Thursday-Sunday).\n\n"
    "Answer all menu-related questions accurately, suggest items based on customer preferences, and maintain a warm, playful tone. "
    "For non-menu queries, provide helpful responses and work in a joke when appropriate."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server for Rollover Pizzas is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say("Welcome to Rollover Pizzas!.")
    response.pause(length=1)
    response.say("Alright, you're ready to explore our tasty menu! Start talking!")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        transcript_parts = []
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    # Capture assistant's speech transcript from response.done events
                    if response.get('type') == 'response.done':
                        try:
                            assistant_text = response.get('response', {}).get('output', [{}])[0].get('content', [{}])[0].get('transcript', '')
                            if assistant_text:
                                transcript_parts.append(f"Assistant: {assistant_text}")
                        except (IndexError, KeyError) as e:
                            print(f"Error extracting transcript from response.done: {e}")

                    # Capture user's speech transcript from input_audio_transcription.completed events
                    if response.get('type') == 'input_audio_transcription.completed':
                        user_text = response.get('transcript', '')
                        if user_text:
                            transcript_parts.append(f"User: {user_text}")

                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        # Update last_assistant_item safely
                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")
                # Upload transcript if there's an error
                if stream_sid:
                    await upload_transcript_to_gcs(transcript_parts, stream_sid)

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Welcome to Rollover Pizzas! I'm your AI assistant, ready to help you explore our delicious menu of pizzas, burgers, salads, and specials. Ask me about our Margherita Pizza, BBQ Bacon Burger,  How can I assist you today?"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)