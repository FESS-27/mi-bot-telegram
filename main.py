import os
import logging
import sqlite3
import json
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq
from duckduckgo_search import DDGS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

client = Groq(api_key=GROQ_API_KEY)

# --- SQLite: Memoria persistente ---
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (chat_id INTEGER, role TEXT, content TEXT, tool_calls TEXT, tool_call_id TEXT)''')
    conn.commit()
    conn.close()

def get_history(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content, tool_calls, tool_call_id FROM history WHERE chat_id=?", (chat_id,))
    rows = c.fetchall()
    conn.close()
    history = []
    for role, content, tool_calls, tool_call_id in rows:
        msg = {"role": role}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = json.loads(tool_calls)
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        history.append(msg)
    return history

def save_message(chat_id, role, content=None, tool_calls=None, tool_call_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if tool_calls:
        tool_calls_dict = [tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in tool_calls]
        tool_calls_json = json.dumps(tool_calls_dict)
    else:
        tool_calls_json = None
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?)", 
              (chat_id, role, content, tool_calls_json, tool_call_id))
    conn.commit()
    conn.close()

def clear_history(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()

init_db()

# --- Herramientas ---
def get_weather(location: str) -> str:
    """Consulta el clima actual de una ubicación."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_resp = requests.get(geo_url).json()
        if "results" not in geo_resp:
            return "No encontré esa ubicación."
        
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url).json()
        temp = weather_resp["current_weather"]["temperature"]
        
        return f"En {location} la temperatura actual es de {temp}°C."
    except Exception as e:
        return f"Error al consultar el clima: {str(e)}"

def search_web(query: str) -> str:
    """Busca información en la web usando DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No encontré resultados para esa búsqueda."
            
            summary = []
            for r in results:
                summary.append(f"- {r['title']}: {r['body']}")
            return "\n".join(summary)
    except Exception as e:
        return f"Error al buscar en la web: {str(e)}"

def calculate(expression: str) -> str:
    """Evalúa una expresión matemática."""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Expresión no válida. Solo uso números y operadores matemáticos."
        
        result = eval(expression)
        return f"El resultado de {expression} es {result}"
    except Exception as e:
        return f"Error al calcular: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Consulta el clima actual de una ubicación (ciudad, país).",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Nombre de la ciudad o ubicación (ej: 'Madrid', 'Buenos Aires')"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Busca información actualizada en la web sobre cualquier tema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La consulta de búsqueda (ej: 'últimas noticias tecnología', 'precio bitcoin hoy')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Realiza cálculos matemáticos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "La expresión matemática (ej: '2+2', '15*3', '100/7')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

available_functions = {
    "get_weather": get_weather,
    "search_web": search_web,
    "calculate": calculate
}

# --- Reconocimiento de voz ---
async def transcribe_audio(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Transcribe audio usando Whisper de Groq."""
    try:
        file = await context.bot.get_file(file_id)
        audio_path = f"/tmp/{file_id}.ogg"
        await file.download_to_drive(audio_path)
        
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text"
            )
        
        os.remove(audio_path)
        return transcription
    except Exception as e:
        logging.error(f"Error transcribiendo audio: {e}")
        return None

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes de voz."""
    chat_id = update.effective_chat.id
    file_id = update.message.voice.file_id
    
    await update.message.reply_text("🎤 Transcribiendo tu mensaje de voz...")
    
    transcription = await transcribe_audio(file_id, context)
    if not transcription:
        await update.message.reply_text("No pude transcribir tu mensaje. Intenta de nuevo.")
        return
    
    await update.message.reply_text(f"📝 Transcripción: {transcription}")
    
    # Procesar la transcripción como un mensaje de texto
    save_message(chat_id, "user", transcription)
    history = get_history(chat_id)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            save_message(chat_id, "assistant", None, response_message.tool_calls)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function_response = available_functions[function_name](**function_args)
                save_message(chat_id, "tool", function_response, tool_call_id=tool_call.id)
            
            history = get_history(chat_id)
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=history
            )
            reply = final_response.choices[0].message.content
        else:
            reply = response_message.content
        
        save_message(chat_id, "assistant", reply)
        await update.message.reply_text(reply)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("Hubo un error al procesar tu mensaje.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes de texto."""
    chat_id = update.effective_chat.id
    user_text = update.message.text

    try:
        save_message(chat_id, "user", user_text)
        history = get_history(chat_id)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            save_message(chat_id, "assistant", None, response_message.tool_calls)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function_response = available_functions[function_name](**function_args)
                save_message(chat_id, "tool", function_response, tool_call_id=tool_call.id)
            
            history = get_history(chat_id)
            final_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=history
            )
            reply = final_response.choices[0].message.content
        else:
            reply = response_message.content

        save_message(chat_id, "assistant", reply)
        await update.message.reply_text(reply)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("Hubo un error al procesar tu mensaje.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    save_message(chat_id, "system", "Eres un asistente personal útil. Puedes consultar el clima, buscar en la web y hacer cálculos. Responde en español.")
    await update.message.reply_text(
        "¡Hola! Puedo:\n"
        "🌤️ Consultar el clima\n"
        "🔍 Buscar en la web\n"
        "🧮 Hacer cálculos\n"
        "🎤 Transcribir mensajes de voz\n\n"
        "Usa /reset para borrar la memoria."
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    save_message(chat_id, "system", "Eres un asistente personal útil. Puedes consultar el clima, buscar en la web y hacer cálculos. Responde en español.")
    await update.message.reply_text("Memoria borrada.")

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}"
    )

if __name__ == '__main__':
    main()
