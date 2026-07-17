import os
import logging
import sqlite3
import json
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq

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
                 (chat_id INTEGER, role TEXT, content TEXT, tool_calls TEXT)''')
    conn.commit()
    conn.close()

def get_history(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content, tool_calls FROM history WHERE chat_id=?", (chat_id,))
    rows = c.fetchall()
    conn.close()
    history = []
    for role, content, tool_calls in rows:
        msg = {"role": role}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = json.loads(tool_calls)
        history.append(msg)
    return history

def save_message(chat_id, role, content=None, tool_calls=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?)", 
              (chat_id, role, content, json.dumps(tool_calls) if tool_calls else None))
    conn.commit()
    conn.close()

def clear_history(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE chat_id=?", (chat_id,))
    conn.commit()
    conn.close()
# ------------------------------------

# --- Herramientas (Functions) ---
def get_weather(location: str) -> str:
    """Consulta el clima actual de una ubicación."""
    try:
        # Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_resp = requests.get(geo_url).json()
        if "results" not in geo_resp:
            return "No encontré esa ubicación."
        
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        
        # Clima actual
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url).json()
        temp = weather_resp["current_weather"]["temperature"]
        
        return f"En {location} la temperatura actual es de {temp}°C."
    except Exception as e:
        return f"Error al consultar el clima: {str(e)}"

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
    }
]

available_functions = {
    "get_weather": get_weather
}
# ------------------------------------

init_db()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    save_message(chat_id, "system", "Eres un asistente personal útil. Puedes consultar el clima. Responde en español.")
    await update.message.reply_text("¡Hola! Puedo consultar el clima. Prueba: '¿Cómo está el clima en Madrid?' Usa /reset para borrar la memoria.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    save_message(chat_id, "system", "Eres un asistente personal útil. Puedes consultar el clima. Responde en español.")
    await update.message.reply_text("Memoria borrada.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    try:
        save_message(chat_id, "user", user_text)
        history = get_history(chat_id)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
                
                save_message(chat_id, "tool", function_response)
            
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

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    app.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}"
    )

if __name__ == '__main__':
    main()
