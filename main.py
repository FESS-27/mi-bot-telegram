import os
import logging
import sqlite3
import json
import requests
import urllib.parse
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

client = Groq(api_key=GROQ_API_KEY)

# Solo modelos que han demostrado funcionar (200 OK) en tus logs
GROQ_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]

# --- SQLite: Memoria persistente ---
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (chat_id INTEGER, role TEXT, content TEXT, tool_calls TEXT, tool_call_id TEXT)''')
    conn.commit()
    conn.close()

def get_history(chat_id, max_messages=6):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Obtener el mensaje de sistema
    c.execute("SELECT role, content, tool_calls, tool_call_id FROM history WHERE chat_id=? AND role='system' LIMIT 1", (chat_id,))
    system_row = c.fetchone()
    
    # 2. Obtener solo los últimos 6 mensajes (reducido de 10 para ahorrar más tokens)
    c.execute("SELECT role, content, tool_calls, tool_call_id FROM history WHERE chat_id=? AND role!='system' ORDER BY rowid DESC LIMIT ?", (chat_id, max_messages))
    rows = c.fetchall()
    conn.close()
    
    history = []
    if system_row:
        role, content, tool_calls, tool_call_id = system_row
        msg = {"role": role}
        if content: msg["content"] = content[:1000] # Truncar sistema si es muy largo
        history.append(msg)
        
    for role, content, tool_calls, tool_call_id in reversed(rows):
        msg = {"role": role}
        if content: 
            # TRUNCAR CONTENIDO A 400 CARACTERES PARA EVITAR ERROR 429 DE GROQ
            msg["content"] = (content[:400] + "...[truncado]") if len(content) > 400 else content
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
def search_web(query: str) -> str:
    """Busca información web y recupera contenido útil de los resultados."""
    try:
        logging.info(f"DEBUG BÚSQUEDA - Query: {query}")

        jina_api_key = os.environ.get("JINA_API_KEY")
        if not jina_api_key:
            return "Error: Falta la variable de entorno JINA_API_KEY."

        url = f"https://s.jina.ai/{urllib.parse.quote(query)}"
        headers = {
            "Authorization": f"Bearer {jina_api_key}",
            "Accept": "application/json",
            "X-Retain-Images": "none"
        }

        response = requests.get(
            url,
            headers=headers,
            params={"count": 5},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("data", [])
        if not results:
            return "No encontré resultados para esa búsqueda."

        summaries = []

        for result in results[:5]:
            title = result.get("title", "Sin título")
            source = result.get("url", "")
            content = result.get("content", "")
            description = result.get("description", "")

            text = content or description or "Sin contenido disponible"

            summaries.append(
                f"Título: {title}\n"
                f"Fuente: {source}\n"
                f"Contenido:\n{text[:2500]}"
            )

        summary = "\n\n---\n\n".join(summaries)

        logging.info(
            f"DEBUG BÚSQUEDA - Resultados recuperados: {len(summaries)}"
        )

        return summary

    except requests.exceptions.Timeout:
        logging.error("DEBUG BÚSQUEDA - Timeout")
        return "La búsqueda tardó demasiado. Intenta de nuevo."

    except Exception as e:
        logging.error(f"DEBUG BÚSQUEDA - Error: {str(e)}")
        return f"Error técnico en la búsqueda: {str(e)}"

def calculate(expression: str) -> str:
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Expresión no válida."
        result = eval(expression)
        return f"El resultado de {expression} es {result}"
    except Exception as e:
        return f"Error al calcular: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for real-time, up-to-date information. Use this when the user asks for current data like exchange rates, news, or weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query in English for better results (e.g., 'USD to CRC exchange rate today')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression (e.g., '2+2')"}
                },
                "required": ["expression"]
            }
        }
    }
]

available_functions = {
    "search_web": search_web,
    "calculate": calculate
}

# --- Voz ---
async def transcribe_audio(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    try:
        file = await context.bot.get_file(file_id)
        audio_path = f"/tmp/{file_id}.ogg"
        await file.download_to_drive(audio_path)
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file, model="whisper-large-v3", response_format="text"
            )
        os.remove(audio_path)
        return transcription
    except Exception as e:
        logging.error(f"Error transcribiendo: {e}")
        return None

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("🎤 Transcribiendo...")
    transcription = await transcribe_audio(update.message.voice.file_id, context)
    if not transcription:
        await update.message.reply_text("No pude transcribir tu mensaje.")
        return
    
    await update.message.reply_text(f"📝 Transcripción: {transcription}")
    await process_text(chat_id, transcription, update, context)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_text(update.effective_chat.id, update.message.text, update, context)

async def process_text(chat_id: int, user_text: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        save_message(chat_id, "user", user_text)
        history = get_history(chat_id)

        response = None
        last_error = None
        for model in GROQ_MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=history,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=500,
                )
                logging.info(f"DEBUG - Modelo usado (1ra llamada): {model}")
                break
            except Exception as e:
                last_error = e
                logging.warning(f"Modelo {model} falló: {e}")
                continue
        
        if response is None:
            raise last_error

        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            logging.info("DEBUG - Tool calls detectados correctamente.")
            save_message(chat_id, "assistant", None, response_message.tool_calls)
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                logging.info(f"DEBUG - Ejecutando función: {func_name} con args: {func_args}")
                func_response = available_functions[func_name](**func_args)
                save_message(chat_id, "tool", func_response, tool_call_id=tool_call.id)
            
            history = get_history(chat_id)
            
            final_response = None
            for model in GROQ_MODELS:
                try:
                    final_response = client.chat.completions.create(
                        model=model,
                        messages=history,
                        max_tokens=500,
                    )
                    logging.info(f"DEBUG - Modelo usado (respuesta final): {model}")
                    break
                except Exception as e:
                    last_error = e
                    logging.warning(f"Modelo {model} falló en respuesta final: {e}")
                    continue
            
            if final_response is None:
                raise last_error
                
            reply = final_response.choices[0].message.content
            logging.info(f"DEBUG - Respuesta final enviada al usuario: {reply[:200]}...")
        else:
            logging.warning("DEBUG - No hay tool_calls, el modelo devolvió texto plano.")
            reply = response_message.content

        save_message(chat_id, "assistant", reply)
        await update.message.reply_text(reply)
    except Exception as e:
        logging.error(f"Error crítico: {e}")
        await update.message.reply_text("Hubo un error al procesar tu mensaje.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    system_prompt = (
        "Eres un asistente personal útil. REGLAS ESTRICTAS: "
        "1. Responde ÚNICAMENTE en el mismo idioma en que el usuario te escribió. "
        "2. Sé breve, directo y al punto. Sin explicaciones extensas ni saludos innecesarios. "
        "3. No inventes ni asumas información. Si no sabes algo o te falta información, pregúntalo directamente. "
        "4. Usa las herramientas de búsqueda o cálculo cuando sea necesario para dar datos reales."
    )
    save_message(chat_id, "system", system_prompt)
    await update.message.reply_text("¡Hi FESS! Reglas cargadas. Puedo:\n🔍 Buscar en la web\n🧮 Hacer cálculos\n🎤 Transcribir voz\n\nUsa /reset para reiniciar.")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    system_prompt = (
        "Eres un asistente personal útil. REGLAS ESTRICTAS: "
        "1. Responde ÚNICAMENTE en el mismo idioma en que el usuario te escribió. "
        "2. Sé breve, directo y al punto. Sin explicaciones extensas ni saludos innecesarios. "
        "3. No inventes ni asumas información. Si no sabes algo o te falta información, pregúntalo directamente. "
        "4. Usa las herramientas de búsqueda o cálculo cuando sea necesario para dar datos reales."
    )
    save_message(chat_id, "system", system_prompt)
    await update.message.reply_text("Memoria y reglas reiniciadas.")

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
