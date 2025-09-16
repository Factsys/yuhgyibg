import os
import sys
from unittest.mock import MagicMock

# Mock audioop module before importing discord to prevent ModuleNotFoundError
sys.modules['audioop'] = MagicMock()

import discord
from discord.ext import commands
from discord import app_commands
import asyncio
import logging
from datetime import datetime
import difflib
import random
import requests
import trafilatura
from bs4 import BeautifulSoup
from google import genai

# -----------------------------
# Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

discord_token = os.environ.get("TOKEN")
gemini_key = os.environ.get("AIzaSyAO4jV2tNyVJ1-Hagvuqg9lpUd6kF5KkRs") or "YOUR_GEMINI_KEY_HERE"

if not discord_token:
    logger.warning("⚠️ TOKEN not set!")

if not gemini_key:
    logger.warning("⚠️ GEMINI_API_KEY not set!")

client = None
if gemini_key:
    try:
        client = genai.Client(api_key=gemini_key)
        logger.info("✅ Gemini client initialized")
    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")

# -----------------------------
# Knowledge Base
# -----------------------------
KNOWLEDGE_BASE = {
    "questions": [
        {
            "keywords": ["where is the macro", "macro location", "find macro"],
            "answer": "macros from us will be in https://discord.com/channels/1341949236471926804/1413837110770925578 (click it)"
        },
        {
            "keywords": ["how do i verify", "verification", "verify account"],
            "answer": "go to <#1411335498945003654> and use /verify"
        },
        {
            "keywords": ["macro malware", "macro virus", "is macro safe", "malware", "virus"],
            "answer": "no the macro is open source and is not malware"
        }
    ],
    "triggers": {
        "ahk": {
            "keywords": ["ahk", "autohotkey", "how to open macro"],
            "response": "You need this to open the macro: https://autohotkey.com (download v.1.1)"
        },
        "macro": {
            "keywords": ["where is fish", "fisch macro", "just macro", "fish macro"],
            "response": "if you want the fisch macro go to <#1413837110770925578>"
        }
    }
}

# -----------------------------
# Bot setup
# -----------------------------
intents = discord.Intents.none()
intents.guilds = True
intents.guild_messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# -----------------------------
# Learning Data
# -----------------------------
class LearningData:
    def __init__(self):
        self.conversations = []
        self.learned_content = []
    
    def add_conversation(self, user_id, question, answer):
        self.conversations.append({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_learned_content(self, url, title, content, user_id):
        self.learned_content.append({
            "url": url,
            "title": title,
            "content": content,
            "learned_by": user_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_learned_content_summary(self):
        if not self.learned_content:
            return ""
        summary = "LEARNED DOCUMENTS:\n\n"
        for doc in self.learned_content[-10:]:
            summary += f"Title: {doc['title']}\n"
            summary += f"Content: {doc['content'][:500]}...\n"
            summary += f"Source: {doc['url']}\n\n"
        return summary

learning_data = LearningData()

# -----------------------------
# Hehe feature
# -----------------------------
hehe_config = {
    "chance": 0.5,   # default percent
    "context": "hi"  # default context
}

# -----------------------------
# KB fuzzy matching
# -----------------------------
def fuzzy_match(word, keywords, cutoff=0.7):
    for kw in keywords:
        ratio = difflib.SequenceMatcher(None, word, kw).ratio()
        if ratio >= cutoff:
            return True
    return False

def match_knowledge(message: str):
    text = message.lower()
    question_words = ["where", "how", "what", "why", "when", "who", "get", "find", "location", "verify", "setup", "?"]
    is_question = any(q in text for q in question_words)
    if not is_question:
        return None

    for qa in KNOWLEDGE_BASE["questions"]:
        for kw in qa["keywords"]:
            if kw in text or fuzzy_match(text, [kw]):
                return qa["answer"]

    for trigger_name, trigger in KNOWLEDGE_BASE["triggers"].items():
        for kw in trigger["keywords"]:
            if kw in text or fuzzy_match(text, [kw]):
                return trigger["response"]

    return None

# -----------------------------
# AI helper
# -----------------------------
async def generate_smart_response(question, intent_context=None):
    if not client:
        return "AI is disabled."
    try:
        full_knowledge = "COMMUNITY KNOWLEDGE:\n\n"
        for qa in KNOWLEDGE_BASE["questions"]:
            topics = ", ".join(qa["keywords"])
            full_knowledge += f"Topic: {topics}\nInfo: {qa['answer']}\n\n"
        for trigger_name, trigger in KNOWLEDGE_BASE["triggers"].items():
            topics = ", ".join(trigger["keywords"])
            full_knowledge += f"Topic: {topics}\nInfo: {trigger['response']}\n\n"
        learned_summary = learning_data.get_learned_content_summary()
        if learned_summary:
            full_knowledge += f"\n{learned_summary}"
        prompt = f"""You are Bloom, a helpful Discord bot.

{full_knowledge}

User asked: "{question}"
Interpreted as: {intent_context or 'general'}

Give a natural and helpful response:"""
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text if response.text else "I'm having trouble answering right now."
    except Exception as e:
        logger.error(f"AI error: {e}")
        return "I'm experiencing some technical difficulties."

# -----------------------------
# Events
# -----------------------------
@bot.event
async def on_ready():
    if bot.user:
        logger.info(f"✅ {bot.user.name} connected!")
    try:
        synced = await bot.tree.sync()
        logger.info(f"✅ Synced {len(synced)} commands")
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # --- Hehe random reply ---
    roll = random.uniform(0, 100)
    if roll < hehe_config["chance"]:
        reply_prompt = f"""You are Bloom, a playful Discord bot. 
The user said: "{message.content}".
Random reply context: "{hehe_config['context']}".

Respond casually and naturally."""
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=reply_prompt)
            if response.text:
                await message.reply(response.text.strip())
                return  # exclusive
        except Exception as e:
            logger.error(f"Hehe AI error: {e}")
    # --------------------------

    # KB detection
    kb_answer = match_knowledge(message.content)
    if kb_answer:
        await message.reply(kb_answer)
        learning_data.add_conversation(message.author.id, message.content, kb_answer)
        return

    # AI fallback
    smart_response = await generate_smart_response(message.content, "general")
    await message.reply(smart_response)
    learning_data.add_conversation(message.author.id, message.content, smart_response)

    await bot.process_commands(message)

# -----------------------------
# Slash Commands
# -----------------------------
@bot.tree.command(name="say", description="Make the bot say something")
async def say_command(interaction: discord.Interaction, words: str):
    await interaction.response.send_message(words)

@bot.tree.command(name="learn", description="Teach Bloom something new")
@app_commands.describe(
    url="Optional source link (e.g., Google Doc or website)",
    title="Title of the knowledge",
    content="Optional extra content (if no URL, you must provide content)"
)
async def learn_command(interaction: discord.Interaction, url: str = "", title: str = "Untitled", content: str = ""):
    text_content = content

    # If URL provided, try fetching text
    if url:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                downloaded = trafilatura.extract(resp.text)
                if downloaded:
                    text_content += "\n" + downloaded
                else:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    text_content += "\n" + soup.get_text(separator="\n")
        except Exception as e:
            await interaction.response.send_message(f"⚠️ Failed to fetch from URL: {e}", ephemeral=True)
            return

    if not text_content.strip():
        await interaction.response.send_message("❌ You must provide content or a valid URL.", ephemeral=True)
        return

    learning_data.add_learned_content(url, title, text_content.strip(), interaction.user.id)
    await interaction.response.send_message(f"✅ Learned new content titled **{title}**", ephemeral=True)

@bot.tree.command(name="hehe", description="Set a random reply chance and context")
@app_commands.describe(
    percent="Chance (0.1 - 100) that Bloom will reply randomly",
    context="The playful context for random replies"
)
async def hehe_command(interaction: discord.Interaction, percent: float, context: str):
    if percent < 0.1 or percent > 100:
        await interaction.response.send_message("❌ Please provide a percentage between 0.1 and 100.", ephemeral=True)
        return
    hehe_config["chance"] = percent
    hehe_config["context"] = context
    await interaction.response.send_message(f"✅ Hehe chance set to {percent}% with context: '{context}'", ephemeral=True)

# -----------------------------
# Run bot
# -----------------------------
if __name__ == "__main__":
    if not discord_token:
        logger.error("TOKEN not found. Cannot start bot.")
        exit(1)
    try:
        bot.run(discord_token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
