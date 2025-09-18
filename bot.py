import os
import sys
import json
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
gemini_key = os.environ.get("GEMINI_API_KEY")

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
# Bot setup
# -----------------------------
intents = discord.Intents.none()
intents.guilds = True
intents.guild_messages = True
intents.message_content = True
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
# Load FAQ Documents
# -----------------------------
faq_docs = {}
try:
    with open('learned_docs.json', 'r', encoding='utf-8') as f:
        faq_docs = json.load(f)
    logger.info(f"✅ Loaded {len(faq_docs)} FAQ documents")
except FileNotFoundError:
    logger.warning("⚠️ FAQ documents not found")
except Exception as e:
    logger.error(f"❌ Failed to load FAQ documents: {e}")

# -----------------------------
# Hehe feature
# -----------------------------

# -----------------------------
# KB fuzzy matching
# -----------------------------
def fuzzy_match(word, keywords, cutoff=0.7):
    for kw in keywords:
        ratio = difflib.SequenceMatcher(None, word, kw).ratio()
        if ratio >= cutoff:
            return True
    return False

def search_faq_docs(query: str):
    """Search FAQ documents for relevant content"""
    if not faq_docs:
        return None
    
    query_lower = query.lower()
    
    # Domain-specific keywords that must be present
    domain_keywords = [
        'roblox', 'camera', 'zoom', 'shake', 'macro', 'bloxlink', 'helper', 
        'deadzone', 'tolerance', 'autohotkey', 'fisch', 'graphics', 'navigation',
        'failsafe', 'restart', 'delay', 'cast', 'rod', 'bait', 'microsoft',
        'web', 'transparent', 'head', 'mouse', 'stationary', 'limping', 'staff',
        'banned', 'version', 'scroll', 'sensitivity', 'tooltips'
    ]
    
    # Check if query contains at least one domain keyword
    has_domain_keyword = any(keyword in query_lower for keyword in domain_keywords)
    if not has_domain_keyword:
        return None
    
    # Stopwords to filter out
    stopwords = {'the', 'and', 'with', 'are', 'for', 'that', 'this', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'old', 'see', 'him', 'two', 'how', 'its', 'who', 'oil', 'sit', 'set'}
    
    best_match = None
    best_score = 0
    
    for doc_key, doc_data in faq_docs.items():
        content = doc_data['content'].lower()
        # Score based on meaningful words, filtering stopwords
        words = [w for w in query_lower.split() if len(w) > 2 and w not in stopwords]
        score = sum(1 for word in words if word in content)
        
        if score >= 2 and score > best_score:  # Require at least 2 meaningful matches
            best_score = score
            # Extract relevant section (300 chars max)
            for word in words:
                if word in content:
                    pos = content.find(word)
                    start = max(0, pos - 100)
                    end = min(len(content), pos + 200)
                    excerpt = doc_data['content'][start:end].strip()
                    if len(excerpt) > 300:
                        excerpt = excerpt[:300] + "..."
                    best_match = {
                        'title': doc_data['title'],
                        'excerpt': excerpt,
                        'score': score
                    }
                    break
    
    return best_match


# -----------------------------
# AI helper
# -----------------------------
async def check_doc_relevance(message):
    """Check if message is related to documentation topics"""
    if not client:
        return False, None
    
    try:
        # Build context from docs
        docs_context = ""
        if faq_docs:
            for doc_key, doc_data in faq_docs.items():
                docs_context += f"Document: {doc_data['title']}\n"
                docs_context += f"Key topics: {doc_data['content'][:800]}...\n\n"
        
        prompt = f"""Analyze if this message is related to these documentation topics:

{docs_context}

Message: "{message}"

Respond with:
- "YES" if related to roblox, fisch, macro, autohotkey, helper guidelines, camera settings, or similar topics from the docs
- "NO" if unrelated (greetings, random chat, off-topic)

If YES, provide a helpful answer based ONLY on the documentation content.

Format: YES/NO|Answer (if YES)"""
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        if response.text:
            parts = response.text.strip().split('|', 1)
            is_relevant = parts[0].strip().upper() == 'YES'
            answer = parts[1].strip() if len(parts) > 1 and is_relevant else None
            return is_relevant, answer
        return False, None
    except Exception as e:
        logger.error(f"Relevance check error: {e}")
        return False, None

def get_specific_responses(message):
    """Handle specific common questions"""
    text = message.lower()
    
    # Fisch macro location
    if any(term in text for term in ['fisch', 'fish']) and any(term in text for term in ['macro', 'where', 'location']):
        return "https://discord.com/channels/1341949236471926804/1413837110770925578 - You can find the fisch macro in this channel."
    
    # AHK version question
    if any(term in text for term in ['ahk', 'autohotkey']) and any(term in text for term in ['version', 'what']):
        return "Use AutoHotkey v1.1 for the fisch macro."
    
    # Macro not working
    if any(term in text for term in ['not working', 'broken', 'doesnt work', "doesn't work", 'help']) and 'macro' in text:
        return """To fix macro issues:
1. Download AutoHotkey v1.1 from https://autohotkey.com
2. Use Web Roblox (not Microsoft Store version)
3. Check your Navigation Key settings
4. Ensure Auto Lower Graphics is enabled
5. Make sure you have the correct delays configured
6. Verify your camera zoom and shake settings are properly set up"""
    
    return None

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

    # Use AI to check if message is related to docs FIRST
    is_relevant, ai_answer = await check_doc_relevance(message.content)
    if not is_relevant:
        # If not doc-related, stay completely silent
        await bot.process_commands(message)
        return

    # If relevant, check specific hardcoded responses first
    specific_response = get_specific_responses(message.content)
    if specific_response:
        await message.reply(specific_response)
        learning_data.add_conversation(message.author.id, message.content, specific_response)
        return

    # Then check FAQ documents
    faq_result = search_faq_docs(message.content)
    if faq_result:
        response = f"**{faq_result['title']}:**\\n{faq_result['excerpt']}"
        await message.reply(response)
        learning_data.add_conversation(message.author.id, message.content, response)
        return

    # Finally use AI answer if available
    if ai_answer:
        await message.reply(ai_answer)
        learning_data.add_conversation(message.author.id, message.content, ai_answer)

    await bot.process_commands(message)

# -----------------------------
# Slash Commands
# -----------------------------

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
            resp = requests.get(url, timeout=10)
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

@bot.tree.command(name="ask", description="Ask a question about macros or helpers")
@app_commands.describe(question="Your question about macros, settings, or helper guidelines")
async def ask_command(interaction: discord.Interaction, question: str):
    # Check specific responses first
    specific_response = get_specific_responses(question)
    if specific_response:
        await interaction.response.send_message(specific_response)
        learning_data.add_conversation(interaction.user.id, question, specific_response)
        return
    
    # Check FAQ documents
    faq_result = search_faq_docs(question)
    if faq_result:
        response = f"**{faq_result['title']}:**\n{faq_result['excerpt']}"
        await interaction.response.send_message(response)
        learning_data.add_conversation(interaction.user.id, question, response)
        return
    
    # Use AI to check relevance and generate answer
    is_relevant, ai_answer = await check_doc_relevance(question)
    if is_relevant and ai_answer:
        await interaction.response.send_message(ai_answer)
        learning_data.add_conversation(interaction.user.id, question, ai_answer)
    else:
        await interaction.response.send_message("Question not related to available documentation.", ephemeral=True)


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
