import os
import sys
import json
from unittest.mock import MagicMock
from datetime import datetime, timedelta
import re
import time

# Mock audioop module before importing discord to prevent ModuleNotFoundError
sys.modules['audioop'] = MagicMock()

import discord
from discord.ext import commands
from discord import app_commands
import asyncio
import logging
import random
import requests
try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import wikipedia
except ImportError:
    wikipedia = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

# -----------------------------
# Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Admin controls - Users who can use restricted commands
ADMIN_USER_IDS = [
    1334138321412296725  # Your ID - add more IDs here separated by commas
]

# Role ID that can use /tellmeajoke command
TELLMEAJOKE_ROLE_ID = 1234567890123456789  # Replace with actual role ID

discord_token = os.getenv('TOKEN')
gemini_key = os.getenv('API')
news_api_key = os.getenv('NEWS_API_KEY')  # Get from newsapi.org

# Initialize NewsAPI client if key exists
news_client = None
if news_api_key and NewsApiClient:
    try:
        news_client = NewsApiClient(api_key=news_api_key)
        logger.info("✅ NewsAPI client initialized")
    except Exception as e:
        logger.error(f"❌ NewsAPI init failed: {e}")
elif not NewsApiClient:
    logger.info("ℹ️ NewsAPI not available (package not installed)")

if not discord_token:
    logger.warning("⚠️ TOKEN not set!")

if not gemini_key:
    logger.warning("⚠️ GEMINI_API_KEY not set!")

client = None
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        client = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("✅ Gemini client initialized")
    except Exception as e:
        logger.error(f"❌ Gemini init failed: {e}")

# -----------------------------
# Bot setup
# -----------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# -----------------------------
# Rate limiting
# -----------------------------
last_ai_request = {}  # user_id: timestamp
AI_COOLDOWN = 3  # seconds between AI requests per user

# -----------------------------
# Persistent data storage
# -----------------------------
persistent_names = {}  # user_id: enforced_nickname

def load_persistent_data():
    global persistent_names
    try:
        with open('persistent_data.json', 'r') as f:
            data = json.load(f)
            persistent_names = data.get('names', {})
    except FileNotFoundError:
        pass

def save_persistent_data():
    with open('persistent_data.json', 'w') as f:
        json.dump({'names': persistent_names}, f)

load_persistent_data()

# -----------------------------
# Helper functions
# -----------------------------
def is_advanced_question(text: str) -> bool:
    """
    Very restrictive question detection - only responds to clear questions
    """
    if not text or len(text.strip()) < 3:
        return False

    text = text.strip().lower()

    # 1. Must have question mark
    if text.endswith('?'):
        return True

    # 2. Only very clear question starters
    clear_question_starters = ['how do', 'how can', 'what is', 'where is', 'why is', 'when is']
    
    if any(text.startswith(starter) for starter in clear_question_starters):
        return True

    # 3. Only respond to explicit help requests
    explicit_help = [
        'help me', 'please help', 'can you help', 'need help'
    ]

    if any(pattern in text for pattern in explicit_help):
        return True

    return False

def is_admin_user(user_id):
    """Check if user ID is in admin list"""
    return user_id in ADMIN_USER_IDS

def has_tellmeajoke_permission(member):
    """Check if user can use tellmeajoke command"""
    # Check if user is admin
    if is_admin_user(member.id):
        return True

    # Check if user has the required role
    if hasattr(member, 'roles'):
        for role in member.roles:
            if role.id == TELLMEAJOKE_ROLE_ID:
                return True

    return False

def can_make_ai_request(user_id):
    """Check if user can make an AI request (rate limiting)"""
    current_time = time.time()
    if user_id in last_ai_request:
        if current_time - last_ai_request[user_id] < AI_COOLDOWN:
            return False
    last_ai_request[user_id] = current_time
    return True

async def discord_api_search(query: str) -> str:
    """Search Discord API for relevant information"""
    if not aiohttp:
        return ""
    
    try:
        # Simple Discord API search for public information
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            # Search Discord's public API docs or status
            search_url = f"https://discord.com/api/v10/applications/@me"
            headers = {
                'Authorization': f'Bot {discord_token}',
                'User-Agent': 'DiscordBot (https://discord.com, 1.0)'
            }
            
            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return f"**🤖 Discord API Info**: Connected as {data.get('name', 'Unknown')} (ID: {data.get('id', 'Unknown')})"
    except Exception as e:
        logger.error(f"Discord API search error: {e}")
    
    return ""

def multi_source_search(query: str) -> str:
    """Search multiple real-time sources for accurate information with enhanced current events coverage"""
    results = []
    query_lower = query.lower()

    # Enhanced current events detection
    current_events_keywords = [
        'news', 'today', 'recent', 'latest', 'current', 'happening', 'died', 'death', 'breaking',
        'arrested', 'caught', 'situation', 'controversy', 'incident', 'scandal', 'trending',
        'shot', 'killed', 'murder', 'crime', 'police', 'investigation', 'celebrity', 'famous'
    ]
    is_current_event = any(keyword in query_lower for keyword in current_events_keywords)

    # 1. News search with reduced timeout
    if news_client and (is_current_event or len(query.split()) <= 3):
        try:
            news_results = news_client.get_everything(
                q=query,
                language='en',
                page_size=2,
                sort_by='publishedAt',
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            )
            
            if news_results['articles']:
                for article in news_results['articles'][:1]:
                    published_date = article['publishedAt'][:10]
                    source = article['source']['name']
                    results.append(f"**📰 {article['title']}** ({source}, {published_date}): {article['description'][:100]}...")
        except Exception as e:
            logger.error(f"News search error: {e}")

    # 2. Quick DuckDuckGo search
    if DDGS and len(results) < 2:
        try:
            with DDGS() as ddgs:
                web_results = list(ddgs.text(query, max_results=1))
                for r in web_results:
                    results.append(f"**🌐 {r['title']}**: {r['body'][:100]}...")
        except Exception as e:
            logger.error(f"DDG search error: {e}")

    # 3. Quick stock search
    if yf and any(keyword in query_lower for keyword in ['stock', 'price', '$']):
        try:
            words = query.upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    try:
                        ticker = yf.Ticker(word)
                        info = ticker.info
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                        if current_price:
                            company_name = info.get('shortName', word)
                            results.append(f"**💹 {company_name} ({word})**: ${current_price:.2f}")
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Stock search error: {e}")

    return "\n\n".join(results) if results else "I couldn't find reliable information for this query right now. Please try rephrasing your question or try again later."

def is_account_too_new(member):
    """Check if account is less than 7 days old"""
    account_age = datetime.now(member.created_at.tzinfo) - member.created_at
    return account_age.days < 7

async def kick_new_account(member):
    """Kick new accounts with appropriate message"""
    try:
        days_remaining = 7 - (datetime.now(member.created_at.tzinfo) - member.created_at).days
        await member.send(f"Please come back in {days_remaining} days. Only accounts 7+ days old can join this server.")
        await member.kick(reason="Account too new (< 7 days)")
        logger.info(f"Kicked {member.name} - account too new")
    except Exception as e:
        logger.error(f"Failed to kick {member.name}: {e}")

# -----------------------------
# Knowledge Base Responses
# -----------------------------
def get_knowledge_response(message_content):
    """Get response based on new knowledge base"""
    text = message_content.lower()

    # Grox response - works in any sentence
    if 'grox' in text:
        grox_responses = [
            "Kill grox",
            "Nobody asked grox",
            "Grox needs to disappear",
            "Imagine being grox 💀",
            "Grox = L + ratio + touch grass"
        ]
        return random.choice(grox_responses)

    # Andrew joke response - only for "who is andrew"
    if 'who is andrew' in text:
        if client:
            try:
                joke_categories = [
                    "career failures and professional incompetence",
                    "social awkwardness and failed relationships",
                    "terrible life choices and poor decision making",
                    "unsuccessful hobbies and boring personality",
                    "lack of intelligence and common sense",
                    "physical appearance and fashion disasters"
                ]

                selected_category = random.choice(joke_categories)
                joke_prompt = f"""Create a harsh but funny two-line roast about Andrew focusing on {selected_category}. Make it brutal but playful. Include an emoji at the end. Be creative and unique."""

                response = client.generate_content(joke_prompt)
                if response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Andrew joke generation error: {e}")

        return "Andrew? Just some random person nobody cares about! 🤷"

    # Rushi response - AI generated
    if 'who is rushi' in text:
        if client:
            try:
                rushi_prompt = """Create a cool, positive response about Rushi being a developer. Make it enthusiastic and mention that she's a cool individual, developer, and nerdy in a positive way. Keep it to 1-2 sentences and include an emoji. Make it sound genuine and awesome."""

                response = client.generate_content(rushi_prompt)
                if response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Rushi response generation error: {e}")

        return "Rushi is an amazing developer, a super cool individual and proudly nerdy! 🤓✨"

    # Werzzzy response
    if 'who is werzzzy' in text:
        werzzzy_responses = [
            "Werzzzy is an absolute legend and coding genius! 🔥",
            "Werzzzy? That's the coolest developer in the game! 💯",
            "Werzzzy is basically a programming wizard ✨",
            "The one and only Werzzzy - pure awesomeness! 🚀"
        ]
        return random.choice(werzzzy_responses)

    # How to read response
    if 'how to read' in text:
        read_jokes = [
            "Here's your reading tutorial: https://www.wikihow.com/Teach-Yourself-to-Read\n\n*Imagine not being able to read in 2025... peak Discord behavior* 📚",
            "Reading lessons: https://www.wikihow.com/Teach-Yourself-to-Read\n\n*This is why aliens don't visit us anymore* 🛸📚",
            "Your literacy salvation: https://www.wikihow.com/Teach-Yourself-to-Read\n\n*POV: You're asking Discord how to read instead of just... reading* 🤦📚"
        ]
        return random.choice(read_jokes)

    # Enhanced macro location detection with fuzzy matching
    macro_keywords = ['macro', 'fisch']
    where_keywords = ['where', 'find', 'location', 'link', 'get']

    # Check if text contains macro-related words and location-seeking words
    has_macro = any(keyword in text for keyword in macro_keywords)
    has_where = any(keyword in text for keyword in where_keywords)

    # Also check for specific patterns
    macro_patterns = [
        'where can i find the fisch macro',
        'where fisch macro',
        'where macro',
        'where fisch',
        'fisch macro location',
        'macro location',
        'macro fisch',
        'fisch macro link',
        'get macro',
        'download macro'
    ]

    if (has_macro and has_where) or any(pattern in text for pattern in macro_patterns):
        return "**Fisch Macro:** https://discord.com/channels/1341949236471926804/1413837110770925578/1417999310443905116"

    # Enhanced config location detection for rod configs - only respond to actual questions
    config_keywords = ['config', 'configs', 'settings']
    rod_keywords = ['rod', 'rods']
    
    # Only respond if it's actually a question about configs
    # Must have question indicators AND config/rod keywords
    has_config = any(keyword in text for keyword in config_keywords)
    has_rod = any(keyword in text for keyword in rod_keywords)
    has_question_word = any(keyword in text for keyword in where_keywords + ['how', 'what'])
    
    # Specific question patterns about rod configs
    config_question_patterns = [
        'where can i find the config',
        'where can i find the fisch config',
        'where fisch config',
        'where config',
        'fisch config location',
        'config location',
        'rod config',
        'configs for rod',
        'fisch rod config',
        'macro config',
        'where rod settings',
        'how to config',
        'what config',
        'need config',
        'find config'
    ]

    # Only show config link if it's clearly a question about configs
    if (has_config and (has_question_word or text.endswith('?'))) or any(pattern in text for pattern in config_question_patterns):
        return "**Fisch Rod Configs:** https://discord.com/channels/1341949236471926804/1411335491457913014"

    # Mango/Fisch macro location
    if 'mango' in text and ('you know' in text or 'where' in text or 'find' in text):
        return "Fisch macro: https://discord.com/channels/1341949236471926804/1413837110770925578/1417999310443905116"

    # General Issues Keywords
    if any(keyword in text for keyword in ['ahk', 'autohotkey', 'auto hotkey']):
        return "**AutoHotkey Version:** Use AHK v1.1 (NOT v2) - v2 is not supported for the current fisch macro."

    if any(keyword in text for keyword in ['roblox version', 'wrong roblox', 'microsoft roblox']):
        return "**Wrong Roblox Version:** Use Web Roblox (Chrome/Brave/etc), NOT Microsoft Store version. Microsoft Roblox will break the macro completely."

    if any(keyword in text for keyword in ['bannable', 'banned', 'ban']) and 'macro' in text:
        return "**Is the macro bannable?** NO - The macro is like an advanced autoclicker. It doesn't inject anything into the game, making it safe and saves you time on games you love."

    if any(keyword in text for keyword in ['roblox settings', 'settings']):
        return "**Roblox Settings:** Fullscreen OFF, graphics 1, dark avatar, brightness/saturation OFF, disable camera shake."

    if any(keyword in text for keyword in ['install', 'installation']):
        return "**Installation Issues:** If AHK fails, uninstall & reinstall. Check antivirus/browser blocking IRUS."

    if any(keyword in text for keyword in ['moved forward', 'moving forward', 'move forward']):
        return "**Being Moved Forward:** Cause = click-to-move enabled or failed catch. Fix = disable click-to-move, use better rods+bait+configs to reduce fails."

    # Debugging Flow
    if any(keyword in text for keyword in ['shake not working', 'shake issue', 'debug shake']):
        return "**Shake Not Working:** If mouse not moving → wrong Roblox version (use Web Roblox, not Microsoft)."

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
async def on_member_join(member):
    """Auto-kick accounts less than 7 days old"""
    if is_account_too_new(member):
        await kick_new_account(member)

@bot.event
async def on_member_update(before, after):
    """Enforce persistent nicknames"""
    if str(after.id) in persistent_names:
        enforced_nick = persistent_names[str(after.id)]
        if after.display_name != enforced_nick:
            try:
                await after.edit(nick=enforced_nick, reason="Name persistence enforced")
                logger.info(f"Reset {after.name}'s nickname to {enforced_nick}")
            except Exception as e:
                logger.error(f"Failed to reset nickname for {after.name}: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Respond to DMs or mentions
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions

    # Special responses that don't need question indicators
    special_triggers = ['grox', 'who is andrew', 'who is rushi', 'who is werzzzy', 'how to read']

    # Check for special triggers or use advanced question detection
    has_special_trigger = any(trigger in message.content.lower() for trigger in special_triggers)
    is_question = is_advanced_question(message.content)

    # Respond if it's a DM, mention, has special trigger, or is a question
    if is_dm or is_mentioned or has_special_trigger or is_question:
        response = get_knowledge_response(message.content)
        if response:
            await message.reply(response)
        elif is_dm or is_mentioned:
            # If no specific response but it's a DM/mention, give a general response
            await message.reply("Hey! I'm here to help with Fisch macros or you can use `/askbloom` to ask me anything!")

    await bot.process_commands(message)

# -----------------------------
# Commands
# -----------------------------
@bot.tree.command(name="askbloom", description="Ask Bloom anything with web search for accurate info")
@app_commands.describe(question="Your question - I'll search the web for current information!")
async def askbloom_command(interaction: discord.Interaction, question: str):
    if not client:
        await interaction.response.send_message("❌ AI service not available.", ephemeral=True)
        return

    # Rate limiting check
    if not can_make_ai_request(interaction.user.id):
        await interaction.response.send_message("⏱️ Please wait a few seconds before asking another question!", ephemeral=True)
        return

    # Content filter
    question_lower = question.lower()
    banned_words = ['racist', 'racism', 'nazi', 'hitler', 'slur', 'hate speech', 'nigger', 'faggot']

    if any(word in question_lower for word in banned_words):
        await interaction.response.send_message("❌ I can't help with that. Ask something else!", ephemeral=True)
        return

    # Send initial response and defer for longer processing
    await interaction.response.defer()

    try:
        # First check if it's a Fisch macro related question
        knowledge_response = get_knowledge_response(question)

        if knowledge_response:
            # Use existing knowledge base for Fisch-related questions
            await interaction.followup.send(knowledge_response)
        else:
            try:
                # Quick Discord API search first
                discord_info = await discord_api_search(question)
                
                # Run optimized search with shorter timeout
                search_results = multi_source_search(question)
                
                # Combine Discord info with search results
                if discord_info:
                    search_results = discord_info + "\n\n" + search_results
                
                if not search_results or search_results.strip() == "":
                    await interaction.followup.send("No information found. Please try rephrasing your question.")
                    return

                prompt = f"""You are Bloom, a Discord bot assistant. Analyze the search results and provide a direct, accurate answer.

SEARCH RESULTS:
{search_results}

USER QUESTION: {question}

Provide a helpful response under 1800 characters. Be direct and informative."""

                try:
                    response = client.generate_content(prompt)
                    if response.text:
                        answer = response.text.strip()
                        if len(answer) > 1800:
                            answer = answer[:1797] + "..."
                        await interaction.followup.send(answer)
                    else:
                        await interaction.followup.send("❌ Couldn't generate response. Try again!")
                except Exception as ai_error:
                    error_str = str(ai_error)
                    logger.error(f"AI generation error: {ai_error}")
                    if "429" in error_str or "Too Many Requests" in error_str:
                        await interaction.followup.send("⏱️ AI service is currently rate limited. Please try again in a few minutes!")
                    else:
                        await interaction.followup.send("❌ AI service temporarily unavailable. Try again!")

            except Exception as search_error:
                logger.error(f"Search error: {search_error}")
                await interaction.followup.send("❌ Search failed. Try a simpler question or try again later.")

    except asyncio.TimeoutError:
        logger.error("AskBloom timeout")
        await interaction.followup.send("❌ Request timed out. Try a simpler question!")
    except Exception as e:
        logger.error(f"AskBloom error: {e}")
        await interaction.followup.send("❌ Something went wrong. Try again!")

@bot.tree.command(name="ban", description="Ban a user (Admin only)")
@app_commands.describe(user="User to ban")
async def ban_command(interaction: discord.Interaction, user: discord.Member):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        return

    try:
        await user.ban(reason=f"Banned by {interaction.user.name}")
        await interaction.response.send_message(f"✅ Banned {user.mention}")
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to ban user: {e}", ephemeral=True)

@bot.tree.command(name="namepersist", description="Force a user to keep a specific nickname (Admin only)")
@app_commands.describe(user="User to enforce nickname on", nickname="Nickname to enforce")
async def namepersist_command(interaction: discord.Interaction, user: discord.Member, nickname: str):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        return

    try:
        await user.edit(nick=nickname, reason=f"Name persistence set by {interaction.user.name}")
        persistent_names[str(user.id)] = nickname
        save_persistent_data()
        await interaction.response.send_message(f"✅ {user.mention} will now always have the nickname: **{nickname}**")
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to set persistent nickname: {e}", ephemeral=True)

@bot.tree.command(name="say", description="Make Bloom say something (Admin only)")
@app_commands.describe(
    words="What should Bloom say?",
    channel="Optional: Channel to send message to (current channel if not specified)"
)
async def say_command(interaction: discord.Interaction, words: str, channel: discord.TextChannel = None):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        return

    if len(words) > 2000:
        await interaction.response.send_message("❌ Message too long! Keep it under 2000 characters.", ephemeral=True)
        return

    target_channel = channel or interaction.channel

    try:
        await target_channel.send(words)
        # Silent confirmation - only respond if sending to different channel
        if channel and channel != interaction.channel:
            await interaction.response.send_message(f"✅ Message sent to {channel.mention}", ephemeral=True)
        else:
            await interaction.response.send_message("✅", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to send message: {e}", ephemeral=True)

@bot.tree.command(name="saywb", description="Make Bloom say something with embed (Admin only)")
@app_commands.describe(
    title="Embed title",
    text="Embed text content",
    channel="Optional: Channel to send embed to (current channel if not specified)"
)
async def saywb_command(interaction: discord.Interaction, title: str, text: str, channel: discord.TextChannel = None):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        return

    if len(title) > 256:
        await interaction.response.send_message("❌ Title too long! Keep it under 256 characters.", ephemeral=True)
        return

    if len(text) > 4096:
        await interaction.response.send_message("❌ Text too long! Keep it under 4096 characters.", ephemeral=True)
        return

    target_channel = channel or interaction.channel

    try:
        embed = discord.Embed(title=title, description=text, color=0x00ff00)
        await target_channel.send(embed=embed)
        # Silent confirmation - only respond if sending to different channel
        if channel and channel != interaction.channel:
            await interaction.response.send_message(f"✅ Embed sent to {channel.mention}", ephemeral=True)
        else:
            await interaction.response.send_message("✅", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to send embed: {e}", ephemeral=True)

@bot.tree.command(name="tellmeajoke", description="Get a custom AI-generated joke")
@app_commands.describe(context="Context for the joke (e.g., 'say something bad about my name')")
async def tellmeajoke_command(interaction: discord.Interaction, context: str):
    # Check permissions
    if not has_tellmeajoke_permission(interaction.user):
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        return

    if not client:
        await interaction.response.send_message("❌ AI service not available.", ephemeral=True)
        return

    # Rate limiting check
    if not can_make_ai_request(interaction.user.id):
        await interaction.response.send_message("⏱️ Please wait a few seconds before requesting another joke!", ephemeral=True)
        return

    # Content filter
    context_lower = context.lower()
    banned_words = ['racist', 'racism', 'nazi', 'hitler', 'slur', 'hate speech', 'nigger', 'faggot']

    if any(word in context_lower for word in banned_words):
        await interaction.response.send_message("❌ I can't make jokes about that. Try something else!", ephemeral=True)
        return

    try:
        prompt = f"""You are Bloom, a Discord bot with a sharp sense of humor. Create a funny, clever joke based on this context: "{context}"

Make it witty and humorous but not offensive or mean-spirited. Keep it under 500 characters and add an appropriate emoji at the end. Be creative and entertaining!"""

        response = client.generate_content(prompt)
        if response.text:
            joke = response.text.strip()
            if len(joke) > 500:
                joke = joke[:497] + "..."
            await interaction.response.send_message(joke)
        else:
            await interaction.response.send_message("❌ Couldn't generate a joke. Try again!", ephemeral=True)

    except Exception as e:
        error_str = str(e)
        logger.error(f"Tellmeajoke error: {e}")
        if "429" in error_str or "Too Many Requests" in error_str:
            await interaction.response.send_message("⏱️ AI service is currently rate limited. Please try again in a few minutes!", ephemeral=True)
        else:
            await interaction.response.send_message("❌ Something went wrong. Try again!", ephemeral=True)

@bot.tree.command(name="whatisthisserverabout", description="Learn about this Discord server")
async def whatisthisserverabout_command(interaction: discord.Interaction):
    server_responses = [
        "🥭 **This server is all about mangoes and tiny tasks!** We're building something amazing here - this server will soon be a big Discord server as we have big plans! Join us on this exciting journey! 🚀",

        "🍃 **Welcome to the mango paradise!** This community focuses on mangoes and small but important tasks. We're growing fast and have huge plans ahead - you're part of something special! 🌟",

        "🥭 **Mango lovers unite!** This server revolves around mangoes and managing tiny tasks together. We're small now but we have massive plans brewing. Welcome to what will soon be an epic Discord community! ⚡",

        "🌱 **The mango hub is here!** We're all about those sweet mangoes and tackling small tasks together. This server is destined for greatness - stick around for the journey! 🎯",

        "🥭 **Mangoes + Tiny Tasks = Magic!** That's what this server is about! We're small now but we have massive plans brewing. Welcome to what will soon be an epic Discord community! ⚡"
    ]

    random_response = random.choice(server_responses)
    await interaction.response.send_message(random_response)

# -----------------------------
# Web server for deployment
# -----------------------------
from flask import Flask
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return "Discord Bot is running!"

@app.route('/health')
def health():
    return {"status": "healthy", "bot": "online"}

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

# -----------------------------
# Run bot
# -----------------------------
if __name__ == "__main__":
    if not discord_token:
        logger.error("TOKEN not found. Cannot start bot.")
        exit(1)

    # Start Flask server in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("✅ Web server started on port 5000")

    import time
    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            bot.run(discord_token)
            break  # If successful, exit the loop
        except discord.errors.HTTPException as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                retry_count += 1
                wait_time = min(300, (2 ** retry_count) * 30)  # Exponential backoff, max 5 minutes
                logger.warning(f"Rate limited. Retry {retry_count}/{max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP error: {e}")
                break
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Bot error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 60
                logger.info(f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Exiting.")
                break
