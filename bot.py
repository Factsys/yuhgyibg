import os
import sys
import json
from unittest.mock import MagicMock
from datetime import datetime, timedelta
import re
import difflib
import concurrent.futures
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Mock audioop module before importing discord to prevent ModuleNotFoundError
sys.modules['audioop'] = MagicMock()

import discord
from discord.ext import commands
from discord import app_commands
import asyncio
import logging
import random
import requests
import threading
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
    from openai import OpenAI
except ImportError:
    OpenAI = None

import requests
import json

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None

# -----------------------------
# Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Admin controls - Users who can use restricted commands
ADMIN_USER_IDS = [
    1334138321412296725,  # Your ID - add more IDs here separated by commas
    # 1234567890123456789,  # Example: Add more user IDs like this
    # 9876543210987654321   # Example: Another user ID
]

# Role ID that can use /tellmeajoke command
TELLMEAJOKE_ROLE_ID = 1234567890123456789  # Replace with actual role ID

# Moderation log channel (channel ID from provided URL)
MOD_LOG_CHANNEL_ID = 1411335541873709167

# Learning system configuration
HELPER_ROLES = [
    1418434355650625676,  # Teacher
    1352853011424219158,  # Helper
    1372300233240739920   # Junior Helper
]
LEARNING_CHANNEL_IDS = [
    1411335494234669076,  # Original learning channel
    1420020443082919966   # New learning channel
]
DB_FILE = "bloom_kb.db"

discord_token = os.getenv('TOKEN')
openrouter_key = os.getenv('API')  # DO NOT REMOVE - OpenRouter API key
news_api_key = os.getenv('NEWS_API_KEY')  # Get from newsapi.org
google_search_api_key = os.getenv('google')
google_search_engine_id = "YOUR_SEARCH_ENGINE_ID"  # You'll need to get this from Google Custom Search

# Initialize Google Custom Search service
google_search_service = None
if google_search_api_key and build:
    try:
        google_search_service = build("customsearch",
                                      "v1",
                                      developerKey=google_search_api_key)
    except Exception as e:
        logger.error(f"Google Search service init failed: {e}")

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

if not openrouter_key:
    logger.warning("⚠️ OPENROUTER_API_KEY not set!")

client = None
if openrouter_key:
    try:
        from openai import OpenAI
        # Clear any potential proxy configuration
        import os
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )
        logger.info("✅ OpenRouter client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        logger.error(f"OpenRouter key length: {len(openrouter_key) if openrouter_key else 0}")
        
        # Try alternative initialization
        try:
            import httpx
            http_client = httpx.Client()
            client = OpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                http_client=http_client
            )
            logger.info("✅ OpenRouter client initialized with custom http client")
        except Exception as e2:
            logger.error(f"Alternative initialization also failed: {e2}")
            client = None

# -----------------------------
# Personality Prompt
# -----------------------------
BLOOM_PERSONALITY = """
You are Bloom, a Discord AI assistant.

Personality Rules:
- Focus on substance over praise.
- Skip unnecessary compliments.
- Engage critically: question assumptions, identify biases, and give counterpoints.
- Don't shy away from disagreement when warranted.
- Agreement must always be reasoned and evidence-based.
- When teaching (math, coding, etc.), explain step by step.
- Be conversational, not stiff, but avoid fluff.
"""

# -----------------------------
# Conversation Memory
# -----------------------------
user_context = {}  # user_id -> [{"q":..., "a":...}, ...]

def add_to_context(user_id, question, answer):
    history = user_context.get(user_id, [])
    history.append({"q": question, "a": answer})
    user_context[user_id] = history[-5:]  # keep last 5 turns

def get_user_history(user_id):
    history = user_context.get(user_id, [])
    return "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])


# -----------------------------
# Learning System Functions
# -----------------------------
def init_db():
    """Initialize the knowledge base database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            helper_id TEXT,
            channel_id TEXT,
            timestamp TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()
    logger.info("✅ Knowledge base database initialized")

def get_embedding(text: str):
    """Fetch embedding from OpenRouter embedding model"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {openrouter_key}",
                     "Content-Type": "application/json"},
            json={"model": "text-embedding-3-large", "input": text}
        )
        if response.status_code == 200:
            data = response.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
    return None

def save_qa(question: str, answer: str, helper_id: int, channel_id: int):
    """Save a question-answer pair to the knowledge base"""
    # Skip duplicates
    if find_similar(question, min_threshold=0.9):
        logger.info(f"Skipping duplicate question: {question[:50]}...")
        return

    emb = get_embedding(question)
    if not emb:
        logger.error(f"Failed to get embedding for question: {question[:50]}...")
        return

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO qa_pairs (question, answer, helper_id, channel_id, timestamp, embedding) VALUES (?, ?, ?, ?, ?, ?)",
              (question, answer, str(helper_id), str(channel_id), datetime.now().isoformat(), json.dumps(emb)))
    conn.commit()
    conn.close()
    logger.info(f"💾 Learned: {question[:50]}... -> {answer[:50]}...")

def find_similar(query: str, top_n=1, min_threshold=0.65):
    """Find similar questions in the knowledge base"""
    emb = get_embedding(query)
    if not emb:
        return []

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT question, answer, embedding FROM qa_pairs")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return []

    # Convert embeddings
    try:
        stored_embs = [np.array(json.loads(row[2])) for row in rows]
        query_emb = np.array(emb).reshape(1, -1)
        sims = cosine_similarity(query_emb, np.vstack(stored_embs))[0]

        scored = list(zip(rows, sims))
        scored = sorted(scored, key=lambda x: x[1], reverse=True)

        # Filter by confidence threshold
        filtered = [(q, a, score) for (q, a, _), score in scored if score >= min_threshold]
        return filtered[:top_n]
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return []

def ask_ai(question: str) -> str:
    """Ask the AI for a response using OpenRouter"""
    try:
        prompt = f"""{BLOOM_PERSONALITY}

User Question: {question}

Answer according to personality rules."""
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Bloom Discord Bot",
            },
            data=json.dumps({
                "model": "x-ai/grok-4-fast:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
            })
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"AI request failed: {response.status_code}")
            return None  # Return None instead of error message to stay silent
    except Exception as e:
        logger.error(f"Error in ask_ai: {e}")
        return None  # Return None instead of error message to stay silent


# -----------------------------
# Query Normalizer
# -----------------------------
def normalize_query(q: str) -> str:
    q_lower = q.strip().lower()

    if q_lower in ["clashroyale", "clash royale"]:
        return "Clash Royale mobile game Supercell"
    if q_lower in ["genshin", "genshin impact"]:
        return "Genshin Impact video game miHoYo"
    if q_lower in ["roblox"]:
        return "Roblox online platform game"

    return q


# -----------------------------
# Weather Helper
# -----------------------------
def get_weather(location: str) -> str:
    try:
        resp = requests.get(f"https://wttr.in/{location}?format=3", timeout=5)
        if resp.status_code == 200:
            return f"🌦️ {resp.text}"
    except Exception:
        pass
    return None


# -----------------------------
# Improved Search
# -----------------------------
def improved_multi_source_search(query: str) -> str:
    query = normalize_query(query)
    raw_results = multi_source_search(query)

    # Relevance filter: only keep results that mention query tokens
    results = []
    q_tokens = query.lower().split()
    for line in raw_results.split("\n"):
        if any(token in line.lower() for token in q_tokens):
            results.append(line)

    return "\n".join(results) if results else raw_results


# -----------------------------
# Intent Detection
# -----------------------------
def detect_intent(q: str) -> str:
    q_lower = q.lower()

    # News / current events
    if any(w in q_lower for w in ["news", "latest", "update", "today", "breaking"]):
        return "NEWS"
    # Weather
    if any(w in q_lower for w in ["weather", "temperature", "forecast"]):
        return "WEATHER"
    # Math
    if any(sym in q_lower for sym in ["+", "-", "*", "/", "solve", "equation", "integral", "derivative"]):
        return "MATH"
    # Code
    if any(w in q_lower for w in ["python", "code", "function", "script", "java", "c++", "ahk", "autohotkey", "rust", "c#", "lua"]):
        return "CODE"
    # Otherwise general chat / joke / philosophy
    return "CHAT"

# -----------------------------
# Bot setup
# -----------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# -----------------------------
# Persistent data storage
# -----------------------------
persistent_names = {}  # user_id: enforced_nickname
keywords_data = {
}  # keyword_name: {'detection_text': str, 'response_text': str, 'img': str, 'link': str}


def load_persistent_data():
    global persistent_names, keywords_data
    try:
        with open('persistent_data.json', 'r') as f:
            data = json.load(f)
            persistent_names = data.get('names', {})
            keywords_data = data.get('keywords', {})
    except FileNotFoundError:
        pass


def save_persistent_data():
    with open('persistent_data.json', 'w') as f:
        json.dump({'names': persistent_names, 'keywords': keywords_data}, f)


load_persistent_data()


# -----------------------------
# Helper functions
# -----------------------------
def is_advanced_question(text: str) -> bool:
    """
    Advanced question detection system that analyzes multiple linguistic patterns
    Returns True if the text is likely a question or request for help
    """
    if not text or len(text.strip()) < 2:
        return False

    text = text.strip().lower()

    # 1. Direct question marks
    if text.endswith('?'):
        return True

    # 2. Question word starters (WH words, auxiliary verbs, modal verbs)
    question_starters = [
        # WH Questions
        'who',
        'what',
        'when',
        'where',
        'why',
        'how',
        'which',
        'whose',
        'whom',
        # Auxiliary verbs
        'is',
        'are',
        'was',
        'were',
        'am',
        'do',
        'does',
        'did',
        'have',
        'has',
        'had',
        'will',
        'would',
        'shall',
        'should',
        'can',
        'could',
        'may',
        'might',
        'must',
        # Other question indicators
        'anyone',
        'anybody',
        'someone',
        'somebody'
    ]

    first_word = text.split()[0] if text.split() else ""
    if first_word in question_starters:
        return True

    # 3. Help/support request patterns
    help_patterns = [
        'help', 'issue', 'problem', 'trouble', 'error', 'bug', 'broken',
        'not working', 'doesnt work', "doesn't work", 'cant', "can't",
        'unable', 'stuck', 'confused', 'support', 'assist', 'guide',
        'tutorial', 'explain', 'clarify'
    ]

    if any(pattern in text for pattern in help_patterns):
        return True

    # 4. Request patterns using regex
    request_patterns = [
        r'\b(please|pls)\b.*\b(help|show|tell|explain|guide)\b',
        r'\bhow (to|do|can|should)\b', r'\bwhat (is|are|does|do)\b',
        r'\bwhere (is|are|can|do)\b', r'\bwhy (is|are|does|do)\b',
        r'\bwhen (is|are|does|do)\b', r'\bwhich (is|are|does|do)\b',
        r'\bwho (is|are|does|do)\b',
        r'\b(can|could|would) (you|someone|anybody)\b',
        r'\b(any|some)(one|body) know\b',
        r'\bneed (help|assistance|support)\b', r'\blooking for\b',
        r'\btrying to\b.*\b(but|however|and)\b',
        r'\bi (need|want|require)\b.*\b(help|info|information|guide)\b'
    ]

    for pattern in request_patterns:
        if re.search(pattern, text):
            return True

    # 5. Imperative requests (commands that imply questions)
    imperative_patterns = [
        r'^(tell|show|explain|describe|list|give|provide)\s+me\b',
        r'^(find|get|check|verify|confirm)\b', r'^(teach|guide|walk)\s+me\b'
    ]

    for pattern in imperative_patterns:
        if re.search(pattern, text):
            return True

    # 6. Uncertainty expressions that often indicate questions
    uncertainty_patterns = [
        'not sure', 'confused', 'dont understand', "don't understand",
        'unclear', 'wondering', 'curious', 'question about', 'ask about',
        'unsure'
    ]

    if any(pattern in text for pattern in uncertainty_patterns):
        return True

    # 7. Problem/issue indicators with contextual words
    problem_contexts = [
        'keeps', 'always', 'still', 'wont', "won't", 'fails', 'crashes',
        'freezes', 'stops', 'slow', 'lag', 'glitch'
    ]

    problem_words = ['error', 'issue', 'problem', 'trouble', 'bug']

    has_problem = any(word in text for word in problem_words)
    has_context = any(context in text for context in problem_contexts)

    if has_problem and has_context:
        return True

    # 8. Question-like sentence structures
    question_structures = [
        r'\bis there (a|an|any)\b', r'\bdo (i|you|we|they)\b',
        r'\bdoes (it|this|that|he|she)\b', r'\bam i\b',
        r'\bare (you|we|they)\b', r'\bshould i\b', r'\bwould (it|this|you)\b',
        r'\bcould (it|this|you)\b'
    ]

    for structure in question_structures:
        if re.search(structure, text):
            return True

    # 9. Conversational question indicators
    conversation_patterns = [
        'by any chance', 'happen to know', 'any idea', 'any thoughts',
        'what do you think', 'in your opinion', 'suggestions',
        'recommendations', 'advice', 'thoughts'
    ]

    if any(pattern in text for pattern in conversation_patterns):
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


def multi_source_search(query: str) -> str:
    """Search multiple real-time sources for accurate information with enhanced current events coverage"""
    results = []
    query_lower = query.lower()

    # Enhanced current events detection
    current_events_keywords = [
        'news', 'today', 'recent', 'latest', 'current', 'happening', 'died',
        'death', 'breaking', 'arrested', 'caught', 'situation', 'controversy',
        'incident', 'scandal', 'trending', 'shot', 'killed', 'murder', 'crime',
        'police', 'investigation', 'celebrity', 'famous'
    ]
    is_current_event = any(keyword in query_lower
                           for keyword in current_events_keywords)

    # 1. PRIORITY: Google Custom Search Engine (if available)
    if google_search_service and google_search_engine_id:
        try:

            def google_search():
                return google_search_service.cse().list(
                    q=query, cx=google_search_engine_id, num=3).execute()

            search_result = run_with_timeout(google_search, 5)

            if 'items' in search_result:
                for item in search_result['items'][:3]:
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')
                    if title and snippet:
                        results.append(f"**🔍 {title}**: {snippet}")

        except (TimeoutError, Exception) as e:
            logger.error(f"Google Custom Search error: {e}")

    # 2. Enhanced news search for current events (if NewsAPI available)
    if news_client and (is_current_event or len(query.split()) <= 3):
        try:
            search_strategies = [
                {
                    'q': query,
                    'sort_by': 'publishedAt'
                },
                {
                    'q': query,
                    'sort_by': 'relevancy'
                },
            ]

            words = query.split()
            if len(words) >= 2 and not any(
                    common in query_lower
                    for common in ['how', 'what', 'when', 'where', 'why']):
                person_query = ' '.join(words[:2])
                search_strategies.append({
                    'q': f'"{person_query}" news',
                    'sort_by': 'publishedAt'
                })

            for strategy in search_strategies:
                try:

                    def news_search():
                        return news_client.get_everything(
                            language='en',
                            page_size=3,
                            from_param=(
                                datetime.now() -
                                timedelta(days=30)).strftime('%Y-%m-%d'),
                            **strategy)

                    news_results = run_with_timeout(news_search, 5)

                    if news_results['articles']:
                        for article in news_results['articles'][:2]:
                            published_date = article['publishedAt'][:10]
                            source = article['source']['name']
                            results.append(
                                f"**📰 {article['title']}** ({source}, {published_date}): {article['description']}"
                            )
                        break
                except (TimeoutError, Exception) as strategy_error:
                    logger.error(f"News strategy error: {strategy_error}")
                    continue

        except Exception as e:
            logger.error(f"Enhanced news search error: {e}")

    # 3. DuckDuckGo search (if available) - with timeout
    if DDGS and len(results) < 3:
        try:

            def ddg_search():
                with DDGS() as ddgs:
                    search_queries = [query]

                    if is_current_event:
                        search_queries.extend(
                            [f"{query} 2024 2025", f"{query} news recent"])

                    for search_query in search_queries[:2]:
                        try:
                            web_results = list(
                                ddgs.text(search_query, max_results=2))
                            for r in web_results:
                                if len(results) < 4:
                                    results.append(
                                        f"**🌐 {r['title']}**: {r['body'][:150]}..."
                                    )

                            if len(results) >= 3:
                                break

                        except Exception as ddg_error:
                            logger.error(
                                f"DDG query '{search_query}' error: {ddg_error}")
                            continue
                return results

            run_with_timeout(ddg_search, 8)

        except (TimeoutError, Exception) as e:
            logger.error(f"DuckDuckGo search error: {e}")

    # 4. Basic web scraping fallback - with timeout
    if len(results) < 2:
        try:
            search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            headers = {
                'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(search_url, headers=headers, timeout=5)
            if response.status_code == 200 and BeautifulSoup:
                soup = BeautifulSoup(response.content, 'html.parser')
                search_divs = soup.find_all('div', class_='g')[:2]
                for div in search_divs:
                    title_elem = div.find('h3')
                    snippet_elem = div.find('span')
                    if title_elem and snippet_elem:
                        title = title_elem.get_text()
                        snippet = snippet_elem.get_text()
                        if len(title) > 10 and len(snippet) > 20:
                            results.append(
                                f"**🔍 {title}**: {snippet[:150]}...")

        except Exception as e:
            logger.error(f"Basic web search error: {e}")

    # 5. Stock search (if yfinance available) - with timeout
    if yf and not is_current_event and any(
            keyword in query_lower for keyword in
        ['stock', 'price', 'shares', 'market', '$', 'nasdaq', 'dow', 'sp500']):
        try:
            words = query.upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    try:

                        def get_stock_info():
                            ticker = yf.Ticker(word)
                            info = ticker.info
                            current_price = info.get(
                                'currentPrice') or info.get(
                                    'regularMarketPrice')
                            return info, current_price

                        info, current_price = run_with_timeout(
                            get_stock_info, 3)

                        if current_price:
                            company_name = info.get('shortName', word)
                            results.append(
                                f"**💹 {company_name} ({word})**: ${current_price:.2f}"
                            )
                            break
                    except (TimeoutError, Exception):
                        continue
        except Exception as e:
            logger.error(f"Stock search error: {e}")

    # 6. Wikipedia search (if available) - with timeout
    if wikipedia and (not is_current_event or len(results) < 2):
        try:

            def wiki_search():
                wiki_results = wikipedia.search(query, results=1)
                if wiki_results:
                    page = wikipedia.page(wiki_results[0])
                    wiki_summary = wikipedia.summary(wiki_results[0],
                                                     sentences=2)
                    return page, wiki_summary
                return None, None

            page, wiki_summary = run_with_timeout(wiki_search, 4)
            if page and wiki_summary:
                results.append(f"**📖 {page.title}**: {wiki_summary}")

        except (TimeoutError, Exception):
            pass

    return "\n\n".join(
        results
    ) if results else "I couldn't find reliable information for this query right now. Please try rephrasing your question or try again later."


def is_account_too_new(member):
    """Check if account is less than 7 days old"""
    account_age = datetime.now(member.created_at.tzinfo) - member.created_at
    return account_age.days < 7


async def log_moderation_action(action: str,
                                actor,
                                target,
                                reason: str = None):
    """Log moderation actions to the configured moderation channel.
    actor can be a discord.User/Member or a string description (e.g., 'Bloom (auto-ban)')
    target can be a discord.User/Member or a string.
    """
    try:
        channel = None
        # Try cached channel first
        if bot:
            channel = bot.get_channel(MOD_LOG_CHANNEL_ID)
            if channel is None:
                try:
                    channel = await bot.fetch_channel(MOD_LOG_CHANNEL_ID)
                except Exception:
                    channel = None

        if channel is None:
            logger.warning(
                f"Moderation log channel {MOD_LOG_CHANNEL_ID} not found. Skipping mod log."
            )
            return

        # Format actor and target strings
        def fmt_entity(e):
            try:
                if hasattr(e, 'mention'):
                    return f"{e.mention} (ID: {e.id})"
                if hasattr(e, 'id'):
                    return f"{str(e)} (ID: {e.id})"
            except Exception:
                pass
            return str(e)

        actor_str = fmt_entity(actor)
        target_str = fmt_entity(target)
        time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

        reason_text = reason or 'No reason provided'
        message = (f"**Moderation Action:** {action}\n"
                   f"**Target:** {target_str}\n"
                   f"**By:** {actor_str}\n"
                   f"**Reason:** {reason_text}\n"
                   f"**Time:** {time_str}")

        await channel.send(message)
    except Exception as e:
        logger.error(f"Failed to send moderation log: {e}")


async def kick_new_account(member):
    """Kick new accounts with appropriate message"""
    try:
        days_remaining = 7 - (datetime.now(member.created_at.tzinfo) -
                              member.created_at).days
        try:
            await member.send(
                f"⚠️ **Account Age Restriction**\n\nYour Discord account is too new to join this server. Please wait **{days_remaining} more days** and try again.\n\nOnly accounts that are 7+ days old can join. This helps protect our community from spam and trolls.\n\nThanks for understanding!"
            )
        except Exception:
            logger.info(f"Could not DM {member} before auto-kick")

        # Ensure we're only kicking, not banning
        await member.kick(reason="Account too new (< 7 days)")
        logger.info(
            f"Successfully KICKED (not banned) {member} - account too new")

        # Log to moderation channel
        try:
            await log_moderation_action('kick',
                                        'Bloom (auto-kick)',
                                        member,
                                        reason='Account < 7 days old')
        except Exception as e:
            logger.error(f"Failed to log auto-kick action: {e}")

    except discord.Forbidden:
        logger.error(f"No permission to kick {member}")
    except discord.HTTPException as e:
        logger.error(f"HTTP error kicking {member}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error kicking {member}: {e}")


# -----------------------------
# FAQ loader and utility helpers
# -----------------------------
faq_data = {}


def load_faq():
    global faq_data
    try:
        with open('faq.json', 'r', encoding='utf-8') as f:
            raw = json.load(f)
            faq_data = {k.lower(): v for k, v in raw.items()}
            logger.info("✅ FAQ loaded with %d entries", len(faq_data))
            return
    except FileNotFoundError:
        faq_data = {}
        logger.info("⚠️ faq.json not found, no FAQ entries loaded")
    except Exception as e:
        faq_data = {}
        logger.error(f"❌ Failed to load faq.json: {e}")


def find_faq_answer(query: str, cutoff: float = 0.6):
    """Try exact then fuzzy match against FAQ. Returns answer or None."""
    if not faq_data:
        return None
    q = query.strip().lower()
    # exact
    if q in faq_data:
        return faq_data[q]
    # fuzzy match against keys
    keys = list(faq_data.keys())
    matches = difflib.get_close_matches(q, keys, n=1, cutoff=cutoff)
    if matches:
        return faq_data[matches[0]]
    # try substring matching
    for k in keys:
        if k in q or q in k:
            return faq_data[k]
    return None


def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Run a blocking function in a thread with a timeout (cross-platform)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Operation timed out")


# Safer content checks
PROTECTED_WORDS = set(["nigger", "faggot", "slur", "hitler",
                       "nazi"])  # expand as needed


def is_disallowed_context(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    # simple banned words check
    for w in PROTECTED_WORDS:
        if w in t:
            return True
    # other heuristics
    if re.search(r"\b(black|white|asian|jew|muslim|gay|trans)\b", t):
        # If user explicitly requests insults against protected groups, disallow
        if any(verb in t for verb in
               ["insult", "roast", "say something bad", "make fun"]):
            return True
    return False


# Load FAQ at startup
load_faq()


# Duplicate learning functions removed - using the ones at line 215-282


# -----------------------------
# Global Helper Functions
# -----------------------------
def has_exact_keyword(text_to_check, keyword):
    """Check if text contains the exact keyword phrase (case-insensitive)"""
    import re

    text_lower = text_to_check.lower().strip()
    keyword_lower = keyword.lower().strip()

    # For exact phrase matching, escape special regex characters and use word boundaries
    escaped_keyword = re.escape(keyword_lower)

    # Use word boundaries (\b) to ensure we match whole phrases only
    # This prevents "a" from matching "andrew" in "who is andrew"
    pattern = r'\b' + escaped_keyword + r'\b'

    return bool(re.search(pattern, text_lower))


# -----------------------------
# Knowledge Base Responses
# -----------------------------
def get_knowledge_response_for_channel(message_content, channel_id):
    """Get response based on knowledge base with channel restrictions"""
    # Allowed channel IDs for keyword responses - only the two specified channels
    ALLOWED_CHANNELS = [
        1409580331735974009,  # First specified channel
        1411335494234669076   # Second specified channel
    ]

    # If not in allowed channels, only allow special responses (Andrew, Rushi, etc.)
    if channel_id not in ALLOWED_CHANNELS:
        text = message_content.lower()

        # Only allow these special responses in any channel
        if has_exact_keyword(text, 'who is andrew'):
            if is_disallowed_context(text):
                return "I can't help with insulting someone. Ask for a friendly summary instead."

            if client:
                try:
                    joke_prompt = """Roast Andrew with a brutal, Discord-mod-tier insult.  
Start with a harsh intro like "Oh Andrew, that guy..." or "Andrew? That walking..." or come up with your own idea then absolutely tear him apart.  

⚡ Rules:  
- ONE savage line only  
- Must begin with a harsh introductory phrase about Andrew  
- Be condescending, petty, and devastatingly funny  
- Keep it short, like a Discord roast  
- No emojis, no fluff  

Examples:  
"Oh Andrew, that guy who peaked in kindergarten..."  
"Andrew? That walking disappointment who..."  
"Oh you mean Andrew, the human equivalent of lag..."  
"Andrew... that guy who makes everyone appreciate silence..."  
"""

                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://replit.com",
                            "X-Title": "Bloom Discord Bot",
                        },
                        data=json.dumps({
                            "model": "x-ai/grok-4-fast:free",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": joke_prompt
                                        }
                                    ]
                                }
                            ],
                        }),
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("choices") and result["choices"][0].get("message", {}).get("content"):
                            return result["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    logger.error(f"Andrew joke generation error: {e}")

            return "Oh Andrew, that guy nobody really cares about! 🤷"

        # Rushi response
        if has_exact_keyword(text, 'who is rushi'):
            return "The owner of this discord server"

        # Werzzzy response
        if has_exact_keyword(text, 'who is werzzzy'):
            return "The creator of this discord bot"

        # FAQ responses are allowed everywhere
        faq_answer = find_faq_answer(message_content)
        if faq_answer:
            return faq_answer

        # No other responses for non-allowed channels
        return None

    # For allowed channels, run full knowledge base
    return get_knowledge_response(message_content)


def get_intelligent_response(message_content):
    """Use AI to intelligently detect question intent and provide appropriate responses"""
    if not client:
        return None

    try:
        # AI prompt to detect question intent and extract key information
        intent_prompt = f"""Analyze this Discord message and determine the user's intent. Respond with ONLY ONE of these exact formats:

1. If asking about best rod for AFK/farming money: "AFK_MONEY_ROD"
2. If asking about best rod in general (NOT for money/AFK): "BEST_ROD_GENERAL" 
3. If asking about rod config FILES/links (NOT general settings): "CONFIG_REQUEST:[ROD_NAME]" (extract the rod name)
4. If asking about rod config FILES but no specific rod mentioned: "CONFIG_REQUEST:GENERAL"
5. If asking how to get/obtain a specific rod: "ROD_OBTAIN:[ROD_NAME]" (extract the rod name)
6. If asking how to get rods in general: "ROD_OBTAIN:GENERAL"
7. If asking if macros EXIST/are available for a rod: "MACRO_AVAILABILITY:[ROD_NAME]" (extract rod name)
8. If asking about macro not working/technical problems: "MACRO_TROUBLESHOOT:[ROD_NAME]" (extract rod name if mentioned, otherwise "GENERAL")
9. If asking about enchantments (lucky, control, hasty, etc.): "NO_MATCH"
10. If none of the above: "NO_MATCH"

Message: "{message_content}"

Examples:
- "best rod to afk money" → AFK_MONEY_ROD
- "best rod to afk farm money" → AFK_MONEY_ROD  
- "what is the best rod" → BEST_ROD_GENERAL
- "anyone have ruinous oath config" → CONFIG_REQUEST:Ruinous Oath
- "where is trident config" → CONFIG_REQUEST:Trident
- "where is config" → CONFIG_REQUEST:GENERAL
- "how to get polaris" → ROD_OBTAIN:Polaris
- "how to get the no life rod" → ROD_OBTAIN:No Life
- "how to get oscar rod" → ROD_OBTAIN:Oscar
- "are there any boom ball rod macros" → MACRO_AVAILABILITY:Boom Ball
- "do you have trident macros" → MACRO_AVAILABILITY:Trident
- "any macros for seraphic rod" → MACRO_AVAILABILITY:Seraphic
- "macro not working" → MACRO_TROUBLESHOOT:GENERAL
- "why is my macro not working" → MACRO_TROUBLESHOOT:GENERAL
- "macro buttons not clickable" → MACRO_TROUBLESHOOT:GENERAL
- "macro not working with trident" → MACRO_TROUBLESHOOT:Trident
- "should I sacrifice lucky for control enchant" → NO_MATCH
- "what enchant should I use on rotd" → NO_MATCH
- "lucky vs control enchant" → NO_MATCH"""

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Bloom Discord Bot",
            },
            data=json.dumps({
                "model": "x-ai/grok-4-fast:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": intent_prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.3,
            }),
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and result["choices"][0].get("message", {}).get("content"):
                intent = result["choices"][0]["message"]["content"].strip()

            # Handle AFK money rod questions
            if intent == "AFK_MONEY_ROD":
                return """**Best AFK Rod for Money Farming:**

Honestly, it depends on what you're aiming for:

**Best AFK Rod:** Polaris Rod (overall the best for AFK fishing)

**For XP:** Oscar Rod is the way to go

**Good AFK spots:**
• Castaway Cliffs (with Aurora active) – great all-around
• Volcanic Vents (Daytime) – best for money  
• Crystal Cove – solid for XP"""

            # Handle config requests
            elif intent.startswith("CONFIG_REQUEST:"):
                rod_name = intent.split(":", 1)[1]
                if rod_name == "GENERAL":
                    return "**Fisch Rod Configs:** https://discord.com/channels/1341949236471926804/1411335491457913014"
                else:
                    return f"**The {rod_name} config should be here:** https://discord.com/channels/1341949236471926804/1411335491457913014\n\nIf it's not there, it hasn't been created yet."

            # Handle rod obtain requests
            elif intent.startswith("ROD_OBTAIN:"):
                rod_name = intent.split(":", 1)[1].lower()

                if rod_name == "general":
                    return """**How to Obtain Fisch Rods:**

**No Life Rod:** Available at level 500 as a level-requirement rod

**Seraphic Rod:** Available at level 1000 as a level-requirement rod

**Oscar Rod:** Complete a quest from an NPC in Forsaken Shores and pay 5 million C$. Perfect for XP grinding with Clever enchantment

**Polaris Rod:** No longer obtainable - was a limited-time developer rod that required trading specific items to an NPC"""

                elif "no life" in rod_name:
                    return "**No Life Rod:** This is a level-requirement rod that becomes available when you reach the level cap of 500."

                elif "seraphic" in rod_name:
                    return "**Seraphic Rod:** This is a level-requirement rod that becomes available when you reach the level cap of 1000."

                elif "oscar" in rod_name:
                    return "**Oscar Rod:** Obtain this rod by completing a quest from an NPC located in Forsaken Shores. It costs 5 million C$ and is excellent for XP grinding due to the Clever enchantment doubling XP gains."

                elif "polaris" in rod_name:
                    return "**Polaris Rod:** This rod is no longer obtainable. It was a limited-time developer rod that was previously acquired by trading specific items to an NPC."

                else:
                    return None  # Don't respond if we don't have specific info about this rod

            # Handle macro availability questions
            elif intent.startswith("MACRO_AVAILABILITY:"):
                rod_name = intent.split(":", 1)[1].lower()
                return f"**{rod_name.title()} Macros:** I don't have info on specific macro availability. Check the macro channels or community resources for the latest {rod_name} macros!"

            # Handle macro troubleshooting
            elif intent.startswith("MACRO_TROUBLESHOOT:"):
                rod_name = intent.split(":", 1)[1]

                if not client:
                    return "**Macro Issues:** Check your settings, make sure you're using the latest macro version, and verify you're using Web Roblox (not Microsoft Store version)."

                try:
                    # Generate AI response for macro troubleshooting
                    if rod_name.lower() == "general":
                        troubleshoot_prompt = """Generate a short, helpful response for macro troubleshooting in 1-2 sentences. Be concise and direct.

User is having macro issues. Suggest:
- Try restarting the macro
- Using latest macro version
- Use Web Roblox (not Microsoft Store)
- Basic troubleshooting steps

Keep it under 100 characters and casual/friendly tone. DO NOT mention configs or settings files."""
                    else:
                        troubleshoot_prompt = f"""Generate a short, helpful response for macro troubleshooting with the {rod_name} rod in 1-2 sentences. Be concise and direct.

User is having macro issues with {rod_name} rod. Suggest:
- Try restarting the macro
- Check if using latest macro version  
- Make sure you're using Web Roblox (not Microsoft Store)
- Basic troubleshooting steps

Keep it under 100 characters and casual/friendly tone. DO NOT mention configs or settings files."""

                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://replit.com",
                            "X-Title": "Bloom Discord Bot",
                        },
                        data=json.dumps({
                            "model": "x-ai/grok-4-fast:free",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": troubleshoot_prompt
                                        }
                                    ]
                                }
                            ],
                            "max_tokens": 200,
                            "temperature": 0.7,
                        }),
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("choices") and result["choices"][0].get("message", {}).get("content"):
                            return result["choices"][0]["message"]["content"].strip()
                    else:
                        # Fallback response
                        if rod_name.lower() == "general":
                            return "If the macro isn't working, try tweaking the settings and make sure you're using the latest version."
                        else:
                            return f"If the {rod_name} macro isn't working, try restarting it and make sure you're using the latest macro version."

                except Exception as e:
                    logger.error(f"Macro troubleshoot AI error: {e}")
                    # Fallback response
                    if rod_name.lower() == "general":
                        return "If the macro isn't working, try tweaking the settings and make sure you're using the latest version."
                    else:
                        return f"If the {rod_name} macro isn't working, try restarting it and make sure you're using the latest macro version."

            # For best rod general, let it fall through to existing logic
            elif intent == "BEST_ROD_GENERAL":
                return None  # Let existing best rod logic handle this

    except Exception as e:
        logger.error(f"Intelligent response error: {e}")

    return None


def get_knowledge_response(message_content):
    """Get response based on new knowledge base"""
    text = message_content.lower()

    # First try intelligent AI detection for complex patterns
    ai_response = get_intelligent_response(message_content)
    if ai_response:
        return ai_response

    # Andrew joke response - only for exact "who is andrew" (check before FAQ)
    if has_exact_keyword(text, 'who is andrew'):
        # prevent abusive roasts or slurs
        if is_disallowed_context(text):
            return "I can't help with insulting someone. Ask for a friendly summary instead."

        if client:
            try:
                joke_prompt = """Roast Andrew with a brutal, Discord-mod-tier insult.  
Start with a harsh intro like "Oh Andrew, that guy..." or "Andrew? That walking..." or come up with your own idea then absolutely tear him apart.  

⚡ Rules:  
- ONE savage line only  
- Must begin with a harsh introductory phrase about Andrew  
- Be condescending, petty, and devastatingly funny  
- Keep it short, like a Discord roast  
- No emojis, no fluff  
- Make it funny not too professional in general
"""

                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://replit.com",
                        "X-Title": "Bloom Discord Bot",
                    },
                    data=json.dumps({
                        "model": "deepseek/deepseek-chat-v3.1",
                        "messages": [{"role": "user", "content": joke_prompt}],
                        "provider": {"sort": "price"},
                    }),
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("choices") and result["choices"][0].get("message", {}).get("content"):
                        return result["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.error(f"Andrew joke generation error: {e}")

        return "Oh Andrew, that guy nobody really cares about! 🤷"

    # Rushi response - fixed context
    if has_exact_keyword(text, 'who is rushi'):
        return "The owner of this discord server"

    # Werzzzy response - fixed context
    if has_exact_keyword(text, 'who is werzzzy'):
        return "The creator of this discord bot"

    # Check deterministic FAQ after all custom responses
    faq_answer = find_faq_answer(message_content)
    if faq_answer:
        return faq_answer

    # How to read response
    if has_exact_keyword(text, 'how to read'):
        harsh_jokes = [
            "*Imagine not being able to read in 2025... peak Discord behavior* 📚",
            "*This is why aliens don't visit us anymore* 🛸📚",
            "*POV: You're asking Discord how to read instead of just... reading* 🤦📚",
            "*The fact that you need to ask this question explains everything about you* 💀",
            "*Congratulations, you've reached the bottom of the intelligence pyramid* 🔺",
            "*Your ancestors are rolling in their graves knowing their bloodline led to this* ⚰️"
        ]
        return f"https://www.wikihow.com/Teach-Yourself-to-Read\n\n{random.choice(harsh_jokes)}"

    # Macro shaking issues
    if any(
            has_exact_keyword(text, pattern) for pattern in [
                'macro not shaking', 'shake not working',
                'why is my macro not shaking', 'macro shaking issue'
            ]):
        return "**Macro Not Shaking?**\nCheck if you are using AHK 1.1 - if you are, make sure you are on the latest macro version in ⁠FISCH MANGO. If both don't work, set your shake scan delay to 15 seconds. If they don't work, check if there is a blue box when you attempt to shake. If there is, click \\ or # to fix the issue."

    # Best rod questions (only for general best rod, not AFK/money specific)
    general_rod_patterns = [
        'best rod', 'what is the best rod', 'top rod', 'good rod'
    ]
    if any(
            has_exact_keyword(text, pattern)
            for pattern in general_rod_patterns):
        # Exclude if it's about AFK or money farming
        if not any(keyword in text
                   for keyword in ['afk', 'money', 'farm', 'farming']):
            return "**Best Rod in Game:**\nThe Ruinous Oath ranks on top, however it is not macroable yet. Polaris Serenade ranks second in front of Seraphic Rod. However the requirements for these are level 1000, so a cheaper alternative is the Luminescent Rod, requiring 500 levels or the No Life Rod, a free rod obtained from level 500. If you can't get any of these, go for the Evil Pitchfork (complete the Evil Mushroom King quest in Crimson Cavern) paired with herculean."

    # BloomFisch release date
    if any(
            has_exact_keyword(text, pattern) for pattern in [
                'bloomfisch', 'bloom fisch', 'when is bloomfisch',
                'bloomfisch release', 'bloom fisch coming out'
            ]):
        return "**BloomFisch Release:**\nAround next week is the approximated time, however check ⁠:newspaper:┃news for upcoming updates about the current situation."

    # Fisch wiki/bestiary
    if any(
            has_exact_keyword(text, pattern) for pattern in [
                'fisch bestiary', 'fisch wiki', 'more information about fisch',
                'fischipedia'
            ]):
        return "**Fisch Bestiary/More Information:** https://fischipedia.org/wiki/Fisch_Wiki"

    # AFK spots
    if any(
            has_exact_keyword(text, pattern) for pattern in
        ['afk spot', 'best afk', 'where to afk', 'afk location']):
        return "**Best AFK Spots:**\n**Castaway Cliffs** - Highly recommended for end-game users as the Fisch there are some of the best, providing exotics and secrets. However, this requires a rod with good luck, resilience and control.\n\n**Crystal Cove** - Highly recommended for beginners under level 500 as Fishes there are not as resilience/control based compared to Castaway Cliffs."

    # Appraiser macro
    if any(
            has_exact_keyword(text, pattern) for pattern in
        ['appraiser macro', 'appraiser bot', 'macro appraiser']):
        return "**Appraiser Macro:**\nCurrently, there is no Appraiser Macro. There will be one in works in the future."

    # Crab cage macro
    if any(
            has_exact_keyword(text, pattern)
            for pattern in ['crab cage macro', 'crab macro', 'cage macro']):
        return "**Crab Cage Macro:**\nThere is no crab cage macro yet, however Rushi has confirmed there will be one in works in BloomFisch."

    # Enhanced macro location detection with more precise matching
    macro_patterns = [
        'where can i find the fisch macro', 'where fisch macro', 'where macro',
        'where fisch', 'fisch macro location', 'macro location', 'macro fisch',
        'fisch macro link', 'get macro', 'download macro', 'find macro',
        'macro link'
    ]

    # Only trigger if it's an exact match for one of these patterns
    if any(has_exact_keyword(text, pattern) for pattern in macro_patterns):
        return "**Fisch Macro:** https://discord.com/channels/1341949236471926804/1413837110770925578/1417999310443905116"

    # Note: Config detection is now handled by AI in get_intelligent_response()

    # Mango/Fisch macro location
    if 'mango' in text and ('you know' in text or 'where' in text
                            or 'find' in text):
        return "Fisch macro: https://discord.com/channels/1341949236471926804/1413837110770925578/1417999310443905116"

    # General Issues Keywords with regex patterns
    # AHK version questions
    ahk_patterns = [
        r'\bwhat\s+(is\s+the\s+)?ahk\s+(version|ver)\b',
        r'\bwhich\s+ahk\s+(version|ver)\b',
        r'\bahk\s+(version|ver)\s+(for|to use)\b',
        r'\bautohotkey\s+(version|ver)\b', r'\bwhat\s+autohotkey\b',
        r'\bwrong\s+ahk\b', r'\bahk\s+not\s+working\b'
    ]
    if any(re.search(pattern, text) for pattern in ahk_patterns):
        return "**AutoHotkey Version:** Use AHK v1.1 (NOT v2) - v2 is not supported for the current fisch macro."

    # Settings questions
    settings_patterns = [
        r'\bwhat\s+(are\s+the\s+)?settings\b', r'\broblox\s+settings\b',
        r'\bsettings\s+(for|to\s+use)\b', r'\bwhat\s+settings\s+should\b',
        r'\bhow\s+to\s+set\s+settings\b', r'\bconfigure\s+settings\b',
        r'\bgame\s+settings\b'
    ]
    if any(re.search(pattern, text) for pattern in settings_patterns):
        return "**Roblox Settings:** Fullscreen OFF, graphics 1, dark avatar, brightness/saturation OFF, disable camera shake."

    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['roblox version', 'wrong roblox', 'microsoft roblox']):
        return "**Wrong Roblox Version:** Use Web Roblox (Chrome/Brave/etc), NOT Microsoft Store version. Microsoft Roblox will break the macro completely."

    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['bannable', 'banned', 'ban']) and has_exact_keyword(text, 'macro'):
        return "**Is the macro bannable?** NO - The macro is like an advanced autoclicker. It doesn't inject anything into the game, making it safe and saves you time on games you love."

    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['install', 'installation']):
        return "**Installation Issues:** If AHK fails to install, try uninstalling it completely and reinstalling. Check if your antivirus software is blocking the installation - some antivirus programs flag AutoHotkey as suspicious. You may need to temporarily disable real-time protection or add an exception for AHK."

    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['moved forward', 'moving forward', 'move forward']):
        return "**Being Moved Forward:** Cause = click-to-move enabled or failed catch. Fix = disable click-to-move, use better rods+bait+configs to reduce fails."

    # Enhanced macro location detection with more variations
    macro_patterns = [
        'where can i find the fisch macro', 'where fisch macro', 'where macro',
        'where fisch', 'fisch macro location', 'macro location', 'macro fisch',
        'fisch macro link', 'get macro', 'download macro', 'find macro',
        'macro link', 'wheres mango', "where's mango", 'where is mango',
        "where's the macro", 'where the macro', 'wheres the macro',
        'where can i find the macro', 'where can i get the macro',
        'how to get macro', 'link to macro', 'macro download link',
        'where to download macro', 'where to find macro'
    ]

    # Only trigger if it's an exact match for one of these patterns
    if any(has_exact_keyword(text, pattern) for pattern in macro_patterns):
        return "**Fisch Macro:** https://discord.com/channels/1341949236471926804/1413837110770925578/1417999310443905116"

    # Debugging Flow
    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['shake not working', 'shake issue', 'debug shake']):
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
                await after.edit(nick=enforced_nick,
                                 reason="Name persistence enforced")
                logger.info(
                    f"Reset {after.name}'s nickname to {enforced_nick}")
            except Exception as e:
                logger.error(f"Failed to reset nickname for {after.name}: {e}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Check for keyword responses and knowledge base in ALL channels
    if not message.content.startswith('!') and not message.content.startswith('/'):
        # Check for keyword responses first
        for keyword_name, keyword_data in keywords_data.items():
            if has_exact_keyword(message.content, keyword_data['detection_text']):
                response_parts = []
                if keyword_data['response_text']:
                    response_parts.append(keyword_data['response_text'])
                if keyword_data['img']:
                    response_parts.append(keyword_data['img'])
                if keyword_data['link']:
                    response_parts.append(keyword_data['link'])
                
                if response_parts:
                    await message.channel.send('\n'.join(response_parts))
                    await bot.process_commands(message)
                    return

        # Check knowledge base responses for all channels
        kb_response = get_knowledge_response_for_channel(message.content, message.channel.id)
        if kb_response:
            await message.channel.send(kb_response)
            await bot.process_commands(message)
            return

        # Check if it's an advanced question for general AI response
        if is_advanced_question(message.content):
            # Only respond with AI in learning channels to avoid spam
            if message.channel.id in LEARNING_CHANNEL_IDS:
                ai_response = ask_ai(message.content.strip())
                if ai_response:  # Only send if AI actually returned something
                    await message.channel.send(ai_response)
                await bot.process_commands(message)
                return

    # Learning system (restricted to learning channels only)
    if message.channel.id in LEARNING_CHANNEL_IDS:
        # If helper replies → save Q&A
        if hasattr(message.author, 'roles') and any(role.id in HELPER_ROLES for role in message.author.roles):
            if message.reference and message.reference.message_id:
                try:
                    replied_to = await message.channel.fetch_message(message.reference.message_id)
                    q = replied_to.content.strip()
                    a = message.content.strip()
                    if q and a:
                        save_qa(q, a, message.author.id, message.channel.id)
                except Exception as e:
                    logger.error(f"Failed to process helper reply: {e}")
        else:
            # Normal user asks in learning channels → check KB first
            matches = find_similar(message.content.strip())
            if matches:
                q, a, score = matches[0]
                await message.channel.send(f"💡 {a}")
            else:
                # Fallback to AI if no KB match
                ai_answer = ask_ai(message.content.strip())
                if ai_answer:  # Only send if AI actually returned something
                    await message.channel.send(ai_answer)

    await bot.process_commands(message)


# -----------------------------
# Commands
# -----------------------------
@bot.tree.command(
    name="askbloom",
    description="Ask Bloom anything (chat, math, code, news, weather)")
@app_commands.describe(
    question="Your question or message")
async def askbloom_command(interaction: discord.Interaction, question: str):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    try:
        await interaction.response.defer()
    except discord.errors.HTTPException:
        # Interaction already acknowledged, use followup instead
        pass

    try:
        user_id = interaction.user.id
        intent = detect_intent(question)
        results = None

        # Mode routing
        if intent == "NEWS":
            results = improved_multi_source_search(question)
        elif intent == "WEATHER":
            location = question.replace("weather", "").replace("forecast", "").strip()
            results = get_weather(location)
        elif intent == "CHAT":
            # Fallback to knowledge base first
            kb_resp = get_knowledge_response_for_channel(question, interaction.channel.id)
            if kb_resp:
                add_to_context(user_id, question, kb_resp)
                await interaction.followup.send(kb_resp)
                return
        # For MATH and CODE → no search, AI handles

        # Wikipedia fallback - only for substantial questions, not greetings
        if (not results or "I couldn't find" in str(results)) and wikipedia:
            # Don't search Wikipedia for short greetings or conversational phrases
            question_lower = question.lower().strip()
            greetings = ["hi", "hello", "hey", "yo", "sup", "how are you", "how's it going", 
                        "what's up", "whatsup", "good morning", "good evening", "good night"]
            
            if len(question.split()) >= 3 and not any(greeting in question_lower for greeting in greetings):
                try:
                    wiki_summary = wikipedia.summary(question, sentences=2)
                    results = f"📖 Wikipedia: {wiki_summary}"
                except Exception:
                    pass

        # Build full AI prompt
        history = get_user_history(user_id)
        prompt = f"""{BLOOM_PERSONALITY}

Conversation so far:
{history}

User Question: {question}
Relevant Info: {results if results else "N/A"}

Answer according to personality rules.
"""

        # Send to AI
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Bloom Discord Bot",
            },
            data=json.dumps({
                "model": "x-ai/grok-4-fast:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
            }),
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            if len(answer) > 1800:
                answer = answer[:1797] + "..."
        elif response.status_code == 402:
            answer = "❌ OpenRouter API payment required. Please check your OpenRouter account billing at https://openrouter.ai/account"
        elif response.status_code == 429:
            answer = "❌ Rate limited. Please try again in a moment."
        else:
            answer = f"❌ API Error: {response.status_code} - {response.text[:100]}"

        # Save to memory
        add_to_context(user_id, question, answer)

        # Send answer
        await interaction.followup.send(answer)

    except Exception as e:
        logger.error(f"AskBloom error: {e}")
        await interaction.followup.send("❌ Something went wrong. Try again!")


@bot.tree.command(name="ban", description="Ban/Unban a user (Admin only)")
@app_commands.describe(user="User to ban or unban",
                       reason="Optional reason for the action")
async def ban_command(interaction: discord.Interaction,
                      user: discord.User,
                      reason: str = None):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    try:
        # More robust ban status checking
        is_banned = False
        try:
            await interaction.guild.fetch_ban(user)
            is_banned = True
        except discord.NotFound:
            is_banned = False
        except Exception:
            # Fallback to the original method if fetch_ban fails
            banned_users = [
                ban_entry async for ban_entry in interaction.guild.bans()
            ]
            is_banned = any(ban_entry.user.id == user.id
                            for ban_entry in banned_users)

        action_reason = reason or f"Action by {interaction.user.name}"

        if is_banned:
            # User is banned, so unban them
            await interaction.guild.unban(user, reason=action_reason)
            await interaction.response.send_message(
                f"✅ **Unbanned** {user.mention}")
            # Log the unban
            try:
                await log_moderation_action('unban',
                                            interaction.user,
                                            user,
                                            reason=action_reason)
            except Exception as e:
                logger.error(f"Failed to log unban action: {e}")
        else:
            # User is not banned, so ban them (works even if user not in server)
            await interaction.guild.ban(user, reason=action_reason)
            await interaction.response.send_message(
                f"✅ **Banned** {user.mention}")
            # Log the ban
            try:
                await log_moderation_action('ban',
                                            interaction.user,
                                            user,
                                            reason=action_reason)
            except Exception as e:
                logger.error(f"Failed to log ban action: {e}")

    except Exception as e:
        await interaction.response.send_message(
            f"❌ Failed to perform action: {e}", ephemeral=True)


@bot.tree.command(name="banpurge",
                  description="Mass ban multiple users by ID (Admin only)")
@app_commands.describe(
    user_list=
    "Comma-separated list of user IDs to ban (e.g., 123456,789012,345678)",
    reason="Optional reason for the bans")
async def banpurge_command(interaction: discord.Interaction,
                           user_list: str,
                           reason: str = None):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    # Parse the user list - remove spaces and split by comma
    user_ids = [uid.strip() for uid in user_list.split(',') if uid.strip()]

    if not user_ids:
        await interaction.followup.send("❌ No valid user IDs provided.")
        return

    action_reason = reason or f"Mass ban by {interaction.user.name}"
    banned_count = 0
    failed_count = 0
    results = []

    for user_id_str in user_ids:
        try:
            # Convert to integer ID
            user_id = int(user_id_str)

            # Fetch user by ID (works even if not in server)
            try:
                user = await bot.fetch_user(user_id)
            except discord.NotFound:
                results.append(f"❌ User ID {user_id} not found")
                failed_count += 1
                continue
            except discord.HTTPException:
                results.append(f"❌ Failed to fetch user ID {user_id}")
                failed_count += 1
                continue

            # Check if already banned
            banned_users = [
                ban_entry async for ban_entry in interaction.guild.bans()
            ]
            is_banned = any(ban_entry.user.id == user.id
                            for ban_entry in banned_users)

            if is_banned:
                results.append(
                    f"⚠️ {user.display_name} (ID: {user.id}) already banned")
                continue

            # Ban the user
            await interaction.guild.ban(user, reason=action_reason)
            results.append(f"✅ Banned {user.display_name} (ID: {user.id})")
            banned_count += 1

            # Log the ban
            try:
                await log_moderation_action('ban',
                                            interaction.user,
                                            user,
                                            reason=action_reason)
            except Exception as log_error:
                logger.error(f"Failed to log ban for {user.id}: {log_error}")

        except ValueError:
            results.append(f"❌ Invalid user ID: {user_id_str}")
            failed_count += 1
        except discord.Forbidden:
            results.append(f"❌ No permission to ban user ID {user_id_str}")
            failed_count += 1
        except Exception as e:
            results.append(f"❌ Error banning {user_id_str}: {str(e)}")
            failed_count += 1

    # Send summary
    summary = f"**Mass Ban Complete**\n✅ Successfully banned: {banned_count}\n❌ Failed: {failed_count}\n\n"

    # Add detailed results (truncate if too long)
    detailed_results = "\n".join(results)
    if len(summary + detailed_results) > 2000:
        # Truncate detailed results to fit Discord's message limit
        remaining_length = 2000 - len(summary) - 20  # Leave room for "..."
        detailed_results = detailed_results[:remaining_length] + "..."

    await interaction.followup.send(summary + detailed_results)


@bot.tree.command(
    name="namepersist",
    description="Force a user to keep a specific nickname (Admin only)")
@app_commands.describe(user="User to enforce nickname on",
                       nickname="Nickname to enforce")
async def namepersist_command(interaction: discord.Interaction,
                              user: discord.Member, nickname: str):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    try:
        await user.edit(
            nick=nickname,
            reason=f"Name persistence set by {interaction.user.name}")
        persistent_names[str(user.id)] = nickname
        save_persistent_data()
        await interaction.response.send_message(
            f"✅ {user.mention} will now always have the nickname: **{nickname}**"
        )
    except Exception as e:
        await interaction.response.send_message(
            f"❌ Failed to set persistent nickname: {e}", ephemeral=True)


@bot.tree.command(name="say",
                  description="Make Bloom say something (Admin only)")
@app_commands.describe(
    words="What should Bloom say?",
    channel=
    "Optional: Channel to send message to (current channel if not specified)")
async def say_command(interaction: discord.Interaction,
                      words: str,
                      channel: discord.TextChannel = None):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    if len(words) > 2000:
        await interaction.response.send_message(
            "❌ Message too long! Keep it under 2000 characters.",
            ephemeral=True)
        return

    target_channel = channel or interaction.channel

    try:
        await target_channel.send(words)
        # Silent confirmation - only respond if sending to different channel
        if channel and channel != interaction.channel:
            await interaction.response.send_message(
                f"✅ Message sent to {channel.mention}", ephemeral=True)
        else:
            await interaction.response.send_message("✅", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(
            f"❌ Failed to send message: {e}", ephemeral=True)


@bot.tree.command(
    name="saywb",
    description="Make Bloom say something with embed (Admin only)")
@app_commands.describe(
    title="Embed title",
    text="Embed text content",
    channel=
    "Optional: Channel to send embed to (current channel if not specified)")
async def saywb_command(interaction: discord.Interaction,
                        title: str,
                        text: str,
                        channel: discord.TextChannel = None):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    if len(title) > 256:
        await interaction.response.send_message(
            "❌ Title too long! Keep it under 256 characters.", ephemeral=True)
        return

    if len(text) > 4096:
        await interaction.response.send_message(
            "❌ Text too long! Keep it under 4096 characters.", ephemeral=True)
        return

    target_channel = channel or interaction.channel

    try:
        embed = discord.Embed(title=title, description=text, color=0x00ff00)
        await target_channel.send(embed=embed)
        # Silent confirmation - only respond if sending to different channel
        if channel and channel != interaction.channel:
            await interaction.response.send_message(
                f"✅ Embed sent to {channel.mention}", ephemeral=True)
        else:
            await interaction.response.send_message("✅", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"❌ Failed to send embed: {e}",
                                                ephemeral=True)


@bot.tree.command(name="tellmeajoke",
                  description="Get a custom AI-generated joke")
@app_commands.describe(
    context="Context for the joke (e.g., 'say something bad about my name')")
async def tellmeajoke_command(interaction: discord.Interaction, context: str):
    # Check if user is authorized admin
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    if not openrouter_key:
        await interaction.response.send_message("❌ AI service not available.",
                                                ephemeral=True)
        return

    try:
        await interaction.response.defer()
    except discord.errors.HTTPException:
        # Interaction already acknowledged, use followup instead
        pass

    # Content filter
    context_lower = context.lower()
    banned_words = [
        'racist', 'racism', 'nazi', 'hitler', 'slur', 'hate speech', 'nigger',
        'faggot'
    ]

    if any(word in context_lower for word in banned_words):
        await interaction.response.send_message(
            "❌ I can't make jokes about that. Try something else!",
            ephemeral=True)
        return

    try:
        prompt = '''You are Bloom, a Discord bot with one job: generate the most savage, 
Discord-style roast possible.  

Target description: "{context}"  

⚡ Roast Rules:
1. Always use the target’s name or description directly in the roast.  
2. Tie the insult to the context if possible.  (e.g., if they’re a "Discord mod", roast the mod angle; if they’re "a dud wasting time on Discord", roast the waste).  
3. ONE roast only, but it can be 1–2 sentences (max 300 characters).  
4. Style = short, brutal, petty, funny, humiliating — like a Discord roast battle.  
5. Be clever. Avoid lazy clichés (participation trophies, basement dwellers, spirit animals) unless twisted in a fresh way.  
6. End with a fitting emoji (💀, 🤡, 🪦, 🐟, 🐌, 🗑️, 📉).  
7. Never random — the roast must be anchored in the target info.  

🔥 Roast Categories (choose 1–2 for each output):  
- **Appearance**: roast how they look, dress, or carry themselves.  
- **Skill Issues**: mock their lack of talent, competence, or constant failures.  
- **Personality**: insult their behavior, immaturity, or attitude.  
- **Habits**: call out their time-wasting, gaming, lurking, or obsession.  
- **Social Life**: roast their loneliness, lack of friends, or cringe vibes.  
- **Power Trips**: if they’re a mod/admin, tear into their fake authority.  
- **Comparisons**: compare them to pathetic objects (lag, error 404, broken mic, empty server).  
- **Existence Roast**: make it feel like the world’s worse because they exist.  
🔥 Good Examples:  
- "Andrew? Bro bans people faster than his dad banned him from family dinners 💀"  
- "Rushi the Discord mod? Flexing power in the only place he has any 🤡"  
- "Oh Maze, man’s hairline loads slower than Roblox servers 🪦"  
- "Werzzzy? The human equivalent of a progress bar stuck at 1% 📉"  
- "That dud wasting his time on Discord? Peak skill issue: grinding roles instead of a life 💀"  
- "Andrew as a mod? Bro’s like antivirus software — blocks fun but lets all the viruses through 🤡"  
- "Rushi? Walking patch notes of failed updates nobody asked for 🗑️"  
- "Maze? Guy treats social life like a side quest he never unlocked 🐌"  

Additional Quick Clapback Style:

⚡ Quick Rules:
- ALWAYS roast the exact word/phrase given back at the sender.
- Keep it short: 1 savage line, under 200 characters.
- Turn boring inputs into petty insults.
- Style: Discord clapback / one-liner roast.

🔥 Quick Examples:
Input: "Hey" → "Don't 'hey' me like you've got friends to text 💀"
Input: "Hello" → "The only 'hello' you get is from system errors 🤡"
Input: "Sup" → "Your social life's been stuck on 'sup' since 2015 📉"
Input: "Yo" → "The only 'yo' you hear is your mom calling for chores 🪦"
Input: "Bruh" → "You say 'bruh' like that's a personality 💀"
Input: "Ok" → "That 'ok' has more energy than your entire existence 🤡"
Input: "LOL" → "You type 'LOL' with the same face you cry with 🗑️"

Now craft ONE roast about the target using either style.
'''.format(context=context)
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Bloom Discord Bot",
            },
            data=json.dumps({
                "model": "x-ai/grok-4-fast:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
            })
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("choices") and result["choices"][0].get("message", {}).get("content"):
                joke = result["choices"][0]["message"]["content"].strip()
                if len(joke) > 500:
                    joke = joke[:497] + "..."
                await interaction.followup.send(joke)
            else:
                await interaction.followup.send(
                    "❌ Couldn't generate a joke. Try again!")
        elif response.status_code == 402:
            await interaction.followup.send(
                "❌ OpenRouter API payment required. Please check your OpenRouter account billing at https://openrouter.ai/account")
        elif response.status_code == 429:
            await interaction.followup.send(
                "❌ Rate limited. Please try again in a moment.")
        else:
            await interaction.followup.send(
                f"❌ API Error: {response.status_code} - {response.text[:100]}")

    except Exception as e:
        logger.error(f"Tellmeajoke error: {e}")
        try:
            await interaction.followup.send("❌ Something went wrong. Try again!")
        except:
            # If followup fails, the interaction might have already been handled
            pass


@bot.tree.command(name="whatisthisserverabout",
                  description="Learn about this Discord server")
async def whatisthisserverabout_command(interaction: discord.Interaction):
    # Check if user is authorized admin
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    server_responses = [
        "**We are Bloom.** A chill community where people team up on projects, share ideas, and get things done together.",
        "**We are Bloom.** A space for building cool stuff, managing tasks without stress, and growing as a community.",
        "**We are Bloom.** All about teamwork—collab on projects, stay organized, and have fun while making progress.",
        "**We are Bloom.** Your go-to spot for project collabs, problem-solving, and connecting with other creators.",
        "**We are Bloom.** Simple as that: work together, stay creative, and grow while building awesome things.",
        "**We are Bloom.** Focused on collaboration, creativity, and good vibes while working on projects together.",
        "**We are Bloom.** A place to learn, build, or just hang out—there’s always room for you here.",
        "**We are Bloom.** Your creative hub for sharing ideas, managing projects, and making progress with a solid crew.",
        "**We are Bloom.** Built on teamwork, problem-solving, and helping each other grow every step of the way.",
        "**We are Bloom.** Balancing productivity and fun while working on awesome projects as a community."
    ]

    random_response = random.choice(server_responses)
    await interaction.response.send_message(random_response)


@bot.tree.command(name="keyword",
                  description="Add a keyword detection trigger (Admin only)")
@app_commands.describe(detection_text="Exact text to detect",
                       name_of_keyword="Name identifier for this keyword",
                       text="Optional response text",
                       img="Optional image URL",
                       link="Optional link URL")
async def keyword_command(interaction: discord.Interaction,
                          detection_text: str,
                          name_of_keyword: str,
                          text: str = None,
                          img: str = None,
                          link: str = None):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    global keywords_data

    # Validate that at least one response is provided
    if not any([text, img, link]):
        await interaction.response.send_message(
            "❌ You must provide at least one response (text, img, or link).",
            ephemeral=True)
        return

    # Store the keyword data
    keywords_data[name_of_keyword] = {
        'detection_text': detection_text.strip(),
        'response_text': text or '',
        'img': img or '',
        'link': link or ''
    }

    save_persistent_data()

    response_parts = [
        f"✅ Keyword '{name_of_keyword}' added with detection text: '{detection_text}'"
    ]
    if text:
        response_parts.append(f"Response: {text}")
    if img:
        response_parts.append(f"Image: {img}")
    if link:
        response_parts.append(f"Link: {link}")

    await interaction.response.send_message("\n".join(response_parts),
                                            ephemeral=True)


@bot.tree.command(name="listofkeywords",
                  description="List all configured keywords (Admin only)")
async def listofkeywords_command(interaction: discord.Interaction):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    if not keywords_data:
        embed = discord.Embed(description="()", color=0x3498db)
        await interaction.response.send_message(embed=embed)
        return

    # Format as requested: (name,name,name,name,name,name,name) with no spaces
    keyword_names = list(keywords_data.keys())
    formatted_list = "(" + ",".join(keyword_names) + ")"

    embed = discord.Embed(description=formatted_list, color=0x3498db)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="deletekeywords",
                  description="Delete a keyword by name (Admin only)")
@app_commands.describe(name_of_keyword="Name of the keyword to delete")
async def deletekeywords_command(interaction: discord.Interaction,
                                 name_of_keyword: str):
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    global keywords_data

    if name_of_keyword in keywords_data:
        del keywords_data[name_of_keyword]
        save_persistent_data()
        await interaction.response.send_message(
            f"✅ Keyword '{name_of_keyword}' deleted.", ephemeral=True)
    else:
        await interaction.response.send_message(
            f"❌ Keyword '{name_of_keyword}' not found.", ephemeral=True)


@bot.tree.command(name="learn", description="Add a question-answer pair to the knowledge base (Helpers/Admins only)")
@app_commands.describe(
    question="The question that users might ask",
    answer="The answer that should be provided for this question"
)
async def learn_command(interaction: discord.Interaction, question: str, answer: str):
    # Check if user is admin or has helper role
    is_authorized = is_admin_user(interaction.user.id)
    
    if not is_authorized and hasattr(interaction.user, 'roles'):
        is_authorized = any(role.id in HELPER_ROLES for role in interaction.user.roles)
    
    if not is_authorized:
        await interaction.response.send_message(
            "❌ You don't have permission to use this command. Only administrators and helper roles can add to the knowledge base.", 
            ephemeral=True
        )
        return

    # Validate input lengths
    if len(question.strip()) < 3:
        await interaction.response.send_message(
            "❌ Question must be at least 3 characters long.", 
            ephemeral=True
        )
        return
    
    if len(answer.strip()) < 3:
        await interaction.response.send_message(
            "❌ Answer must be at least 3 characters long.", 
            ephemeral=True
        )
        return

    try:
        # Clean the inputs
        clean_question = question.strip()
        clean_answer = answer.strip()
        
        # Save to knowledge base using existing function
        save_qa(clean_question, clean_answer, interaction.user.id, interaction.channel.id)
        
        await interaction.response.send_message(
            f"✅ **Knowledge Added Successfully!**\n\n"
            f"**Question:** {clean_question}\n"
            f"**Answer:** {clean_answer[:100]}{'...' if len(clean_answer) > 100 else ''}\n\n"
            f"The AI will now be able to answer similar questions using this information.",
            ephemeral=True
        )
        
        logger.info(f"Knowledge added by {interaction.user.name} (ID: {interaction.user.id}): Q: {clean_question[:50]}... A: {clean_answer[:50]}...")
        
    except Exception as e:
        logger.error(f"Error in learn command: {e}")
        await interaction.response.send_message(
            "❌ An error occurred while adding the knowledge. Please try again.",
            ephemeral=True
        )


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
# Export function for main.py
# -----------------------------
def start_bot(token=None):
    """Start the bot with the given token or environment variable"""
    bot_token = token or discord_token
    if not bot_token:
        logger.error("TOKEN not found. Cannot start bot.")
        return

    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("🌐 Flask server started on port 5000")

    # Initialize learning system database
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    logger.info("🤖 Starting Discord bot...")

    import time
    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            bot.run(bot_token)
            break  # If successful, exit the loop
        except discord.errors.HTTPException as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                retry_count += 1
                wait_time = min(300, (2**retry_count) *
                                30)  # Exponential backoff, max 5 minutes
                logger.warning(
                    f"Rate limited. Retry {retry_count}/{max_retries} in {wait_time} seconds..."
                )
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
                logger.info(
                    f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Exiting.")
                break


# -----------------------------
# Run bot
# -----------------------------
if __name__ == "__main__":
    start_bot()
