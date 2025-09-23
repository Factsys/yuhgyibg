import os
import sys
import json
from unittest.mock import MagicMock
from datetime import datetime, timedelta
import re
import difflib
import concurrent.futures

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
        client = genai.GenerativeModel('gemini-2.0-flash')
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

    # 1. PRIORITY: Enhanced news search for current events (if NewsAPI available)
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
                    # Add timeout to news API calls
                    import signal

                    def timeout_handler(signum, frame):
                        raise TimeoutError("News API timeout")

                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(5)  # 5 second timeout

                    news_results = news_client.get_everything(
                        language='en',
                        page_size=3,
                        from_param=(datetime.now() -
                                    timedelta(days=30)).strftime('%Y-%m-%d'),
                        **strategy)

                    signal.alarm(0)  # Cancel timeout

                    if news_results['articles']:
                        for article in news_results['articles'][:2]:
                            published_date = article['publishedAt'][:10]
                            source = article['source']['name']
                            results.append(
                                f"**📰 {article['title']}** ({source}, {published_date}): {article['description']}"
                            )
                        break
                except (TimeoutError, Exception) as strategy_error:
                    signal.alarm(0)  # Cancel timeout
                    logger.error(f"News strategy error: {strategy_error}")
                    continue

        except Exception as e:
            logger.error(f"Enhanced news search error: {e}")

    # 2. DuckDuckGo search (if available) - with timeout
    if DDGS and len(results) < 3:
        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("DuckDuckGo timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(8)  # 8 second timeout for DDG

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

            signal.alarm(0)  # Cancel timeout

        except (TimeoutError, Exception) as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"DuckDuckGo search error: {e}")

    # 3. Basic web scraping fallback - with timeout
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

    # 4. Stock search (if yfinance available) - with timeout
    if yf and not is_current_event and any(
            keyword in query_lower for keyword in
        ['stock', 'price', 'shares', 'market', '$', 'nasdaq', 'dow', 'sp500']):
        try:
            words = query.upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():
                    try:
                        ticker = yf.Ticker(word)
                        # Set timeout for yfinance
                        import signal

                        def timeout_handler(signum, frame):
                            raise TimeoutError("YFinance timeout")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(3)  # 3 second timeout

                        info = ticker.info
                        current_price = info.get('currentPrice') or info.get(
                            'regularMarketPrice')

                        signal.alarm(0)  # Cancel timeout

                        if current_price:
                            company_name = info.get('shortName', word)
                            results.append(
                                f"**💹 {company_name} ({word})**: ${current_price:.2f}"
                            )
                            break
                    except (TimeoutError, Exception):
                        signal.alarm(0)  # Cancel timeout
                        continue
        except Exception as e:
            logger.error(f"Stock search error: {e}")

    # 5. Wikipedia search (if available) - with timeout
    if wikipedia and (not is_current_event or len(results) < 2):
        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Wikipedia timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(4)  # 4 second timeout

            wiki_results = wikipedia.search(query, results=1)
            if wiki_results:
                page = wikipedia.page(wiki_results[0])
                wiki_summary = wikipedia.summary(wiki_results[0], sentences=2)
                results.append(f"**📖 {page.title}**: {wiki_summary}")

            signal.alarm(0)  # Cancel timeout

        except (TimeoutError, Exception):
            signal.alarm(0)  # Cancel timeout
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
def get_knowledge_response(message_content):
    """Get response based on new knowledge base"""
    text = message_content.lower()

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

Examples:  
"Oh Andrew, that guy who peaked in kindergarten..."  
"Andrew? That walking disappointment who..."  
"Oh you mean Andrew, the human equivalent of lag..."  
"Andrew... that guy who makes everyone appreciate silence..."  
"""

                response = client.generate_content(joke_prompt)
                if response.text:
                    return response.text.strip()
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

    # Best rod questions
    if any(
            has_exact_keyword(text, pattern) for pattern in
        ['best rod', 'what is the best rod', 'top rod', 'good rod']):
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

    # Enhanced config location detection for rod configs - use exact keyword matching
    config_patterns = [
        'where can i find the config', 'where can i find the fisch config',
        'where fisch config', 'where config', 'fisch config location',
        'config location', 'rod config', 'configs for rod', 'fisch rod config',
        'macro config', 'where rod settings', 'rod configs',
        'fisch rod configs', 'find config', 'get config'
    ]

    # Only trigger if it's an exact match for one of these patterns
    if any(has_exact_keyword(text, pattern) for pattern in config_patterns):
        return "**Fisch Rod Configs:** https://discord.com/channels/1341949236471926804/1411335491457913014"

    # Mango/Fisch macro location
    if 'mango' in text and ('you know' in text or 'where' in text
                            or 'find' in text):
        return "Fisch macro: https://discord.com/channels/1341949236471926804/1413837110770925578/1417999310443905116"

    # General Issues Keywords
    if any(
            has_exact_keyword(text, keyword)
            for keyword in ['ahk', 'autohotkey', 'auto hotkey']):
        return "**AutoHotkey Version:** Use AHK v1.1 (NOT v2) - v2 is not supported for the current fisch macro."

    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['roblox version', 'wrong roblox', 'microsoft roblox']):
        return "**Wrong Roblox Version:** Use Web Roblox (Chrome/Brave/etc), NOT Microsoft Store version. Microsoft Roblox will break the macro completely."

    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['bannable', 'banned', 'ban']) and has_exact_keyword(text, 'macro'):
        return "**Is the macro bannable?** NO - The macro is like an advanced autoclicker. It doesn't inject anything into the game, making it safe and saves you time on games you love."

    if any(
            has_exact_keyword(text, keyword)
            for keyword in ['roblox settings', 'settings']):
        return "**Roblox Settings:** Fullscreen OFF, graphics 1, dark avatar, brightness/saturation OFF, disable camera shake."

    if any(
            has_exact_keyword(text, keyword)
            for keyword in ['install', 'installation']):
        return "**Installation Issues:** If AHK fails, uninstall & reinstall. Check antivirus/browser blocking IRUS."

    if any(
            has_exact_keyword(text, keyword) for keyword in
        ['moved forward', 'moving forward', 'move forward']):
        return "**Being Moved Forward:** Cause = click-to-move enabled or failed catch. Fix = disable click-to-move, use better rods+bait+configs to reduce fails."

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
    if message.author == bot.user:
        return

    # Check knowledge base responses FIRST (for priority responses like "who is andrew")
    knowledge_response = get_knowledge_response(message.content)
    if knowledge_response:
        await message.reply(knowledge_response)
        return  # Don't process other responses

    # Then check for exact keyword matches
    message_text = message.content.strip()

    for keyword_name, keyword_data in keywords_data.items():
        detection_text = keyword_data['detection_text']

        # Check for EXACT match (case-insensitive)
        if message_text.casefold() == detection_text.casefold():
            response_parts = []

            # Add response text if available
            if keyword_data['response_text']:
                response_parts.append(keyword_data['response_text'])

            # Add image if available
            if keyword_data['img']:
                response_parts.append(keyword_data['img'])

            # Add link if available
            if keyword_data['link']:
                response_parts.append(keyword_data['link'])

            # Send response if we have anything to send
            if response_parts:
                await message.reply('\n'.join(response_parts))
                return  # Don't process other responses

    # Respond to DMs or mentions
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions

    # Special responses that don't need question indicators
    special_triggers = [
        'who is andrew', 'who is rushi', 'who is werzzzy', 'how to read'
    ]

    # Check for special triggers FIRST
    has_special_trigger = any(
        has_exact_keyword(message.content, trigger)
        for trigger in special_triggers)

    # Only check if it's a general question if no special trigger was found AND it's a DM or mention
    is_question = False
    if not has_special_trigger and (is_dm or is_mentioned):
        is_question = is_advanced_question(message.content)

    # Respond if it's a DM, mention, has special trigger, or is a question
    if is_dm or is_mentioned or has_special_trigger or is_question:
        response = get_knowledge_response(message.content)
        if response:
            await message.reply(response)

    await bot.process_commands(message)


# -----------------------------
# Commands
# -----------------------------
@bot.tree.command(
    name="askbloom",
    description="Ask Bloom anything with web search for accurate info")
@app_commands.describe(
    question="Your question - I'll search the web for current information!")
async def askbloom_command(interaction: discord.Interaction, question: str):
    # Check if user is authorized admin
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    if not client:
        await interaction.response.send_message("❌ AI service not available.",
                                                ephemeral=True)
        return

    # Content filter
    question_lower = question.lower()
    banned_words = [
        'racist', 'racism', 'nazi', 'hitler', 'slur', 'hate speech', 'nigger',
        'faggot'
    ]

    if any(word in question_lower for word in banned_words):
        await interaction.response.send_message(
            "❌ I can't help with that. Ask something else!", ephemeral=True)
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
            # Add timeout wrapper for the entire search process
            async def search_with_timeout():
                try:
                    # Run search in a separate thread to avoid blocking
                    import concurrent.futures

                    def run_search():
                        return multi_source_search(question)

                    # Use ThreadPoolExecutor with timeout
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_search)
                        try:
                            search_results = future.result(
                                timeout=15)  # 15 second total timeout
                        except concurrent.futures.TimeoutError:
                            return "Search timed out. Please try a simpler question or try again later."

                    if not search_results or search_results.strip() == "":
                        return "No information found. Please try rephrasing your question."

                    prompt = f"""You are Bloom, a Discord bot assistant. Analyze the search results and provide a direct, accurate answer.

CRITICAL ANALYSIS REQUIRED:
- DOnt be biased 
- Question assumptions in the query if data contradicts them
- Identify potential biases in sources
- Offer counterpoints when evidence supports them
- Don't sugarcoat - be direct about facts even if uncomfortable
- Challenge popular misconceptions with evidence

SEARCH RESULTS:
{search_results}

USER QUESTION: {question}

Provide a substantive, evidence-based response under 1800 characters. Focus on accuracy over politeness. If sources conflict, explain why. If the question contains false assumptions, correct them directly."""

                    # Add timeout for AI generation
                    try:
                        response = client.generate_content(prompt)
                        if response.text:
                            answer = response.text.strip()
                            if len(answer) > 1800:
                                answer = answer[:1797] + "..."
                            return answer
                        else:
                            return "❌ Couldn't generate response. Try again!"
                    except Exception as ai_error:
                        logger.error(f"AI generation error: {ai_error}")
                        return "❌ AI service temporarily unavailable. Try again!"

                except Exception as search_error:
                    logger.error(f"Search error: {search_error}")
                    return "❌ Search failed. Try a simpler question or try again later."

            # Run the search with timeout
            result = await search_with_timeout()
            await interaction.followup.send(result)

    except asyncio.TimeoutError:
        logger.error("AskBloom timeout")
        await interaction.followup.send(
            "❌ Request timed out. Try a simpler question!")
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

    if not client:
        await interaction.response.send_message("❌ AI service not available.",
                                                ephemeral=True)
        return

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
        prompt = f"""You are Bloom, a Discord bot with a dark, savage sense of humor. Given the context: "{context}", craft a devastatingly funny roast.  

⚡ Rules:  
- Be ruthless, clever, and original  
- Keep it under 500 characters  
- Deliver maximum humiliation with wit, not randomness  
- End with a fitting emoji  

Now, unleash the most savage joke possible."""

        response = client.generate_content(prompt)
        if response.text:
            joke = response.text.strip()
            if len(joke) > 500:
                joke = joke[:497] + "..."
            await interaction.response.send_message(joke)
        else:
            await interaction.response.send_message(
                "❌ Couldn't generate a joke. Try again!", ephemeral=True)

    except Exception as e:
        logger.error(f"Tellmeajoke error: {e}")
        await interaction.response.send_message(
            "❌ Something went wrong. Try again!", ephemeral=True)


@bot.tree.command(name="whatisthisserverabout",
                  description="Learn about this Discord server")
async def whatisthisserverabout_command(interaction: discord.Interaction):
    # Check if user is authorized admin
    if not is_admin_user(interaction.user.id):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.", ephemeral=True)
        return

    server_responses = [
        "**This server is dedicated to collaborative project development and task management.** We focus on building innovative solutions while maintaining efficient workflow coordination. Our community is actively expanding with strategic development plans for the future.",
        "**Welcome to our professional development community.** This server serves as a central hub for project coordination and technical collaboration. We maintain a focus on systematic task completion and strategic community growth.",
        "**This community specializes in collaborative development and organized task execution.** Our server operates as a structured environment for professional networking and project advancement. We are implementing comprehensive growth strategies for sustained expansion.",
        "**Our server functions as a professional development platform.** We concentrate on systematic project management and collaborative problem-solving. The community is designed for sustained growth through strategic planning and execution.",
        "**This server operates as a structured development environment focused on collaborative task management and project coordination.** We maintain professional standards while implementing strategic growth initiatives for long-term community expansion."
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

    # Note: Flask server is handled by main.py, not here
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
