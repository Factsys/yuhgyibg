import os
import discord
from discord.ext import commands
from discord import app_commands
import asyncio
import logging
import json
from datetime import datetime, timedelta
import re
from google import genai
from google.genai import types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required environment variables
discord_token = os.environ.get("TOKEN")
gemini_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyAO4jV2tNyVJ1-Hagvuqg9lpUd6kF5KkRs"

if not discord_token:
    logger.warning("⚠️ Warning: TOKEN environment variable not set!")
    logger.warning("Please set your Discord bot token using the secrets manager to run the bot.")

if not gemini_key:
    logger.warning("⚠️ Warning: GEMINI_API_KEY environment variable not set!")
    logger.warning("Please set your Gemini API key using the secrets manager for AI features.")

# Initialize Gemini client (conditional)
client = None
if gemini_key:
    try:
        client = genai.Client(api_key=gemini_key)
        logger.info("✅ Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")
else:
    logger.warning("⚠️ Gemini client not initialized - AI features will be disabled")

# Knowledge Base - Q&A pairs and responses
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
        },
        {
            "keywords": ["duskwire config", "duskwire setup"],
            "answer": "there is no duskwire config, please, get the serpent and string yourself as setting up duskwire with a macro is not easy."
        },
        {
            "keywords": ["configs", "loading problem", "extract file"],
            "answer": "Right click on the name and there should be an option to extract the file"
        },
        {
            "keywords": ["polaris config", "polaris macro"],
            "answer": "Polaris is not compatible yet, dm me for the custom one (credits to GHOST). THERE IS ALSO ONE IN <#1411335491457913014>"
        },
        {
            "keywords": ["get help", "need help", "support"],
            "answer": "Dm the helpers (purple tag). Also you can dm me i will try to respond as fast as i can but i might be sleeping"
        },
        {
            "keywords": ["macro bugging", "macro broken", "macro not working"],
            "answer": "Did you download v1.1 and v2 autohotkey"
        },
        {
            "keywords": ["casting", "shaking", "rod problem"],
            "answer": "use click and not nav"
        },
        {
            "keywords": ["mango broken", "opera gx"],
            "answer": "Your browser is opera gx some funny stuff happens to it"
        },
        {
            "keywords": ["bar minigame", "minigame not working"],
            "answer": "The mangos aren't compatible with dusk and wing, dm me for the custom one for polaris (credits to GHOST)"
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
    },
    "ai_triggers": {
        "ahk_version": {
            "keywords": ["ahk problem", "autohotkey problem", "ahk not working", "autohotkey not working", "ahk error", "autohotkey error", "macro won't run", "can't run macro"]
        }
    }
}

# Bot setup
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

class LearningData:
    def __init__(self):
        self.conversations = []
        self.user_questions = {}
    
    def add_conversation(self, user_id, question, answer):
        self.conversations.append({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        if user_id not in self.user_questions:
            self.user_questions[user_id] = []
        self.user_questions[user_id].append(question)

learning_data = LearningData()

def find_best_answer_qa_only(message_content):
    """Find the best answer from Q&A knowledge base only (excludes triggers)"""
    message_lower = message_content.lower()
    
    # Only check Q&A pairs - triggers are handled separately for exact responses
    best_match = None
    highest_score = 0
    
    for qa in KNOWLEDGE_BASE["questions"]:
        score = 0
        for keyword in qa["keywords"]:
            if keyword in message_lower:
                score += len(keyword.split())
        
        if score > highest_score:
            highest_score = score
            best_match = qa["answer"]
    
    return best_match

async def generate_ai_response(question, context=""):
    """Generate AI response using Gemini API"""
    # Check if Gemini client is available
    if not client:
        return "AI features are currently unavailable. Please check your Gemini API key configuration."
    
    try:
        # Create a prompt that includes the knowledge base context
        knowledge_context = ""
        for qa in KNOWLEDGE_BASE["questions"]:
            knowledge_context += f"Q: {', '.join(qa['keywords'])}\nA: {qa['answer']}\n\n"
        
        prompt = f"""You are Bloom, a helpful Discord bot assistant for a gaming community focused on macros and automation tools. 

Here's what you know:
{knowledge_context}

User question: {question}
Additional context: {context}

Respond naturally and helpfully. If the question relates to something in your knowledge base, provide that information but rephrase it in your own words to sound natural. If you don't know something specific, be honest about it but try to be helpful anyway."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        return response.text if response.text else "I'm having trouble processing that right now. Can you try asking again?"
    
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I'm experiencing some technical difficulties. Please try again later."

@bot.event
async def on_ready():
    if bot.user:
        logger.info(f'✅ {bot.user.name} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guild(s)')
    
    try:
        synced = await bot.tree.sync()
        logger.info(f"✅ Synced {len(synced)} slash command(s)")
    except Exception as e:
        logger.error(f"❌ Failed to sync commands: {e}")

@bot.event
async def on_member_join(member):
    """Auto-kick users with accounts less than 7 days old"""
    try:
        # Use discord.utils.utcnow() for proper timezone-aware comparison
        current_time = discord.utils.utcnow()
        account_age = current_time - member.created_at
        
        if account_age < timedelta(days=7):
            try:
                await member.send("You cannot join this server as your account is less than 7 days old")
            except:
                pass  # User might have DMs disabled
            
            await member.kick(reason="Account less than 7 days old")
            logger.info(f"Kicked {member.name} - account age: {account_age.days} days")
    
    except Exception as e:
        logger.error(f"Error in on_member_join: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    message_lower = message.content.lower()
    
    # Check for exact trigger matches first - return EXACT text without AI
    for trigger_name, trigger_data in KNOWLEDGE_BASE["triggers"].items():
        for keyword in trigger_data["keywords"]:
            if keyword in message_lower:
                # Return exact trigger response without AI modification
                await message.reply(trigger_data["response"])
                learning_data.add_conversation(message.author.id, message.content, trigger_data["response"])
                await bot.process_commands(message)
                return
    
    # Check for AI triggers that need generated responses
    for trigger_name, trigger_data in KNOWLEDGE_BASE.get("ai_triggers", {}).items():
        for keyword in trigger_data["keywords"]:
            if keyword in message_lower:
                if trigger_name == "ahk_version":
                    # Generate AI response about AHK version issue
                    ai_response = await generate_ai_response(message.content, 
                        "The user is having problems with AutoHotkey or running the macro. They likely need to use AHK v1.1 instead of v2 because the macro isn't built for the latest AHK version. Explain this naturally.")
                    learning_data.add_conversation(message.author.id, message.content, ai_response)
                    await message.reply(ai_response)
                    await bot.process_commands(message)
                    return
    
    # Check if message matches knowledge base Q&A (use AI for these)
    best_answer = find_best_answer_qa_only(message.content)
    
    if best_answer:
        # Generate AI response based on the knowledge
        ai_response = await generate_ai_response(message.content, best_answer)
        
        # Add learning data
        learning_data.add_conversation(message.author.id, message.content, ai_response)
        
        await message.reply(ai_response)
    
    # Process other commands
    await bot.process_commands(message)

# Slash Commands
@bot.tree.command(name="say", description="Make the bot say something")
@app_commands.describe(words="The words you want the bot to say")
async def say_command(interaction: discord.Interaction, words: str):
    await interaction.response.send_message(words)

@bot.tree.command(name="saywb", description="Send an embedded message")
@app_commands.describe(
    words="The main message content",
    title="Optional title for the embed",
    description="Optional description for the embed"
)
async def saywb_command(interaction: discord.Interaction, words: str, title: str = "", description: str = ""):
    embed = discord.Embed(
        title=title if title else "Message",
        description=description if description else words,
        color=discord.Color.blue()
    )
    
    if title and description:
        embed.add_field(name="Content", value=words, inline=False)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="askbloom", description="Ask Bloom AI a question with context")
@app_commands.describe(context="Your question or context for Bloom to respond to")
async def askbloom_command(interaction: discord.Interaction, context: str):
    await interaction.response.defer()
    
    try:
        # Check if question matches knowledge base
        best_answer = find_best_answer_qa_only(context)
        
        if best_answer:
            # Generate AI response with knowledge base context
            ai_response = await generate_ai_response(context, best_answer)
        else:
            # Generate general AI response
            ai_response = await generate_ai_response(context)
        
        # Add to learning data
        learning_data.add_conversation(interaction.user.id, context, ai_response)
        
        # Create embed for response
        embed = discord.Embed(
            title="🤖 Bloom AI Response",
            description=ai_response,
            color=discord.Color.green()
        )
        embed.set_footer(text=f"Asked by {interaction.user.display_name}")
        
        await interaction.followup.send(embed=embed)
    
    except Exception as e:
        logger.error(f"Error in askbloom command: {e}")
        await interaction.followup.send("I'm having trouble processing your request. Please try again later.")

# Error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    
    logger.error(f"Command error: {error}")
    await ctx.send("An error occurred while processing your command.")

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    logger.error(f"App command error: {error}")
    if not interaction.response.is_done():
        await interaction.response.send_message("An error occurred while processing your command.", ephemeral=True)

# Run the bot
if __name__ == "__main__":
    if not discord_token:
        logger.error("TOKEN not found in environment variables")
        logger.error("Cannot start bot without Discord token. Please set TOKEN in secrets.")
        exit(1)
    
    try:
        bot.run(discord_token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")bot = commands.Bot(command_prefix='!', intents=intents)

class LearningData:
    def __init__(self):
        self.conversations = []
        self.user_questions = {}
    
    def add_conversation(self, user_id, question, answer):
        self.conversations.append({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        if user_id not in self.user_questions:
            self.user_questions[user_id] = []
        self.user_questions[user_id].append(question)

learning_data = LearningData()

def find_best_answer_qa_only(message_content):
    """Find the best answer from Q&A knowledge base only (excludes triggers)"""
    message_lower = message_content.lower()
    
    # Only check Q&A pairs - triggers are handled separately for exact responses
    best_match = None
    highest_score = 0
    
    for qa in KNOWLEDGE_BASE["questions"]:
        score = 0
        for keyword in qa["keywords"]:
            if keyword in message_lower:
                score += len(keyword.split())
        
        if score > highest_score:
            highest_score = score
            best_match = qa["answer"]
    
    return best_match

async def generate_ai_response(question, context=""):
    """Generate AI response using Gemini API"""
    # Check if Gemini client is available
    if not client:
        return "AI features are currently unavailable. Please check your Gemini API key configuration."
    
    try:
        # Create a prompt that includes the knowledge base context
        knowledge_context = ""
        for qa in KNOWLEDGE_BASE["questions"]:
            knowledge_context += f"Q: {', '.join(qa['keywords'])}\nA: {qa['answer']}\n\n"
        
        prompt = f"""You are Bloom, a helpful Discord bot assistant for a gaming community focused on macros and automation tools. 

Here's what you know:
{knowledge_context}

User question: {question}
Additional context: {context}

Respond naturally and helpfully. If the question relates to something in your knowledge base, provide that information but rephrase it in your own words to sound natural. If you don't know something specific, be honest about it but try to be helpful anyway."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        return response.text if response.text else "I'm having trouble processing that right now. Can you try asking again?"
    
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I'm experiencing some technical difficulties. Please try again later."

@bot.event
async def on_ready():
    if bot.user:
        logger.info(f'✅ {bot.user.name} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guild(s)')
    
    try:
        synced = await bot.tree.sync()
        logger.info(f"✅ Synced {len(synced)} slash command(s)")
    except Exception as e:
        logger.error(f"❌ Failed to sync commands: {e}")

@bot.event
async def on_member_join(member):
    """Auto-kick users with accounts less than 7 days old"""
    try:
        # Use discord.utils.utcnow() for proper timezone-aware comparison
        current_time = discord.utils.utcnow()
        account_age = current_time - member.created_at
        
        if account_age < timedelta(days=7):
            try:
                await member.send("You cannot join this server as your account is less than 7 days old")
            except:
                pass  # User might have DMs disabled
            
            await member.kick(reason="Account less than 7 days old")
            logger.info(f"Kicked {member.name} - account age: {account_age.days} days")
    
    except Exception as e:
        logger.error(f"Error in on_member_join: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    message_lower = message.content.lower()
    
    # Check for exact trigger matches first - return EXACT text without AI
    for trigger_name, trigger_data in KNOWLEDGE_BASE["triggers"].items():
        for keyword in trigger_data["keywords"]:
            if keyword in message_lower:
                # Return exact trigger response without AI modification
                await message.reply(trigger_data["response"])
                learning_data.add_conversation(message.author.id, message.content, trigger_data["response"])
                await bot.process_commands(message)
                return
    
    # Check for AI triggers that need generated responses
    for trigger_name, trigger_data in KNOWLEDGE_BASE.get("ai_triggers", {}).items():
        for keyword in trigger_data["keywords"]:
            if keyword in message_lower:
                if trigger_name == "ahk_version":
                    # Generate AI response about AHK version issue
                    ai_response = await generate_ai_response(message.content, 
                        "The user is having problems with AutoHotkey or running the macro. They likely need to use AHK v1.1 instead of v2 because the macro isn't built for the latest AHK version. Explain this naturally.")
                    learning_data.add_conversation(message.author.id, message.content, ai_response)
                    await message.reply(ai_response)
                    await bot.process_commands(message)
                    return
    
    # Check if message matches knowledge base Q&A (use AI for these)
    best_answer = find_best_answer_qa_only(message.content)
    
    if best_answer:
        # Generate AI response based on the knowledge
        ai_response = await generate_ai_response(message.content, best_answer)
        
        # Add learning data
        learning_data.add_conversation(message.author.id, message.content, ai_response)
        
        await message.reply(ai_response)
    
    # Process other commands
    await bot.process_commands(message)

# Slash Commands
@bot.tree.command(name="say", description="Make the bot say something")
@app_commands.describe(words="The words you want the bot to say")
async def say_command(interaction: discord.Interaction, words: str):
    await interaction.response.send_message(words)

@bot.tree.command(name="saywb", description="Send an embedded message")
@app_commands.describe(
    words="The main message content",
    title="Optional title for the embed",
    description="Optional description for the embed"
)
async def saywb_command(interaction: discord.Interaction, words: str, title: str = "", description: str = ""):
    embed = discord.Embed(
        title=title if title else "Message",
        description=description if description else words,
        color=discord.Color.blue()
    )
    
    if title and description:
        embed.add_field(name="Content", value=words, inline=False)
    
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="askbloom", description="Ask Bloom AI a question with context")
@app_commands.describe(context="Your question or context for Bloom to respond to")
async def askbloom_command(interaction: discord.Interaction, context: str):
    await interaction.response.defer()
    
    try:
        # Check if question matches knowledge base
        best_answer = find_best_answer_qa_only(context)
        
        if best_answer:
            # Generate AI response with knowledge base context
            ai_response = await generate_ai_response(context, best_answer)
        else:
            # Generate general AI response
            ai_response = await generate_ai_response(context)
        
        # Add to learning data
        learning_data.add_conversation(interaction.user.id, context, ai_response)
        
        # Create embed for response
        embed = discord.Embed(
            title="🤖 Bloom AI Response",
            description=ai_response,
            color=discord.Color.green()
        )
        embed.set_footer(text=f"Asked by {interaction.user.display_name}")
        
        await interaction.followup.send(embed=embed)
    
    except Exception as e:
        logger.error(f"Error in askbloom command: {e}")
        await interaction.followup.send("I'm having trouble processing your request. Please try again later.")

# Error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    
    logger.error(f"Command error: {error}")
    await ctx.send("An error occurred while processing your command.")

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
