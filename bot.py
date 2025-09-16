import os
import sys
from unittest.mock import MagicMock

# Mock audioop module before importing discord to prevent ModuleNotFoundError
# This allows discord.py to import without the audioop dependency on Python 3.13+
sys.modules['audioop'] = MagicMock()

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
import requests
from bs4 import BeautifulSoup
import trafilatura

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
            "keywords": ["ahk problem", "autohotkey problem", "ahk not working", "autohotkey not working", "ahk error", "autohotkey error", "macro won't run", "can't run macro"],
            "context": "User is having issues with AHK/autohotkey. Explain they need v1.1 not v2 because the macro isn't built for latest AHK"
        }
    }
}

# Bot setup - absolutely minimal intents (no privileged intents)
intents = discord.Intents.none()  # Start with NO intents
intents.guilds = True  # Basic guild access
intents.guild_messages = True  # Receive messages
# Note: message_content intent disabled until enabled in Discord Developer Portal
# To enable smart responses, you need to enable 'Message Content Intent' at:
# https://discord.com/developers/applications/ -> Your Bot -> Bot Section -> Privileged Gateway Intents
# intents.message_content = True  # Uncomment after enabling in portal
bot = commands.Bot(command_prefix='!', intents=intents)

class LearningData:
    def __init__(self):
        self.conversations = []
        self.user_questions = {}
        self.learned_content = []  # Store learned documents
    
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
    
    def add_learned_content(self, url, title, content, user_id):
        self.learned_content.append({
            "url": url,
            "title": title,
            "content": content,
            "learned_by": user_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_learned_content_summary(self):
        """Get a summary of all learned content for AI context"""
        if not self.learned_content:
            return ""
        
        summary = "LEARNED DOCUMENTS:\n\n"
        for doc in self.learned_content[-10:]:  # Last 10 documents
            summary += f"Title: {doc['title']}\n"
            summary += f"Content Summary: {doc['content'][:500]}...\n"
            summary += f"Source: {doc['url']}\n\n"
        return summary

learning_data = LearningData()

async def analyze_user_intent(message_content):
    """Use AI to analyze user intent and find relevant knowledge"""
    if not client:
        return None, None
    
    try:
        # Create knowledge context for AI
        knowledge_context = "Here's what I know about:\n\n"
        
        # Add all Q&A knowledge
        for qa in KNOWLEDGE_BASE["questions"]:
            topics = ", ".join(qa["keywords"])
            knowledge_context += f"• {topics}: {qa['answer']}\n"
        
        # Add trigger information
        for trigger_name, trigger_data in KNOWLEDGE_BASE["triggers"].items():
            topics = ", ".join(trigger_data["keywords"])
            knowledge_context += f"• {topics}: {trigger_data['response']}\n"
        
        # Add AI trigger contexts
        for trigger_name, trigger_data in KNOWLEDGE_BASE["ai_triggers"].items():
            topics = ", ".join(trigger_data["keywords"])
            knowledge_context += f"• {topics}: {trigger_data['context']}\n"
        
        # Add learned content
        learned_summary = learning_data.get_learned_content_summary()
        if learned_summary:
            knowledge_context += f"\n{learned_summary}"
        
        # AI prompt for understanding intent
        intent_prompt = f"""You are an intelligent assistant analyzing user messages for a gaming/macro community.

Knowledge Base:
{knowledge_context}

User Message: "{message_content}"

Analyze this message and respond with ONLY ONE of these formats:
1. If the message relates to something in the knowledge base: "RELEVANT: [brief description of what they're asking about]"
2. If the message is unrelated to the knowledge base: "UNRELATED"
3. If it's a greeting/casual chat: "CASUAL"

Be smart about understanding context - if someone says "where do I get the thing for fishing" they probably mean the macro, even if they don't say "macro" exactly."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=intent_prompt
        )
        
        if response.text:
            result = response.text.strip()
            if result.startswith("RELEVANT:"):
                context = result.replace("RELEVANT:", "").strip()
                return context, "ai_contextual"
            elif result.startswith("CASUAL"):
                return "casual_chat", "casual"
            else:
                return None, None
        
        return None, None
        
    except Exception as e:
        logger.error(f"Error analyzing user intent: {e}")
        return None, None

async def generate_smart_response(question, intent_context=None, response_type="ai_contextual"):
    """Generate intelligent AI response based on user intent and context"""
    if not client:
        return "I'm currently unable to process AI responses. Please try again later."
    
    try:
        # Create comprehensive knowledge base for AI
        full_knowledge = "COMMUNITY KNOWLEDGE:\n\n"
        
        # Add all knowledge with better formatting
        for qa in KNOWLEDGE_BASE["questions"]:
            topics = ", ".join(qa["keywords"])
            full_knowledge += f"Topic: {topics}\nInfo: {qa['answer']}\n\n"
        
        for trigger_name, trigger_data in KNOWLEDGE_BASE["triggers"].items():
            topics = ", ".join(trigger_data["keywords"])
            full_knowledge += f"Topic: {topics}\nInfo: {trigger_data['response']}\n\n"
        
        for trigger_name, trigger_data in KNOWLEDGE_BASE["ai_triggers"].items():
            topics = ", ".join(trigger_data["keywords"])
            full_knowledge += f"Topic: {topics}\nContext: {trigger_data['context']}\n\n"
        
        # Add learned content to knowledge base
        learned_summary = learning_data.get_learned_content_summary()
        if learned_summary:
            full_knowledge += f"\n{learned_summary}"
        
        if response_type == "casual":
            prompt = f"""You are Bloom, a friendly Discord bot for a gaming/macro community. The user sent a casual message: "{question}"

Respond in a friendly, natural way. Keep it conversational and welcoming. You can mention that you're here to help with macros, configs, or any questions they might have."""
        
        else:
            prompt = f"""You are Bloom, an intelligent Discord bot assistant that can help with anything!

{full_knowledge}

User asked: "{question}"
Intent analysis suggests they're asking about: {intent_context or 'general topic'}

Instructions:
- You can help with ANY topic, not just gaming/macros - be a general assistant!
- Use your knowledge base when relevant, but also use your general AI knowledge
- Understand what they REALLY mean, not just literal words
- Explain things clearly and naturally - don't just copy/paste answers
- If their question relates to multiple topics, address them all
- If you're not sure exactly what they need, ask clarifying questions
- Be helpful and friendly
- Draw from both your learned content AND general knowledge to give comprehensive answers

Generate a smart, helpful response:"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text if response.text else "I'm having trouble processing that right now. Can you try asking again?"
    
    except Exception as e:
        logger.error(f"Error generating smart response: {e}")
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

# Note: on_member_join disabled due to privileged intents requirement
# To enable this feature, you need to enable 'Server Members Intent' 
# in Discord Developer Portal under your bot's application settings

# @bot.event
# async def on_member_join(member):
#     """Auto-kick users with accounts less than 7 days old"""
#     try:
#         account_age = discord.utils.utcnow() - member.created_at
#         
#         if account_age < timedelta(days=7):
#             try:
#                 await member.send("You cannot join this server as your account is less than 7 days old")
#             except:
#                 pass  # User might have DMs disabled
#             
#             await member.kick(reason="Account less than 7 days old")
#             logger.info(f"Kicked {member.name} - account age: {account_age.days} days")
#     
#     except Exception as e:
#         logger.error(f"Error in on_member_join: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Use AI to understand what the user is really asking about
    intent_context, response_type = await analyze_user_intent(message.content)
    
    if intent_context and response_type:
        # Generate intelligent response based on understanding
        smart_response = await generate_smart_response(message.content, intent_context, response_type)
        await message.reply(smart_response)
        
        # Add learning data
        learning_data.add_conversation(message.author.id, message.content, smart_response)
    
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

@bot.tree.command(name="askbloom", description="Ask Bloom AI a question - I'll understand what you mean!")
@app_commands.describe(context="Your question - I'll figure out what you're asking about!")
async def askbloom_command(interaction: discord.Interaction, context: str):
    await interaction.response.defer()
    
    try:
        # Use AI to understand the question deeply
        intent_context, response_type = await analyze_user_intent(context)
        
        # Generate intelligent response
        if not intent_context:
            intent_context = "general question"
            response_type = "ai_contextual"
        
        ai_response = await generate_smart_response(context, intent_context, response_type)
        
        # Add to learning data
        learning_data.add_conversation(interaction.user.id, context, ai_response)
        
        # Create enhanced embed for response
        embed = discord.Embed(
            title="🧠 Bloom Smart Response",
            description=ai_response,
            color=discord.Color.purple()
        )
        embed.add_field(
            name="💭 Understanding", 
            value=f"I interpreted this as: {intent_context}", 
            inline=False
        )
        embed.set_footer(text=f"Asked by {interaction.user.display_name}")
        
        await interaction.followup.send(embed=embed)
    
    except Exception as e:
        logger.error(f"Error in smart askbloom command: {e}")
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
        logger.error(f"Bot error: {e}")
