import discord
import json
import re
import io
import os
import aiohttp
import aiofiles
import time
import sys # Added for exiting on missing token
import asyncio
import random
import traceback
import argparse
from googletrans import Translator
# Removed duplicate json import

from tools import gen_image, img2txt, get_response, img2location

# 全局变量
is_generating_response = False
response_lock = asyncio.Lock()

# --- Configuration for Render ---
# Assume persistent disk is mounted at /data
DATA_DIR = "/data"
CONFIG_STATE_FILE = os.path.join(DATA_DIR, "config_state.json")
DATE_FILE = os.path.join(DATA_DIR, "date.json")
SAID_WORDS_JSON_FILE = os.path.join(DATA_DIR, "said_words.json") # Note: Original code writes {} here, maybe unused?
SAID_WORDS_TXT_FILE = os.path.join(DATA_DIR, "said_words.txt")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
# --- End Configuration ---


# save the current date to DATE_FILE if it doesn't exist or needs update
def update_date():
    try:
        if not os.path.exists(DATE_FILE):
            with open(DATE_FILE, "w") as f:
                json.dump({"date": time.time()}, f)
            print(f"{DATE_FILE} created")
            # Also ensure said_words files are initialized if date file is new
            if not os.path.exists(SAID_WORDS_JSON_FILE):
                 with open(SAID_WORDS_JSON_FILE, "w") as f: json.dump({}, f)
            if not os.path.exists(SAID_WORDS_TXT_FILE):
                 with open(SAID_WORDS_TXT_FILE, "w") as f: f.write("")

        else:
            with open(DATE_FILE, "r") as f:
                content = f.read()
                if content:
                    date_data = json.loads(content)
                    # print(f"Current date data: {date_data}") # Commented out for less verbose logs
                    # Check if 3 days (259200 seconds) have passed
                    if time.time() - date_data.get("date", 0) > 259200:
                        with open(DATE_FILE, "w") as f:
                            json.dump({"date": time.time()}, f)
                        # Clear said words files
                        with open(SAID_WORDS_JSON_FILE, "w") as f:
                            json.dump({}, f)
                        with open(SAID_WORDS_TXT_FILE, "w") as f:
                            f.write("")
                        print("Date updated and said words cleared")
                else:
                    # File exists but is empty, initialize it
                     with open(DATE_FILE, "w") as f: json.dump({"date": time.time()}, f)
                     print(f"{DATE_FILE} was empty, initialized.")
                     if not os.path.exists(SAID_WORDS_JSON_FILE):
                          with open(SAID_WORDS_JSON_FILE, "w") as f: json.dump({}, f)
                     if not os.path.exists(SAID_WORDS_TXT_FILE):
                          with open(SAID_WORDS_TXT_FILE, "w") as f: f.write("")

    except Exception as e:
        print(f"Error in update_date: {e}")


async def send_webhook_message(content):
    if not info.get("webhook_url"): # Check if webhook_url is configured
        print("Webhook URL not configured. Skipping message.")
        return
    data = {"content": content}
    headers = {"Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(info["webhook_url"], json=data, headers=headers)
    except Exception as e:
        print(f"Error sending webhook message: {e}")


async def get_name(user):
    try:
        # Check if user object is valid and has attributes
        if user and hasattr(user, 'nick') and user.nick:
            return user.nick
        if user and hasattr(user, 'display_name'):
            return user.display_name
    except Exception as e:
        print(f"Error getting nickname for user {user}: {e}")
        pass
    # Fallback name if user object is problematic or has no suitable name
    return "Unknown User"


async def check_ignored_words(text, ignored_words):
    text_lower = text.lower()
    text_split = text_lower.split()
    for word in text_split:
        # Ensure ignored_words list contains strings
        if isinstance(ignored_words, list):
            if word in ignored_words:
                return True
    return False


async def write_json(data, file_path): # Takes full path now
    try:
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(data, indent=2)) # Added indent for readability
    except Exception as e:
        print(f"Error writing JSON to {file_path}: {e}")


async def read_json(file_path): # Takes full path now
    try:
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
        # Handle empty file case
        if not content:
            print(f"JSON file is empty: {file_path}")
            return None
        return json.loads(content)
    except FileNotFoundError:
        print(f"JSON file not found: {file_path}")
        return None # Return None if file not found
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return None # Return None on decode error
    except Exception as e:
        print(f"Error reading JSON from {file_path}: {e}")
        return None


async def split_response(response, max_length=1900):
    lines = response.splitlines()
    chunks = []
    current_chunk = ""

    for line in lines:
        # Check if adding the next line exceeds max_length
        # Add 1 for the newline character
        if len(current_chunk) + len(line) + 1 > max_length:
            # If current_chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start a new chunk with the current line
            # Handle cases where a single line might exceed max_length
            if len(line) > max_length:
                 # Split the long line itself
                 for i in range(0, len(line), max_length):
                     chunks.append(line[i:i+max_length])
                 current_chunk = "" # Reset chunk as the long line was handled
            else:
                 current_chunk = line
        else:
            # Add line to current chunk
            if current_chunk:
                current_chunk += "\n"
            current_chunk += line

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Ensure no empty chunks are returned
    return [chunk for chunk in chunks if chunk]


async def create_meme(top_text, bottom_text, image_url):
    # Replace special characters in text
    top_text = top_text.replace(" ", "-").replace("?", "~q").replace("&", "~a").replace("%", "~p").replace("#", "~h").replace("/", "~s")
    bottom_text = bottom_text.replace(" ", "-").replace("?", "~q").replace("&", "~a").replace("%", "~p").replace("#", "~h").replace("/", "~s")

    # Create the meme URL
    meme_url = f"https://api.memegen.link/images/custom/{top_text}/{bottom_text}.jpg?background={image_url}"

    return meme_url


# Removed read_json_sync as it's no longer used for initial config load

# --- Configuration Loading ---

info = {} # Global info dictionary

def load_config_and_state():
    """Loads configuration from environment variables and persistent state file."""
    global info
    print("Loading configuration...")

    # --- Load Secrets from Environment Variables ---
    token = os.environ.get("DISCORD_TOKEN")
    webhook_url = os.environ.get("WEBHOOK_URL") # Optional
    dev_id_str = os.environ.get("DEV_ID") # Mandatory

    if not token:
        print("CRITICAL ERROR: DISCORD_TOKEN environment variable not set.")
        sys.exit(1) # Exit if token is missing
    if not dev_id_str:
        print("CRITICAL ERROR: DEV_ID environment variable not set.")
        sys.exit(1) # Exit if dev_id is missing

    try:
        dev_id = int(dev_id_str)
    except ValueError:
        print(f"CRITICAL ERROR: DEV_ID environment variable ('{dev_id_str}') is not a valid integer.")
        sys.exit(1)

    # --- Load Mutable State/Config from Persistent File ---
    config_state = {}
    default_config_state = {
        "system_prompt": "You are a helpful Discord bot.",
        "temperature": 0.7,
        "type": "openai", # Example default, adjust as needed
        "model": "gpt-3.5-turbo", # Example default
        "allowed_channels": [],
        "ignored_words": ["badword1", "badword2"], # Example defaults
        "banned_users": [],
        "default_image_gen": "sd3" # Example default
    }

    try:
        if os.path.exists(CONFIG_STATE_FILE):
            with open(CONFIG_STATE_FILE, "r") as f:
                content = f.read()
                if content: # Check if file is not empty
                    config_state = json.load(f)
                else:
                    print(f"{CONFIG_STATE_FILE} is empty, using defaults.")
                    config_state = default_config_state
                    # Write defaults back to the empty file
                    with open(CONFIG_STATE_FILE, "w") as wf:
                        json.dump(config_state, wf, indent=2)

            print(f"Loaded configuration state from {CONFIG_STATE_FILE}")
        else:
            print(f"{CONFIG_STATE_FILE} not found, creating with defaults.")
            config_state = default_config_state
            with open(CONFIG_STATE_FILE, "w") as f:
                json.dump(config_state, f, indent=2)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading or writing {CONFIG_STATE_FILE}: {e}. Using defaults.")
        config_state = default_config_state
        # Attempt to rewrite the default file in case of corruption
        try:
            with open(CONFIG_STATE_FILE, "w") as f:
                json.dump(config_state, f, indent=2)
        except IOError as write_e:
             print(f"Could not write default config state file: {write_e}")


    # --- Populate global 'info' dictionary ---
    # Use .get with defaults for robustness
    info = {
        "webhook_url": webhook_url, # Can be None if not set
        "dev_id": dev_id,
        "system_prompt": config_state.get("system_prompt", default_config_state["system_prompt"]),
        "temperature": float(config_state.get("temperature", default_config_state["temperature"])),
        "type": config_state.get("type", default_config_state["type"]),
        "model": config_state.get("model", default_config_state["model"]),
        "allowed_channels": config_state.get("allowed_channels", default_config_state["allowed_channels"]),
        "ignored_words": config_state.get("ignored_words", default_config_state["ignored_words"]),
        "banned_users": config_state.get("banned_users", default_config_state["banned_users"]),
        "default_image_gen": config_state.get("default_image_gen", default_config_state["default_image_gen"]),
    }

    # Ensure list types are actually lists after loading
    for key in ["allowed_channels", "ignored_words", "banned_users"]:
        loaded_value = info.get(key)
        if not isinstance(loaded_value, list):
            print(f"Warning: '{key}' in config was not a list (type: {type(loaded_value)}), resetting to default.")
            info[key] = default_config_state[key]


    print("Configuration loaded successfully.")
    # print(f"Info dict: {info}") # Uncomment for debugging
    return token # Return the loaded token

async def save_config_state():
    """Saves the mutable parts of the 'info' dict to the persistent file."""
    keys_to_save = [
        "system_prompt", "temperature", "type", "model",
        "allowed_channels", "ignored_words", "banned_users", "default_image_gen"
    ]
    config_state_to_save = {key: info[key] for key in keys_to_save if key in info}
    print(f"Saving configuration state to {CONFIG_STATE_FILE}...")
    await write_json(config_state_to_save, CONFIG_STATE_FILE) # Use the async write_json


# --- Load Config On Startup ---
TOKEN = load_config_and_state()
# --- End Configuration Loading ---


break_string = "</s>"

last_messages = {}
conversation_history = {}
user_generating = {}


# Use default intents and enable message content
intents = discord.Intents.default()
intents.message_content = True
# Consider adding members intent if needed for get_name reliability,
# but requires enabling in Discord Dev Portal
# intents.members = True

client = discord.Client(intents=intents) # Pass intents


@client.event
async def on_message(message):
    global is_generating_response # Keep using global for this lock flag

    # Ignore messages if already processing another
    if response_lock.locked():
        return

    # Basic checks
    if message.author == client.user:
        return
    if message.author.bot: # Ignore other bots
        return
    if not message.guild: # Ignore DMs for now
        return

    # Check send permissions
    # Use guild.me to get the bot's member object in the guild
    if not message.channel.permissions_for(message.guild.me).send_messages:
        # Silently return if no permission, or log if needed
        # print(f"No send permission in {message.channel.name} ({message.guild.name})")
        return

    # Check if channel is allowed or if user is dev
    # Ensure types are comparable (message.channel.id is int, info["allowed_channels"] should be list of ints)
    is_allowed_channel = message.channel.id in info.get("allowed_channels", [])
    is_dev = message.author.id == info.get("dev_id")

    if not is_allowed_channel and not is_dev:
        return

    # Check if user is banned
    if message.author.id in info.get("banned_users", []):
        return


    # Get the current channel
    channel = message.channel

    # Get the current Guild Name
    guild_name = message.guild.name
    guild_owner_name = "Unknown" # Default
    if message.guild.owner:
        guild_owner_name = await get_name(message.guild.owner)

    # Get the user's server nickname or global name
    user_nickname = await get_name(message.author)


    # --- Command Handling ---
    if message.content.startswith("."):
        # phrases = [".imagine", ".img2prompt", ".img2txt", ".meme", ".enhanceprompt", ".say", ".analyse", ".translate", ".delete", ".change", ".settings", ".clear", ".allowchannel", ".blockchannel", ".ignoreword", ".unignoreword", ".help", ".ban", ".unban", ".openai"]
        phrases = [".say", ".delete", ".change", ".settings", ".clear", ".allowchannel", ".blockchannel", ".ignoreword", ".unignoreword", ".commands", ".ban", ".unban", ".imagine", ".translate", ".img2location", ".help", ".uwu", ".purge"]
        command_part = message.content.lower().split()[0] # Get the first word as command

        # Find matching phrase efficiently
        matched_phrase = None
        for phrase in phrases:
            if command_part == phrase:
                 matched_phrase = phrase
                 break

        if matched_phrase:
            async with message.channel.typing():
                # --- .imagine Command ---
                if matched_phrase == ".imagine":
                    if not message.channel.permissions_for(message.guild.me).attach_files:
                        print(f"Bot does not have permission to send images in the server {message.guild.name} in the channel {message.channel.name}")
                        await asyncio.sleep(round(random.uniform(1,3), 2))
                        await message.reply("Sorry, I don't have the permission to send images in this channel")
                        return

                    prompt_text = message.content[len(matched_phrase):].strip()
                    prompt_text = prompt_text.replace("—", "--")

                    parser = argparse.ArgumentParser(add_help=False) # Disable default help
                    parser.add_argument("--ar", type=str, default="1:1")
                    parser.add_argument("--exact", action="store_true")
                    parser.add_argument("--model", type=str, default=info.get("default_image_gen", "sd3"))
                    parser.add_argument("--video", action="store_true")

                    try:
                        # Parse known args first
                        args, unknown = parser.parse_known_args(prompt_text.split())
                        # The rest is the actual prompt
                        prompt = ' '.join(unknown).strip()
                    except Exception as parse_e:
                        print(f"Error parsing imagine args: {parse_e}")
                        await message.reply(f"Error parsing arguments: {parse_e}")
                        return

                    ar = args.ar
                    exact = args.exact
                    model = args.model
                    video = args.video

                    if video:
                        model = "text2video"

                    if not prompt:
                        await asyncio.sleep(round(random.uniform(1,3), 2))
                        try:
                            await message.reply('Please provide a prompt after the command and arguments.')
                        except Exception:
                            await message.channel.send(f"{message.author.mention} Please provide a prompt after the command and arguments.")
                        return

                    if model not in ["dalle3", "sd3", "text2video"]:
                        await message.reply("Invalid model specified. Use `dalle3`, `sd3`, or `text2video` (via --video).")
                        return

                    generating = user_generating.get(message.author.id, False)
                    if generating:
                        await message.reply(f"Please wait for your previous image/video to generate.")
                        return

                    if message.author.id != info['dev_id']:
                        user_generating[message.author.id] = True

                    msg = None # Initialize msg to None
                    try:
                        await asyncio.sleep(round(random.uniform(1,3), 2))
                        msg = await message.channel.send(f'Generating {"video" if video else "image"} for {message.author.mention} with prompt ```{prompt}``` (Model: {model}, AR: {ar})...')

                        start_time = time.time()
                        # Assuming gen_image handles the video flag internally based on model name
                        image_url = await gen_image(prompt=prompt, ar=ar, exact=exact, model=model)
                        end_time = time.time()
                        time_taken = round(end_time - start_time, 2)

                        if msg: await msg.delete() # Delete status message

                        files_to_send = []
                        file_extension = "gif" if video else "png"
                        file_name = f"video.{file_extension}" if video else f"image.{file_extension}"

                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url) as resp:
                                if resp.status != 200:
                                    print(f"Error fetching generated image/video: Status {resp.status}")
                                    await channel.send(f'Error fetching the generated {"video" if video else "image"}.')
                                    return # Early exit
                                data = io.BytesIO(await resp.read())
                                files_to_send.append(discord.File(data, file_name))

                        await message.channel.send(
                            content=f"**{message.author.mention}'s {'Video' if video else 'Image'}**\n\n**Prompt:** ```{prompt}```\n\n*Created in `{time_taken}` seconds*",
                            files=files_to_send
                        )

                    except Exception as e:
                        if msg: await msg.delete() # Ensure status message is deleted on error
                        print(f"Error during image/video generation or sending: {traceback.format_exc()}")
                        await message.reply(f"An error occurred: `{e}`", delete_after=10.0)
                        await send_webhook_message(
                            f"<@{info['dev_id']}> Error in .imagine command in {message.channel.name} {message.jump_url}, Prompt: `{prompt}`: \n\n**{e}**"
                        )
                    finally:
                         if message.author.id != info['dev_id']:
                             user_generating[message.author.id] = False # Ensure flag is reset
                    return # End .imagine command handling

                # --- .say Command ---
                elif matched_phrase == ".say":
                    if message.author.id != info['dev_id']: return
                    msg_to_say = message.content[len(matched_phrase):].strip()
                    if msg_to_say:
                        await channel.send(msg_to_say)
                    else:
                        await message.reply("What should I say?", delete_after=10.0)
                    return

                # --- .img2txt Command ---
                elif matched_phrase == ".img2txt":
                    if not message.attachments:
                        await message.reply("Please attach an image.", delete_after=10.0)
                        return
                    image_url = None
                    for attachment in message.attachments:
                        if attachment.content_type and attachment.content_type.startswith('image/'):
                             image_url = [attachment.url] # Needs to be a list for img2txt
                             break # Use first image found
                    if not image_url:
                         await message.reply("No valid image attachment found.", delete_after=10.0)
                         return
                    try:
                        caption = await img2txt(image_url)
                        await message.reply (f"**I see:** ```{caption}```")
                    except Exception as e:
                        print(f"Error in img2txt: {traceback.format_exc()}")
                        await message.reply("An error occurred while describing the image.", delete_after=10.0)
                    return

                # --- .translate Command ---
                elif matched_phrase == ".translate":
                    if message.reference is None or message.reference.message_id is None:
                        await message.reply("Please reply to a message to translate it.", delete_after=10.0)
                        return

                    try:
                        original_message = await channel.fetch_message(message.reference.message_id)
                        original_content = original_message.content
                        if not original_content:
                             await message.reply("The replied message has no text content to translate.", delete_after=10.0)
                             return

                        translator = Translator()
                        translated_text = translator.translate(original_content)
                        src = translated_text.src
                        dest = translated_text.dest
                        text = translated_text.text

                        if await check_ignored_words(text, info.get("ignored_words", [])):
                            msg = await message.reply(f"Error: The translated text contains an inappropriate word.")
                            await asyncio.sleep(10)
                            await msg.delete()
                            return

                        await message.reply(f"**Translated text from `{src}` to `{dest}`:** \n```{text}```")

                    except discord.NotFound:
                         await message.reply("Could not find the message you replied to.", delete_after=10.0)
                    except Exception as e:
                        print(f"Error during translation: {traceback.format_exc()}")
                        await message.reply(f"An error occurred during translation: {e}", delete_after=10.0)
                    return

                # --- .delete Command (Dev Only) ---
                elif matched_phrase == ".delete":
                    if message.author.id != info['dev_id']: return
                    if message.reference is None or message.reference.message_id is None:
                        await message.reply("Please reply to the bot message you want to delete.", delete_after=10.0)
                        return
                    try:
                        msg_to_delete = await channel.fetch_message(message.reference.message_id)
                        if msg_to_delete.author.id == client.user.id:
                            await msg_to_delete.delete()
                            await message.delete() # Delete the command message too
                        else:
                            await message.reply("You can only delete messages sent by me.", delete_after=10.0)
                    except discord.NotFound:
                        await message.reply("Could not find the message you replied to.", delete_after=10.0)
                    except discord.Forbidden:
                         await message.reply("I don't have permission to delete messages here.", delete_after=10.0)
                    except Exception as e:
                        print(f"Error deleting message: {e}")
                        await message.reply("An error occurred while trying to delete the message.", delete_after=10.0)
                    return

                # --- .change Command (Dev Only) ---
                elif matched_phrase == ".change":
                    if message.author.id != info['dev_id']: return
                    parts = message.content.split(maxsplit=2)
                    if len(parts) < 3:
                        await message.reply("Usage: `.change <key> <value>`", delete_after=10.0)
                        return

                    key = parts[1]
                    value_str = parts[2]

                    changeable_keys = ["system_prompt", "temperature", "type", "model", "default_image_gen"]

                    if key not in changeable_keys:
                        await message.reply(f"Invalid or non-changeable key '{key}'. Changeable keys: {', '.join(changeable_keys)}", delete_after=10.0)
                        return

                    value = value_str # Assign as string initially
                    if key == "temperature":
                        try:
                            value = float(value_str)
                        except ValueError:
                            await message.reply(f"Invalid value for temperature: '{value_str}'. Must be a number.", delete_after=10.0)
                            return

                    info[key] = value # Update in-memory config
                    await save_config_state() # Save updated state
                    await message.reply(f"Successfully changed '{key}' to '{value}'", delete_after=10.0)
                    return

                # --- .settings Command (Dev Only) ---
                elif matched_phrase == ".settings":
                    if message.author.id != info['dev_id']: return
                    display_info = info.copy()
                    display_info.pop("webhook_url", None) # Don't show webhook
                    try:
                        settings_str = json.dumps(display_info, indent=2)
                        # Split if too long for one message
                        chunks = await split_response(f"**Current Settings:**\n\n```{settings_str}```")
                        for i, chunk in enumerate(chunks):
                             if i == 0:
                                 await message.reply(chunk)
                             else:
                                 await message.channel.send(chunk)
                    except Exception as e:
                         await message.reply(f"Error formatting settings: {e}")
                    return

                # --- .clear Command (Dev Only) ---
                elif matched_phrase == ".clear":
                    if message.author.id != info['dev_id']: return
                    if channel.id in conversation_history:
                        conversation_history[channel.id] = []
                        await message.reply("Successfully cleared my conversation history for this channel.")
                    else:
                        await message.reply("No conversation history found for this channel.")
                    return

                # --- .allowchannel Command (Dev Only) ---
                elif matched_phrase == ".allowchannel":
                    if message.author.id != info['dev_id']: return
                    channel_id = message.channel.id
                    if channel_id in info["allowed_channels"]:
                        await message.reply("Channel already allowed.")
                        return
                    info["allowed_channels"].append(channel_id)
                    await save_config_state()
                    await message.reply(f"Successfully allowed channel {message.channel.mention} (`{channel_id}`)")
                    return

                # --- .blockchannel Command (Dev Only) ---
                elif matched_phrase == ".blockchannel":
                    if message.author.id != info['dev_id']: return
                    channel_id = message.channel.id
                    if channel_id not in info["allowed_channels"]:
                        await message.reply("Channel not currently allowed.")
                        return
                    try:
                        info["allowed_channels"].remove(channel_id)
                        await save_config_state()
                        await message.reply(f"Successfully blocked channel {message.channel.mention} (`{channel_id}`)")
                    except ValueError:
                         await message.reply("Channel was not in the allowed list (this shouldn't happen).") # Should be caught by the check above
                    return

                # --- .ignoreword Command (Dev Only) ---
                elif matched_phrase == ".ignoreword":
                    if message.author.id != info['dev_id']: return
                    parts = message.content.split(maxsplit=1)
                    if len(parts) < 2:
                        await message.reply("Usage: `.ignoreword <word>`", delete_after=10.0)
                        return
                    word = parts[1].lower() # Store ignored words in lowercase
                    if word in info["ignored_words"]:
                        await message.reply(f"Word '{word}' already ignored.")
                        return
                    info["ignored_words"].append(word)
                    await save_config_state()
                    await message.reply(f"Successfully ignored word '{word}'")
                    return

                # --- .unignoreword Command (Dev Only) ---
                elif matched_phrase == ".unignoreword":
                    if message.author.id != info['dev_id']: return
                    parts = message.content.split(maxsplit=1)
                    if len(parts) < 2:
                        await message.reply("Usage: `.unignoreword <word>`", delete_after=10.0)
                        return
                    word = parts[1].lower()
                    if word not in info["ignored_words"]:
                        await message.reply(f"Word '{word}' not currently ignored.")
                        return
                    try:
                        info["ignored_words"].remove(word)
                        await save_config_state()
                        await message.reply(f"Successfully unignored word '{word}'")
                    except ValueError:
                         await message.reply(f"Word '{word}' was not in the ignored list (this shouldn't happen).")
                    return

                # --- .commands Command (Dev Only) ---
                elif matched_phrase == ".commands":
                    if message.author.id != info['dev_id']: return
                    await message.reply(f"**Dev Commands:**\n{', '.join(phrases)}")
                    return

                # --- .ban Command (Dev Only) ---
                elif matched_phrase == ".ban":
                    if message.author.id != info['dev_id']: return
                    parts = message.content.split(maxsplit=1)
                    if len(parts) < 2:
                        await message.reply("Usage: `.ban <user_id or @mention>`", delete_after=10.0)
                        return
                    user_target = parts[1]
                    user_id_to_ban = None
                    if message.mentions:
                        user_id_to_ban = message.mentions[0].id
                    else:
                        try:
                            user_id_to_ban = int(user_target)
                        except ValueError:
                            await message.reply("Invalid User ID provided.", delete_after=10.0)
                            return

                    if user_id_to_ban == info['dev_id']:
                        await message.reply("You cannot ban yourself!")
                        return
                    if user_id_to_ban in info["banned_users"]:
                        await message.reply(f"User `{user_id_to_ban}` already banned.")
                        return
                    info["banned_users"].append(user_id_to_ban)
                    await save_config_state()
                    await message.reply(f"Successfully banned user `{user_id_to_ban}`")
                    return

                # --- .unban Command (Dev Only) ---
                elif matched_phrase == ".unban":
                    if message.author.id != info['dev_id']: return
                    parts = message.content.split(maxsplit=1)
                    if len(parts) < 2:
                        await message.reply("Usage: `.unban <user_id or @mention>`", delete_after=10.0)
                        return
                    user_target = parts[1]
                    user_id_to_unban = None
                    if message.mentions:
                        user_id_to_unban = message.mentions[0].id
                    else:
                        try:
                            user_id_to_unban = int(user_target)
                        except ValueError:
                            await message.reply("Invalid User ID provided.", delete_after=10.0)
                            return

                    if user_id_to_unban not in info["banned_users"]:
                        await message.reply(f"User `{user_id_to_unban}` not currently banned.")
                        return
                    try:
                        info["banned_users"].remove(user_id_to_unban)
                        await save_config_state()
                        await message.reply(f"Successfully unbanned user `{user_id_to_unban}`")
                    except ValueError:
                         await message.reply(f"User `{user_id_to_unban}` was not in the banned list (this shouldn't happen).")
                    return

                # --- .img2location Command ---
                elif matched_phrase == ".img2location":
                    if not message.attachments:
                        await message.reply("Please attach an image.", delete_after=10.0)
                        return
                    image_url = None
                    for attachment in message.attachments:
                        if attachment.content_type and attachment.content_type.startswith('image/'):
                             image_url = attachment.url # img2location likely takes single URL
                             break
                    if not image_url:
                         await message.reply("No valid image attachment found.", delete_after=10.0)
                         return
                    try:
                        location = await img2location(image_url)
                        await message.reply (f"**Location Guess:**\n\n{location}")
                    except Exception as e:
                        print(f"Error in img2location: {traceback.format_exc()}")
                        await message.reply("An error occurred while guessing the location.", delete_after=10.0)
                        await send_webhook_message(
                            f"<@{info['dev_id']}> Error in .img2location in {message.channel.name} {message.jump_url}: \n\n**{e}**"
                        )
                    return

                # --- .help Command ---
                elif matched_phrase == ".help":
                    # Update help message to reflect current commands/defaults
                    help_message = f"""
**Commands:**
- `.imagine <prompt>`: Generates an image.
  - `--ar <ratio>`: Aspect ratio (e.g., `16:9`, `1:1`). Default `1:1`.
  - `--exact`: Prevents AI prompt enhancement.
  - `--model <name>`: Model (`sd3`, `dalle3`). Default `{info.get("default_image_gen", "sd3")}`.
  - `--video`: Generate a video instead (uses `text2video` model).
- `.img2location`: Guess location from attached image.
- `.img2txt`: Describe attached image.
- `.translate`: Reply to a message to translate it.
- `.uwu`: Reply to a message to uwuify it.
- `.help`: Shows this message.

**Dev Only Commands:** `.say`, `.delete`, `.change`, `.settings`, `.clear`, `.allowchannel`, `.blockchannel`, `.ignoreword`, `.unignoreword`, `.commands`, `.ban`, `.unban`, `.purge`
""".strip()
                    await asyncio.sleep(round(random.uniform(1,3), 2))
                    await message.reply(help_message)
                    return

                # --- .uwu Command ---
                elif matched_phrase == ".uwu":
                    if message.reference is None or message.reference.message_id is None:
                        await message.reply("Pwease wepwy to a message to uwuify it >w<", delete_after=10.0)
                        return
                    try:
                        original_message = await channel.fetch_message(message.reference.message_id)
                        original_content = original_message.content
                        if not original_content:
                             await message.reply("The wepwy message has no texty-wexty OwO", delete_after=10.0)
                             return

                        async with aiohttp.ClientSession() as session:
                            async with session.post("https://uwu.pm/api/v1/uwu", json={"text": original_content}) as resp:
                                if resp.status != 200:
                                    print(f"uwu.pm API error: Status {resp.status}")
                                    await channel.send('Oopsie woopsie! The uwu machine bwoke T~T')
                                    return
                                data = await resp.json()
                                uwu_text = data.get("uwu", "Something went wwong :(") # Handle potential API change
                                await message.reply(uwu_text)
                    except discord.NotFound:
                         await message.reply("Couldn't find the message you wepwied to ;~;", delete_after=10.0)
                    except Exception as e:
                        print(f"Error during uwuify: {traceback.format_exc()}")
                        await message.reply("An ewwow occuwwed duwing uwuification T_T", delete_after=10.0)
                    return

                # --- .purge Command (Dev Only) ---
                elif matched_phrase == ".purge":
                    if message.author.id != info['dev_id']: return
                    parts = message.content.split(maxsplit=1)
                    if len(parts) < 2 or not parts[1].isdigit():
                        await message.reply("Usage: `.purge <number_of_messages_to_check>`", delete_after=10.0)
                        return
                    limit = int(parts[1])
                    if limit <= 0 or limit > 100: # Add reasonable limits
                         await message.reply("Please provide a number between 1 and 100.", delete_after=10.0)
                         return
                    try:
                        deleted_count = 0
                        async for msg in channel.history(limit=limit):
                            if msg.author.id == client.user.id:
                                await msg.delete()
                                deleted_count += 1
                                await asyncio.sleep(0.5) # Add small delay to avoid rate limits
                        await message.reply(f"Successfully deleted {deleted_count} of my messages from the last {limit} checked.", delete_after=10.0)
                        await message.delete() # Delete command message
                    except discord.Forbidden:
                         await message.reply("I don't have permission to delete messages here.", delete_after=10.0)
                    except Exception as e:
                        print(f"Error during purge: {e}")
                        await message.reply("An error occurred during purge.", delete_after=10.0)
                    return

        # --- End Command Handling ---

    # --- Regular Message Handling (Mentions / Random Chance) ---

    # Determine if the bot should respond (mention or random chance)
    should_respond_randomly = (random.randint(1, 20) == 1 and
                               is_allowed_channel and # Only respond randomly in allowed channels
                               "system" not in message.content.lower()) # Avoid triggering on "system" keyword
    is_mentioned = client.user.mentioned_in(message) or "minion" in message.content.lower()

    if is_mentioned or should_respond_randomly:
        # Process one request at a time using global response lock
        async with response_lock:
            # --- Prepare Conversation History ---
            history_to_use = []
            if is_mentioned:
                # Use channel-specific history for mentions
                if channel.id not in conversation_history:
                    conversation_history[channel.id] = []
                # Only keep the last 10 messages (5 pairs)
                conversation_history[channel.id] = conversation_history[channel.id][-10:]
                history_to_use = conversation_history[channel.id].copy() # Use a copy

            elif should_respond_randomly:
                # Fetch recent history for random responses
                try:
                    recent_msgs = [m async for m in message.channel.history(limit=10)]
                    recent_msgs.reverse() # Oldest first

                    temp_history = []
                    for msg in recent_msgs:
                        msg_author_name = await get_name(msg.author)
                        msg_content_formatted = f"Name: {msg_author_name}\nMessage: {msg.content}"
                        # Add reply context if present
                        if msg.reference and msg.reference.message_id:
                             try:
                                 ref_msg = await channel.fetch_message(msg.reference.message_id)
                                 ref_author_name = await get_name(ref_msg.author)
                                 msg_content_formatted = f"User is replying to {ref_author_name}'s message: {ref_msg.content}\n\nReply:\n{msg_content_formatted}"
                             except Exception: pass # Ignore if fetching reply fails

                        role = "assistant" if msg.author.id == client.user.id else "user"
                        temp_history.append({"role": role, "content": msg_content_formatted})
                    history_to_use = temp_history # Use this fetched history
                except Exception as e:
                    print(f"Error fetching history for random response: {e}")
                    return # Don't proceed if history fetch fails

            # --- Format Current Message ---
            current_message_content = message.content
            reply_context = ""
            image_context = ""

            # Handle Replies
            if message.reference and message.reference.message_id:
                try:
                    original_message = await channel.fetch_message(message.reference.message_id)
                    original_author_name = await get_name(original_message.author)
                    reply_context = f"User is replying to {original_author_name}'s message: {original_message.content}\n\nReply:\n"
                    # Check if replied-to message had images (for context)
                    if not message.attachments and original_message.attachments:
                         for attachment in original_message.attachments:
                             if attachment.content_type and attachment.content_type.startswith('image/'):
                                 try:
                                     caption = await img2txt([attachment.url])
                                     image_context += f"Context from replied image: {caption}\n"
                                 except Exception as img_e:
                                     print(f"Error getting caption for replied image: {img_e}")
                                     image_context += "Context from replied image: [Could not analyze]\n"
                                 break # Only process first image in reply context
                except Exception as e:
                    print(f"Error fetching referenced message for context: {e}")
                    reply_context = "User is replying to a message.\n\nReply:\n"

            # Handle Images in Current Message
            if message.attachments:
                image_urls = []
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith('image/'):
                        image_urls.append(attachment.url)
                if image_urls:
                    try:
                        # Pass message content as context for image description
                        caption = await img2txt(image_urls, message.content)
                        image_context += f"User sent an image, image description: {caption}\n"
                    except Exception as img_e:
                        print(f"Error getting caption for current image: {img_e}")
                        image_context += "User sent an image: [Could not analyze]\n"

            # Construct final prompt content
            final_content = f"{reply_context}{image_context}Name: {user_nickname}\nMessage: {current_message_content}"
            final_content = final_content.replace(f'<@{client.user.id}>', '@Minion') # Replace bot mention
            # Replace other user mentions
            for mention in message.mentions:
                 if mention.id != client.user.id: # Don't replace bot mention again
                     mention_name = await get_name(mention)
                     final_content = final_content.replace(f'<@{mention.id}>', f"@{mention_name}")

            # Add current message to history for the API call
            history_to_use.append({"role": "user", "content": final_content})

            # Add prompt for random response if needed
            if should_respond_randomly and not is_mentioned:
                 history_to_use.append({"role": "user", "content": "Reply with a random message to send in this conversation"})


            # --- Generate Response ---
            reply = None
            image_url = None
            try:
                update_date() # Update date/clear said words if needed

                # Load said words (read-only for checking)
                said_words = []
                try:
                    with open(SAID_WORDS_TXT_FILE, "rb") as f:
                        said_words = f.read().decode('utf-8').splitlines()
                except FileNotFoundError:
                    print(f"{SAID_WORDS_TXT_FILE} not found, initializing.")
                    with open(SAID_WORDS_TXT_FILE, "w") as f: f.write("") # Create empty file

                await asyncio.sleep(random.uniform(1, 3) if should_respond_randomly else random.uniform(1,6)) # Shorter delay for random

                # Call the LLM API
                response_data = await get_response(
                    type=info["type"],
                    model=info["model"],
                    system_prompt=system_prompt, # System prompt defined earlier
                    prompt=final_content, # Pass the formatted current message as the main prompt
                    temperature=info["temperature"],
                    conversation_history=history_to_use[:-1], # Pass history *before* the current message
                    use_function=True # Allow function calls if applicable (e.g., image gen within response)
                )

                reply = response_data.get("text")
                image_url = response_data.get("image") # Check if LLM decided to generate an image

                # --- Post-generation checks ---
                if reply is None:
                     print("LLM response was None.")
                     return # Exit if no text response

                # Clean up common LLM artifacts
                prefixes_to_remove = ["Human:", "User:", "Reply:", "Message:", "assistant:", "Name:"]
                for prefix in prefixes_to_remove:
                     if reply.startswith(prefix):
                         reply = reply[len(prefix):].strip()
                if "assistant" in reply: # Remove if it appears anywhere
                    reply = reply.replace("assistant", "").strip()


                # Check against said words (only if it was a random response)
                if should_respond_randomly and not is_mentioned:
                    count = 0
                    while any(line in reply for line in said_words) and count < 5:
                        print("Regenerating random response (matched said_words)...")
                        # Rerun generation for random response
                        response_data = await get_response(type=info["type"], model=info["model"], system_prompt=system_prompt, prompt="Reply with a random message to send in this conversation", temperature=info["temperature"], conversation_history=history_to_use[:-1], use_function=False) # No function use for simple random
                        reply = response_data.get("text")
                        if reply is None: break # Exit loop if regen fails
                        # Clean up again
                        for prefix in prefixes_to_remove:
                             if reply.startswith(prefix): reply = reply[len(prefix):].strip()
                        if "assistant" in reply: reply = reply.replace("assistant", "").strip()
                        count += 1

                    if reply is None: # Check again after loop
                         print("Random response regeneration failed.")
                         return

                    # Save the *final* random response to said_words
                    try:
                        with open(SAID_WORDS_TXT_FILE, "ab") as f:
                            f.write((reply + "\n").encode('utf-8'))
                            print(f"Added random response '{reply[:50]}...' to said_words")
                    except IOError as e:
                        print(f"Error writing random response to {SAID_WORDS_TXT_FILE}: {e}")

            except Exception as e:
                exc_traceback = traceback.format_exc()
                print(f"An exception occurred during response generation: \n\n{exc_traceback}\n")
                await send_webhook_message(
                    f"<@{info['dev_id']}> Exception during response generation in {message.channel.name} {message.jump_url}: \n\n**{e}**"
                )
                # Don't pop history here as it wasn't added yet for the assistant turn
                return # Exit on generation error

            # --- Process and Send Valid Response (Ensure this block is outside the try...except) ---
            if reply is None: # Check if reply generation failed silently or errored out
                 print("Response generation resulted in None or error, skipping further processing.")
                 return

            # Check for ignored words

            # Check for ignored words
            if await check_ignored_words(reply, info.get("ignored_words", [])):
                print(f"Response rejected (ignored word): '{reply[:100]}...'")
                await send_webhook_message(
                    f"<@{info['dev_id']}> Response rejected (ignored word) in {message.channel.name} {message.jump_url}: \n\n**{reply}**"
                )
                # Optionally react to the user's message
                try: await message.add_reaction('❌')
                except Exception: pass
                return

            # Handle reactions embedded in the response
            reaction_emoji = None
            if reply.startswith("react("):
                end_index = reply.find(')')
                if end_index != -1:
                    emoji_str = reply[len("react("):end_index].strip("'\" ")
                    if emoji_str:
                        reaction_emoji = emoji_str
                        reply = reply[end_index+1:].strip() # Remove the react() part

            # Limit message splitting
            messages_to_send = reply.split(break_string)
            if len(messages_to_send) > 2: # Limit to max 2 messages
                messages_to_send = [reply.replace(break_string, "\n")] # Join back if too many splits

            # Add assistant response to channel history *only if mentioned*
            if is_mentioned:
                 assistant_response_for_history = reply # Store the full reply before splitting
                 conversation_history[channel.id].append({"role": "assistant", "content": assistant_response_for_history})
                 # Trim history again after adding assistant message
                 conversation_history[channel.id] = conversation_history[channel.id][-10:]


            # Send the response messages
            async with message.channel.typing():
                try:
                    # Add reaction first if present
                    if reaction_emoji:
                        try: await message.add_reaction(reaction_emoji)
                        except Exception as react_e: print(f"Failed to add reaction {reaction_emoji}: {react_e}")

                    # Send text messages
                    sent_reply_message = None
                    for i, msg_chunk in enumerate(messages_to_send):
                        msg_chunk = msg_chunk.strip()
                        if not msg_chunk: continue # Skip empty chunks

                        sleep_time = len(msg_chunk) * 0.08 # Slightly faster typing speed
                        await asyncio.sleep(sleep_time)

                        # Attach image only to the first message if generated
                        file_to_send = None
                        if i == 0 and image_url:
                             try:
                                 async with aiohttp.ClientSession() as session:
                                     async with session.get(image_url) as resp:
                                         if resp.status == 200:
                                             img_data = io.BytesIO(await resp.read())
                                             file_to_send = discord.File(img_data, 'image.png')
                                         else:
                                             print(f"Failed to download generated image: {resp.status}")
                             except Exception as img_dl_e:
                                 print(f"Error downloading generated image: {img_dl_e}")

                        # Send message
                        if i == 0:
                            # Reply to original message for the first chunk
                            sent_reply_message = await message.reply(msg_chunk, file=file_to_send)
                        else:
                            # Send subsequent chunks normally
                            await message.channel.send(msg_chunk)

                except discord.Forbidden:
                     print(f"Forbidden error sending message in {channel.name}")
                     # Maybe remove channel from allowed list if this persists?
                except Exception as send_e:
                    exc_traceback = traceback.format_exc()
                    print(f'An exception occurred while sending the message: \n\n{exc_traceback}\n')
                    await send_webhook_message(
                        f"<@{info['dev_id']}> Exception sending message in {message.channel.name} {message.jump_url}: \n\n**{send_e}**"
                    )
                    # If sending failed, remove the assistant message we optimistically added
                    if is_mentioned and conversation_history[channel.id] and conversation_history[channel.id][-1]["role"] == "assistant":
                         conversation_history[channel.id].pop()
                    return # Exit handler if sending fails

            # --- End Mention/Random Response Block ---
            return # Ensure we exit after handling a mention/random response

    # --- End on_message ---


# --- Run the Bot ---
if __name__ == "__main__":
    print("Starting bot...")
    try:
        client.run(TOKEN)
    except discord.LoginFailure:
        print("CRITICAL ERROR: Improper token passed. Ensure DISCORD_TOKEN is correct.")
    except Exception as e:
        print(f"CRITICAL ERROR: An error occurred running the bot: {e}")
        print(traceback.format_exc())
