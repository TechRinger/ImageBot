import discord
from discord import app_commands
import json
import aiohttp
import asyncio
from datetime import datetime
import os
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import re
import random



# Environment variable configuration
ENHANCE_PROMPT = os.getenv('ENHANCE_PROMPT', 'false').lower() == 'true'
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
COMFYUI_HOST = os.getenv('COMFYUI_HOST', 'localhost')
COMFYUI_PORT = os.getenv('COMFYUI_PORT', '8188')
GOOGLE_DRIVE_FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
CHECKPOINT_NAME = os.getenv('CHECKPOINT_NAME', 'sd_v1.5.ckpt')
NSFW_KEYWORDS_FILE = os.getenv('NSFW_KEYWORDS_FILE', 'nsfw_keywords.txt')

# Load NSFW keywords from file
try:
    with open(NSFW_KEYWORDS_FILE, 'r') as f:
        NSFW_KEYWORDS = [line.strip().lower() for line in f if line.strip()]
except FileNotFoundError:
    NSFW_KEYWORDS = [
        'nude', 'naked', 'explicit', 'nsfw',
        # Default keywords if file not found
    ]

class ImageGenerationBot(discord.Client):
    def __init__(self):
        # Enable message content intent
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self.comfyui_url = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
        self.ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
        self.drive_folder_id = GOOGLE_DRIVE_FOLDER_ID
        self.setup_google_drive()
        
        # Create a local directory for downloaded images
        self.download_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(self.download_dir, exist_ok=True)

    async def setup_hook(self):
        """This is called when the bot starts up"""
        print("Syncing slash commands...")
        try:
            # Sync commands globally
            await self.tree.sync()
            print("Slash commands synced successfully!")
        except Exception as e:
            print(f"Error syncing slash commands: {e}")

    async def enhance_prompt(self, original_prompt: str) -> str:
            """
            Enhance the user's prompt using Ollama LLM.
            """
            if not ENHANCE_PROMPT:
                return original_prompt

            system_prompt = """You are an expert at writing prompts for image generation. 
            Your task is to enhance the given prompt by adding more descriptive details, 
            artistic style references, and technical parameters that will result in a 
            higher quality image. Keep the core subject and intent of the original prompt 
            intact while making it more detailed and artistic. Return only the enhanced 
            prompt without any explanation or additional text."""

            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original prompt: {original_prompt}"}
                ],
                "stream": False  # Add this to get a single response
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.ollama_url}/api/chat",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            # Read the raw response text
                            text = await response.text()
                            try:
                                # Parse the JSON response
                                result = json.loads(text)
                                enhanced_prompt = result.get('message', {}).get('content', '').strip()
                                if enhanced_prompt:
                                    print(f"Enhanced prompt: {enhanced_prompt}")
                                    return enhanced_prompt
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON response: {e}")
                                print(f"Raw response: {text}")
                        return original_prompt
            except Exception as e:
                print(f"Error enhancing prompt: {str(e)}")
                return original_prompt

    async def download_image(self, image_name):
        """Download the image from ComfyUI server"""
        download_url = f"{self.comfyui_url}/api/view"
        local_path = os.path.join(self.download_dir, image_name)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{download_url}?filename={image_name}") as response:
                    if response.status == 200:
                        with open(local_path, 'wb') as f:
                            f.write(await response.read())
                        print(f"Successfully downloaded image to: {local_path}")
                        return local_path
                    else:
                        print(f"Failed to download image. Status: {response.status}")
                        return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
        

    def setup_google_drive(self):
        try:
            # Parse the JSON string from environment variable
            service_account_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
            
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            self.drive_service = build('drive', 'v3', credentials=credentials)
        except Exception as e:
            print(f"Failed to initialize Google Drive service: {e}")
            raise


    async def check_nsfw_content(self, prompt):
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in NSFW_KEYWORDS)

    async def generate_image(self, prompt):
        workflow = {
            "6": {
                "inputs": {
                "text": prompt,
                "clip": [
                    "30",
                    1
                ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                "title": "CLIP Text Encode (Positive Prompt)"
                }
            },
            "8": {
                "inputs": {
                "samples": [
                    "31",
                    0
                ],
                "vae": [
                    "30",
                    2
                ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                "title": "VAE Decode"
                }
            },
            "9": {
                "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    "8",
                    0
                ]
                },
                "class_type": "SaveImage",
                "_meta": {
                "title": "Save Image"
                }
            },
            "27": {
                "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
                },
                "class_type": "EmptySD3LatentImage",
                "_meta": {
                "title": "EmptySD3LatentImage"
                }
            },
            "30": {
                "inputs": {
                "ckpt_name": CHECKPOINT_NAME
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                "title": "Load Checkpoint"
                }
            },
            "31": {
                "inputs": {
                "seed": random.randint(1,4294967294),
                "steps": 4,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": [
                    "30",
                    0
                ],
                "positive": [
                    "6",
                    0
                ],
                "negative": [
                    "33",
                    0
                ],
                "latent_image": [
                    "27",
                    0
                ]
                },
                "class_type": "KSampler",
                "_meta": {
                "title": "KSampler"
                }
            },
            "33": {
                "inputs": {
                "text": "",
                "clip": [
                    "30",
                    1
                ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                "title": "CLIP Text Encode (Negative Prompt)"
                }
            }
            }
        print(f"Trying to generate image with workflow: {prompt}")
    
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.comfyui_url}/prompt",
                    json={"prompt": workflow}
                ) as response:
                    if response.status != 200:
                        print(f"Error from ComfyUI API: {await response.text()}")
                        return None
                    prompt_data = await response.json()
                    prompt_id = prompt_data['prompt_id']
                    print(f"Prompt ID: {prompt_id}")

                while True:
                    async with session.get(f"{self.comfyui_url}/api/history/{prompt_id}") as response:
                        if response.status == 200:
                            history = await response.json()
                            if prompt_id in history:
                                image_data = history[prompt_id]
                                if 'outputs' in image_data:
                                    for node_id, node_data in image_data['outputs'].items():
                                        if 'images' in node_data and node_data['images']:
                                            filename = node_data['images'][0]['filename']
                                            print(f"Found image filename: {filename}")
                                            # Download the image and get its local path
                                            local_path = await self.download_image(filename)
                                            return local_path
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    async def upload_to_drive(self, file_path, new_filename):
            """
            Upload a file to Google Drive
            
            Args:
                file_path (str): The full path to the local file
                new_filename (str): The desired filename in Google Drive
            """
            try:
                print(f"Attempting to upload file: {file_path}")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                    
                file_metadata = {
                    'name': new_filename,
                    'parents': [self.drive_folder_id]
                }
                media = MediaFileUpload(file_path, mimetype='image/png')
                file = self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"Successfully uploaded file to Drive with ID: {file.get('id')}")
                return file.get('id')
            except Exception as e:
                print(f"Error uploading to Drive: {e}")
                return None


    async def format_response_message(self, original_prompt: str, final_prompt: str) -> str:
        """Format the response message to include prompts"""
        if ENHANCE_PROMPT and original_prompt != final_prompt:
            return f"Image prompt: {final_prompt}"
        else:
            return f"Prompt: {final_prompt}"

    async def process_image_request(self, message, original_prompt: str, final_prompt: str):
        """Helper function to handle image generation and response"""
        async with message.channel.typing():
            try:
                local_image_path = await self.generate_image(final_prompt)
                if not local_image_path:
                    await message.reply("Failed to generate image.")
                    return

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{message.author.name}_{timestamp}.png"
                
                drive_file_id = await self.upload_to_drive(local_image_path, new_filename)
                print(f"Uploaded to Google Drive with ID: {drive_file_id}")

                # Format the response message
                response_message = await self.format_response_message(original_prompt, final_prompt)
                
                await message.reply(
                    content=response_message,
                    file=discord.File(local_image_path, filename=new_filename)
                )
            except Exception as e:
                print(f"Error processing message: {e}")
                await message.reply("Sorry, something went wrong while processing your request.")


    async def on_message(self, message):
        if message.author == self.user:
            return

        # Handle !generate command in servers
        if message.content.startswith('!generate '):
            original_prompt = message.content[10:].strip()  # Remove '!generate ' from the start
            
            if await self.check_nsfw_content(original_prompt):
                final_prompt = "my little pony in a field"
            else:
                final_prompt = await self.enhance_prompt(original_prompt)

            await self.process_image_request(message, original_prompt, final_prompt)

        # Handle DM messages without command prefix
        elif isinstance(message.channel, discord.DMChannel):
            original_prompt = message.content
            
            if await self.check_nsfw_content(original_prompt):
                final_prompt = "my little pony in a field"
            else:
                final_prompt = await self.enhance_prompt(original_prompt)

            await self.process_image_request(message, original_prompt, final_prompt)

    @app_commands.command(
        name="generateimage",
        description="Generate an image from a text prompt"
    )
    async def generateimage(self, interaction: discord.Interaction, prompt: str):
        await interaction.response.defer()

        try:
            original_prompt = prompt
            if await self.check_nsfw_content(original_prompt):
                final_prompt = "my little pony in a field"
            else:
                final_prompt = await self.enhance_prompt(original_prompt)

            print(f"Generating image with prompt: {final_prompt}")
            local_image_path = await self.generate_image(final_prompt)
            if not local_image_path:
                await interaction.followup.send("Failed to generate image.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{interaction.user.name}_{timestamp}.png"
            
            drive_file_id = await self.upload_to_drive(local_image_path, new_filename)

            # Format the response message
            response_message = await self.format_response_message(original_prompt, final_prompt)

            await interaction.followup.send(
                content=response_message,
                file=discord.File(local_image_path, filename=new_filename)
            )
        except Exception as e:
            print(f"Error processing command: {e}")
            await interaction.followup.send("Sorry, something went wrong while processing your request.")



async def setup_and_run():
    """Setup and run the bot with proper error handling"""
    required_vars = [
        'DISCORD_BOT_TOKEN',
        'GOOGLE_DRIVE_FOLDER_ID',
        'GOOGLE_SERVICE_ACCOUNT_JSON'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    bot = ImageGenerationBot()
    
    try:
        print("Starting bot...")
        async with bot:
            await bot.start(DISCORD_TOKEN)
    except Exception as e:
        print(f"Error starting bot: {e}")
        raise

def main():
    """Main entry point with proper async handling"""
    asyncio.run(setup_and_run())

if __name__ == "__main__":
    main()