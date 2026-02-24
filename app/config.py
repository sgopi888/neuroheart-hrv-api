import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ["DATABASE_URL"]
API_KEY: str = os.environ["API_KEY"]
TZ: str = os.getenv("TZ", "America/New_York")
