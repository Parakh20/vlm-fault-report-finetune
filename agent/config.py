# agent/config.py
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    max_steps: int = 25
    headless_default: bool = False


def load_settings() -> Settings:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return Settings(gemini_api_key=api_key, gemini_model=model)


if __name__ == "__main__":
    print(load_settings())
