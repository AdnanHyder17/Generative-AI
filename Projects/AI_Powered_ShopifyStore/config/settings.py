"""
config/settings.py - Central configuration for the Shopify AI Agent system.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Shopify
    SHOPIFY_ACCESS_TOKEN: str = os.getenv("X_SHOPIFY_ACCESS_TOKEN", "")
    SHOPIFY_STORE_DOMAIN: str = os.getenv("SHOPIFY_STORE_DOMAIN", "")
    SHOPIFY_API_VERSION: str = os.getenv("SHOPIFY_API_VERSION", "")

    @property
    def shopify_base_url(self) -> str:
        return f"https://{self.SHOPIFY_STORE_DOMAIN}/admin/api/{self.SHOPIFY_API_VERSION}"

    @property
    def shopify_headers(self) -> dict:
        return {
            "X-Shopify-Access-Token": self.SHOPIFY_ACCESS_TOKEN,
            "Content-Type": "application/json",
        }

    def validate(self) -> None:
        """Raise if required environment variables are missing."""
        missing = []
        if not self.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not self.SHOPIFY_ACCESS_TOKEN:
            missing.append("X_SHOPIFY_ACCESS_TOKEN")
        if not self.SHOPIFY_STORE_DOMAIN:
            missing.append("SHOPIFY_STORE_DOMAIN")
        if not self.SHOPIFY_API_VERSION:
            missing.append("SHOPIFY_API_VERSION")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


settings = Settings()