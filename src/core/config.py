"""
Configuration and environment validation module.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration manager for the DeepAgent Search System."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        load_dotenv()
        self._validate_environment()
        
    def _validate_environment(self):
        """Validate all required environment variables."""
        required_vars = {
            "ANTHROPIC_API_KEY": "Claude Sonnet 4 access",
            "MCP_SERPAPI_URL": "SerpApi MCP server URL (default: http://localhost:3001)",
            "MCP_ALGOLIA_URL": "Algolia MCP server URL (default: http://localhost:3002)",
            "MCP_OPENSEARCH_URL": "OpenSearch MCP server URL (default: http://localhost:3003)",
            "REDIS_URL": "Redis cache URL"
        }
        
        missing = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing.append(f"  ❌ {var}: {description}")
        
        if missing:
            print("🚨 Missing required environment variables:")
            print("\n".join(missing))
            print("\nPlease set these variables before running the orchestrator.")
            raise ValueError("Missing required environment variables")
        
        print("✅ All required environment variables validated")
        self._print_config()
        
    def _print_config(self):
        """Display current configuration."""
        print("\n🌐 MCP Server Configuration:")
        print(f"  🔍 SerpApi MCP: {self.serpapi_url}")
        print(f"  🗄️ Algolia MCP: {self.algolia_url}")
        print(f"  📊 OpenSearch MCP: {self.opensearch_url}")
        print(f"  🔍 Redis cache: {self.redis_url}")
    
    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key."""
        return os.getenv("ANTHROPIC_API_KEY", "")
    
    @property
    def serpapi_url(self) -> str:
        """Get SerpApi MCP server URL."""
        return os.getenv("MCP_SERPAPI_URL", "http://localhost:3001")
    
    @property
    def algolia_url(self) -> str:
        """Get Algolia MCP server URL."""
        return os.getenv("MCP_ALGOLIA_URL", "http://localhost:3002")
    
    @property
    def opensearch_url(self) -> str:
        """Get OpenSearch MCP server URL."""
        return os.getenv("MCP_OPENSEARCH_URL", "http://localhost:3003")
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929")
    
    @property
    def model_temperature(self) -> float:
        """Get model temperature."""
        return float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    
    def get_mcp_servers_config(self) -> Dict[str, Dict[str, str]]:
        """Get MCP servers configuration for MultiServerMCPClient."""
        return {
            "serpapi": {
                "url": self.serpapi_url,
                "transport": "streamable_http"
            },
            "algolia": {
                "url": self.algolia_url,
                "transport": "streamable_http"
            },
            "opensearch": {
                "url": self.opensearch_url,
                "transport": "streamable_http"
            }
        }
