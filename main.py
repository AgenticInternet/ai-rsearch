"""
CLI Entry Point for DeepAgents Search Orchestrator
Provides command-line interface for the multi-agent search system
"""

import asyncio
import sys

from src.core import DeepAgentSearchOrchestrator


async def main():
    """Main execution function for CLI."""
    print("🧠 DeepAgent Multiagent Search System (Remote MCP)")
    print("🏆 AI Tinkerers Hackathon Entry")
    print("🌐 Connecting to internet-accessible MCP servers")
    
    try:
        orchestrator = DeepAgentSearchOrchestrator()
        
        # Check command line arguments or run demo by default
        if len(sys.argv) > 1:
            if sys.argv[1] == "demo":
                await orchestrator.run_comprehensive_demo()
            elif sys.argv[1] == "interactive":
                await orchestrator.interactive_mode()
            elif sys.argv[1] == "test":
                connectivity = await orchestrator.test_mcp_connectivity()
                print("\n🔗 MCP Server Connectivity Results:")
                for server, result in connectivity.items():
                    print(f"  {server}: {result['status']} ({result['url']})")
            else:
                print("Usage: python main.py [demo|interactive|test]")
                print("\nOptions:")
                print("  demo        - Run comprehensive demo")
                print("  interactive - Start interactive query mode")
                print("  test        - Test MCP server connectivity")
        else:
            # Default: run comprehensive demo
            await orchestrator.run_comprehensive_demo()
            
    except Exception as e:
        print(f"💥 System initialization failed: {str(e)}")
        print("Please check your MCP server URLs and network connectivity.")


if __name__ == "__main__":
    asyncio.run(main())
