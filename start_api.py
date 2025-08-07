"""
Startup script for ID Photo Validator API.
"""

import uvicorn
import sys
import os

def main():
    """Start the FastAPI server."""
    
    print("🚀 Starting ID Photo Validator API...")
    print("📊 Loading AI models (this may take a moment on first run)...")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 Interactive docs at: http://localhost:8000/docs")
    print("📋 Alternative docs at: http://localhost:8000/redoc")
    print("❤️  Health check at: http://localhost:8000/health")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 API server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
