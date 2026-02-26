"""
Script to start the LTMAgent API server.
"""

import subprocess # nosec B404
import sys
import os

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"]) # nosec B603: Trusted arguments, no shell injection risk.

def run_api_server():
    """Run the API server."""
    print("Starting LTMAgent API server...")
    print("API documentation available at: http://localhost:8000/docs")
    
    try:
        subprocess.run(["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]) # nosec B603, B607: Trusted args, uvicorn is expected in PATH. For production, resolve full path.
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    except FileNotFoundError:
        print("uvicorn not found. Installing dependencies...")
        install_dependencies()
        subprocess.run(["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]) # nosec B603, B607: Trusted args, uvicorn is expected in PATH. For production, resolve full path.

if __name__ == "__main__":
    # Change to the LTMAgent directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_api_server()