"""
Command Line Interface for the LTM application.
Uses the LTMService to provide a clean interface to the core functionality.
"""

from core.service import LTMService
from core.graph_builder import pretty_print_stream_chunk
from config.app_config import validate_config

def main():
    """Main entry point for the CLI application."""
    try:
        # Validate config minimally for MVP
        validate_config()

        print("Initializing LTM Service...")
        service = LTMService()
        
        # Display model information
        model_info = service.get_model_info()
        print(f"Using {model_info['provider']} with model: {model_info['model_name']}")

        # Handle user ID
        print("\n=== User Management ===")
        user_id = input("Enter user ID (press Enter for auto-generated ID): ").strip()
        if not user_id:
            user_id = service.create_user_id()
            print(f"Using auto-generated user ID: {user_id}")

        # Handle thread ID
        print("\n=== Thread Management ===")
        use_existing = input("Use existing thread? (y/n, default: n): ").strip().lower()

        if use_existing == 'y':
            # In a real implementation, we would show existing threads here
            # For now, just ask for the thread ID
            thread_id = input("Enter thread ID: ").strip()
            if not thread_id:
                thread_id = service.create_thread_id()
                print(f"No thread ID provided, using new thread: {thread_id}")
        else:
            thread_id = service.create_thread_id()
            print(f"New conversation thread ID: {thread_id}")

        # Run conversation loop
        print("\nLTM Agent ready for conversation!")
        print(f"User ID: {user_id}, Thread ID: {thread_id}")

        while True:
            try:
                user_prompt = input("User>>> ")
                if not user_prompt.strip():
                    continue

                # Process the user input using the service
                for chunk in service.process_message(user_prompt, user_id, thread_id):
                    pretty_print_stream_chunk(chunk)
            except KeyboardInterrupt:
                print("\nExiting LTM Agent. Goodbye!")
                break
            except Exception as e:
                print(f"Error processing message: {e}")
                print("Please try again or exit with Ctrl+C")
                
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please ensure all required packages are installed.")
        return 1
    except Exception as e:
        print(f"Failed to initialize LTM Service: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)