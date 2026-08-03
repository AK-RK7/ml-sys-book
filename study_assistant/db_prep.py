from dotenv import load_dotenv

from study_assistant.db import init_db

load_dotenv()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()