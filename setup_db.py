import asyncio
import asyncpg

async def init_db():
    passwords_to_try = ["postgres", "root", "admin", "password", ""]
    
    for pwd in passwords_to_try:
        print(f"Trying to connect to postgres on port 5433 with user 'postgres' and password '{pwd}'...")
        try:
            conn = await asyncpg.connect(user="postgres", password=pwd, host="localhost", port=5433, database="postgres")
            print(f"SUCCESS! Connected with password: {pwd}")
            
            # Create user if not exists
            print("Creating docurag_user...")
            try:
                await conn.execute("CREATE USER docurag_user WITH PASSWORD 'change-this-password';")
            except asyncpg.exceptions.DuplicateObjectError:
                print("docurag_user already exists.")
            
            # Create database if not exists
            print("Creating docurag database...")
            try:
                await conn.execute("CREATE DATABASE docurag OWNER docurag_user;")
            except asyncpg.exceptions.DuplicateDatabaseError:
                print("docurag database already exists.")
            
            # Grant privileges
            await conn.execute("GRANT ALL PRIVILEGES ON DATABASE docurag TO docurag_user;")
            
            # Connect to docurag DB to enable pgvector
            await conn.close()
            conn2 = await asyncpg.connect(user="postgres", password=pwd, host="localhost", port=5433, database="docurag")
            print("Enabling pgvector extension...")
            await conn2.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn2.close()
            
            print("Database setup complete.")
            return
        except Exception as e:
            print(f"Failed with {pwd}: {e}")

if __name__ == "__main__":
    asyncio.run(init_db())
