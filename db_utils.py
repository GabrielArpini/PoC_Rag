import psycopg2
import os 

DB_CONFIGS = {
    'dbname':os.getenv('DB_NAME', 'rag_db'),
    'user':os.getenv('DB_USER', 'postgres'),
    'password':os.getenv('DB_PASSWORD', 'postgres'),
    'host':os.getenv('DB_HOST', 'localhost'),
    'port':'5432'
}

def get_conn():
    conn = psycopg2.connect(**DB_CONFIGS)
    return conn


if __name__ == '__main__':
   print("Faz coisa.") 

