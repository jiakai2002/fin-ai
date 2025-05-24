import os
import time
import pandas as pd
import mysql.connector
from metadata import column_mapping

# script to establish company financial info MySQL database

start_time = time.time()  # ⏱ Start timer

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv("MYSQL_ROOT_PASSWORD"),
    'database': 'financial_db'
}

# Read CSV file that exists in same directory as this script
df = pd.read_csv(f'{os.path.dirname(__file__)}/20_year_data.csv')
df = df.rename(columns=column_mapping)

# Remove duplicates: Keep the last entry for each company and year
df = df.sort_values('year').drop_duplicates(subset=['company_id', 'year'], keep='last')

# Drop unused columns
df = df[[col for col in df.columns if col in column_mapping.values()]]

# Connect to DB
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

columns = df.columns.tolist()

# Create table if it doesn't exist
create_table_query = f"""
CREATE TABLE IF NOT EXISTS company_data (
    {', '.join([f"{col} VARCHAR(255)" for col in columns])},
    PRIMARY KEY (company_id, year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
cursor.execute(create_table_query)

# Insert each row
for _, row in df.iterrows():
    values = [row[col] if pd.notna(row[col]) else None for col in columns]

    insert_query = f"""
        INSERT INTO company_data ({', '.join(columns)})
        VALUES ({', '.join(['%s'] * len(columns))})
        ON DUPLICATE KEY UPDATE
        {', '.join(f"{col} = VALUES({col})" for col in columns if col not in ['company_id', 'year'])}
    """
    cursor.execute(insert_query, values)

conn.commit()
print(f"{len(df)} rows inserted successfully")

# Run stored procedure
cursor.callproc('calculate_financial_data')
conn.commit()
print(f"Financial ratios updated successfully")

# Cleanup
cursor.close()
conn.close()

# ⏱ End timer
end_time = time.time()
print(f"⏱ Time taken: {end_time - start_time:.2f} seconds")
