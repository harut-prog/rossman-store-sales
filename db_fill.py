## New commit

import psycopg2
import psycopg2.extras
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()

data = pd.read_csv('history.csv', parse_dates=['Date'], low_memory=False)

with psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost",
    port=5433
) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            DROP TABLE IF EXISTS history;

            CREATE TABLE history(
                Store INTEGER,
                DayOfWeek INTEGER,
                Date TIMESTAMP,
                Sales NUMERIC(10, 2),
                Customers INTEGER,
                Open BOOLEAN,
                Promo BOOLEAN,
                StateHoliday CHAR(1),
                SchoolHoliday BOOLEAN,
                StoreType CHAR(1),
                Assortment CHAR(1),
                CompetitionDistance NUMERIC(10,2),
                CompetitionOpenSinceMonth FLOAT,
                CompetitionOpenSinceYear FLOAT,
                Promo2 BOOLEAN,
                Promo2SinceWeek INTEGER,
                Promo2SinceYear INTEGER,
                PromoInterval VARCHAR(32),

                CONSTRAINT pk_unique UNIQUE (Store, Date)
            );

            CREATE INDEX idx_history_date ON history (Date);
            """
        )

        records = []
        for _, row in data.iterrows():
            records.append((
                row["Store"], row["DayOfWeek"], row["Date"], row["Sales"],
                row["Customers"], row["Open"], row["Promo"], row["StateHoliday"],
                row["SchoolHoliday"], row["StoreType"], row["Assortment"],
                row["CompetitionDistance"], row["CompetitionOpenSinceMonth"],
                row["CompetitionOpenSinceYear"], row["Promo2"], row["Promo2SinceWeek"],
                row["Promo2SinceYear"], row["PromoInterval"]
            ))

        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO history
            VALUES (%s, %s, %s, %s, %s, %s::boolean, %s::boolean, %s, %s::boolean, %s, %s, %s, %s, %s, %s::boolean, %s, %s, %s)
            """,
            records,
            page_size=1000,
        )

    conn.commit()
