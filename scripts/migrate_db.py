"""One-time migration: create the `history` table and load it from `history.csv`.

Works against any PostgreSQL reachable via DATABASE_URL, e.g.:
  local compose:   postgresql://user:pass@localhost:5433/db
  managed YC:      postgresql://user:pass@<host>:6432/db
  k8s service:     postgresql://user:pass@<service>:5432/db
"""

import os

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    url = os.getenv("DATABASE_URL")
    csv_path = os.getenv("HISTORY_CSV", "history.csv")
    if not url:
        raise SystemExit("DATABASE_URL is not set")

    data = pd.read_csv(csv_path, parse_dates=["Date"], low_memory=False)

    with psycopg2.connect(url) as conn:
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

            records = [
                (
                    row["Store"], row["DayOfWeek"], row["Date"], row["Sales"],
                    row["Customers"], row["Open"], row["Promo"], row["StateHoliday"],
                    row["SchoolHoliday"], row["StoreType"], row["Assortment"],
                    row["CompetitionDistance"], row["CompetitionOpenSinceMonth"],
                    row["CompetitionOpenSinceYear"], row["Promo2"], row["Promo2SinceWeek"],
                    row["Promo2SinceYear"], row["PromoInterval"]
                )
                for _, row in data.iterrows()
            ]

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

    print(f"Loaded {len(records)} rows into `history`")


if __name__ == "__main__":
    main()
