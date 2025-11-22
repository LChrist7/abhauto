# scrape_abkhaz_auto.py
import json
import time
from pathlib import Path
from typing import Iterable
import requests
from bs4 import BeautifulSoup

CATEGORY_URL = "https://abkhaz-auto.ru/category/188"
SEEN_FILE = Path("seen_listings.json")
POLL_SECONDS = 60  # 10 minutes

# Adjust filters as needed
FILTERS = {
    "keywords": ["Mitsubishi", "Airtek"],  # matched against title (case-insensitive); empty -> no keyword filter
    "max_price": 1500000,             # None -> no limit
    "min_year": 2000,                 # None -> no limit
}

def load_seen() -> set[str]:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()

def save_seen(ids: Iterable[str]) -> None:
    SEEN_FILE.write_text(json.dumps(list(ids), ensure_ascii=False, indent=2), encoding="utf-8")

def fetch_listings() -> list[dict]:
    resp = requests.get(CATEGORY_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    cards = []
    for item in soup.select(".catalog__item"):
        link = item.select_one("a")
        title_el = item.select_one(".catalog__title")
        price_el = item.select_one(".catalog__price")
        year_el = item.select_one(".catalog__year")
        if not link or not title_el:
            continue
        href = link.get("href") or ""
        listing_id = href.rstrip("/").split("/")[-1]
        title = title_el.get_text(strip=True)
        price_text = price_el.get_text(strip=True).replace(" ", "") if price_el else ""
        year_text = year_el.get_text(strip=True) if year_el else ""
        # crude parsing; adjust if site changes
        price = None
        for ch in price_text:
            if ch.isdigit():
                price = int("".join(c for c in price_text if c.isdigit()))
                break
        year = int(year_text[:4]) if year_text[:4].isdigit() else None
        cards.append(
            {
                "id": listing_id,
                "title": title,
                "url": href if href.startswith("http") else f"https://abkhaz-auto.ru{href}",
                "price": price,
                "year": year,
            }
        )
    return cards

def matches_filters(card: dict) -> bool:
    title = card["title"].lower()
    kws = FILTERS["keywords"]
    if kws and not any(kw.lower() in title for kw in kws):
        return False
    price = card["price"]
    if FILTERS["max_price"] is not None and price is not None and price > FILTERS["max_price"]:
        return False
    year = card["year"]
    if FILTERS["min_year"] is not None and year is not None and year < FILTERS["min_year"]:
        return False
    return True

def main():
    seen = load_seen()
    while True:
        try:
            listings = fetch_listings()
            new_cards = [c for c in listings if c["id"] not in seen and matches_filters(c)]
            if new_cards:
                print(f"Found {len(new_cards)} new matching listings:")
                for c in new_cards:
                    print(f"- {c['title']} ({c.get('year')}) | {c.get('price')} | {c['url']}")
                seen.update(c["id"] for c in new_cards)
                save_seen(seen)
            else:
                print("No new matching listings.")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
