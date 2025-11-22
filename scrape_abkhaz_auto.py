# scrape_abkhaz_auto.py
"""
Telegram bot for scraping abkhaz-auto.ru listings.

The bot responds to commands to configure filters (brand, model, mileage,
year, price, and keywords contained in the listing description) and returns
matches from the first four pages of the configured category.

Environment variable (optional):
    TELEGRAM_BOT_TOKEN - Token obtained from @BotFather. If not provided,
    DEFAULT_BOT_TOKEN below will be used.
"""

from __future__ import annotations

import json
import logging
import os
import asyncio
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from bs4 import BeautifulSoup

# APScheduler (используется внутри JobQueue) ожидает pytz-таймзону. Эта
# переменная окружения заставляет tzlocal возвращать pytz-объект, даже если
# система настроена на zoneinfo, до импорта JobQueue.
os.environ.setdefault("TZLOCAL_USE_DEPRECATED_PYTZ", "1")

_telegram_spec = importlib.util.find_spec("telegram")
_telegram_ext_spec = importlib.util.find_spec("telegram.ext")
if _telegram_spec is None or _telegram_ext_spec is None:
    raise SystemExit(
        "Требуется установить библиотеку python-telegram-bot (например, `pip install python-telegram-bot>=20`)."
    )

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue


# Base configuration
CATEGORY_URL = "https://abkhaz-auto.ru/category/188"
SEEN_FILE = Path("seen_listings.json")
PAGE_COUNT = 4  # base page + next 3 pages
DEFAULT_BOT_TOKEN = "8277337729:AAHgG2z4cet7VGJJlXnn57kbpoXapkj7Mw0"


# Logging setup
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


@dataclass
class FilterSettings:
    brand: Optional[str] = None
    model: Optional[str] = None
    max_mileage: Optional[int] = None
    min_year: Optional[int] = None
    max_price: Optional[int] = None
    description_keywords: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return "\n".join(
            [
                f"Марка: {self.brand or 'не задано'}",
                f"Модель: {self.model or 'не задано'}",
                f"Максимальный пробег: {self.max_mileage or 'не задано'}",
                f"Минимальный год: {self.min_year or 'не задано'}",
                f"Максимальная цена: {self.max_price or 'не задано'}",
                f"Ключевые слова в описании: {', '.join(self.description_keywords) if self.description_keywords else 'не задано'}",
            ]
        )


def load_seen() -> set[str]:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Не удалось загрузить seen_listings.json: %s", exc)
            return set()
    return set()


def save_seen(ids: Iterable[str]) -> None:
    SEEN_FILE.write_text(json.dumps(list(ids), ensure_ascii=False, indent=2), encoding="utf-8")


def build_page_url(page: int) -> str:
    return CATEGORY_URL if page <= 1 else f"{CATEGORY_URL}?page={page}"


def fetch_listings() -> list[dict]:
    listings: list[dict] = []
    for page in range(1, PAGE_COUNT + 1):
        url = build_page_url(page)
        LOGGER.info("Загрузка страницы %s", url)
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
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
            price = int("".join(ch for ch in price_text if ch.isdigit())) if any(ch.isdigit() for ch in price_text) else None
            year = int(year_text[:4]) if year_text[:4].isdigit() else None
            listings.append(
                {
                    "id": listing_id,
                    "title": title,
                    "url": href if href.startswith("http") else f"https://abkhaz-auto.ru{href}",
                    "price": price,
                    "year": year,
                }
            )
    return listings


def parse_detail_page(url: str) -> tuple[Optional[int], str]:
    """Return (mileage, description) from the listing page."""
    LOGGER.info("Загрузка объявления %s", url)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    mileage = _extract_mileage(text)
    description_el = soup.select_one(".description, .catalog__description, .advert__text")
    description = description_el.get_text(" ", strip=True) if description_el else text
    return mileage, description


def _extract_mileage(text: str) -> Optional[int]:
    lowered = text.lower()
    for marker in ["пробег", "км", "km"]:
        idx = lowered.find(marker)
        if idx != -1:
            start = max(0, idx - 12)
            window = text[start: idx]
            digits = "".join(ch for ch in window if ch.isdigit())
            if digits.isdigit():
                return int(digits)
    return None


def matches_filters(card: dict, filters: FilterSettings) -> bool:
    title = card.get("title", "")
    url = card.get("url", "")
    mileage, description = parse_detail_page(url)

    if filters.brand and filters.brand.lower() not in title.lower() and filters.brand.lower() not in description.lower():
        return False
    if filters.model and filters.model.lower() not in title.lower() and filters.model.lower() not in description.lower():
        return False
    if filters.max_price is not None and card.get("price") is not None and card["price"] > filters.max_price:
        return False
    if filters.min_year is not None and card.get("year") is not None and card["year"] < filters.min_year:
        return False
    if filters.max_mileage is not None and mileage is not None and mileage > filters.max_mileage:
        return False
    if filters.description_keywords:
        desc_lower = description.lower()
        if not all(keyword.lower() in desc_lower for keyword in filters.description_keywords):
            return False

    card["mileage"] = mileage
    card["description"] = description
    return True


def parse_filter_args(args: list[str]) -> FilterSettings:
    joined = " ".join(args)
    parts = [p.strip() for p in joined.split(";") if p.strip()]
    settings = FilterSettings()
    for part in parts:
        if "=" not in part:
            continue
        key, value = [segment.strip() for segment in part.split("=", 1)]
        if not value:
            continue
        if key == "brand":
            settings.brand = value
        elif key == "model":
            settings.model = value
        elif key == "max_mileage":
            if value.isdigit():
                settings.max_mileage = int(value)
        elif key == "min_year":
            if value.isdigit():
                settings.min_year = int(value)
        elif key == "max_price":
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits.isdigit():
                settings.max_price = int(digits)
        elif key == "keywords":
            settings.description_keywords = [word.strip() for word in value.split(",") if word.strip()]
    return settings


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я буду искать объявления на abkhaz-auto.ru.\n"
        "Настрой фильтры командой /filters, пример:\n"
        "``/filters brand=Toyota; model=Camry; max_mileage=150000; min_year=2010; max_price=1200000; keywords=левый руль,газ``\n"
        "Используй /search чтобы получить актуальные объявления."
    )


async def set_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = parse_filter_args(context.args)
    context.user_data["filters"] = settings
    await update.message.reply_text(
        "Фильтры обновлены:\n" + settings.describe()
    )


async def show_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: FilterSettings = context.user_data.get("filters") or FilterSettings()
    await update.message.reply_text("Текущие фильтры:\n" + settings.describe())


async def search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: FilterSettings = context.user_data.get("filters") or FilterSettings()
    await update.message.reply_text("Начинаю поиск объявлений...")
    seen = load_seen()
    listings = fetch_listings()
    matches: list[dict] = []
    for card in listings:
        if card["id"] in seen:
            continue
        try:
            if matches_filters(card, settings):
                matches.append(card)
                seen.add(card["id"])
        except Exception as exc:  # pragma: no cover - network parsing errors
            LOGGER.warning("Ошибка при обработке объявления %s: %s", card.get("url"), exc)
            continue
    save_seen(seen)
    if not matches:
        await update.message.reply_text("Ничего не найдено по текущим фильтрам.")
        return
    reply_lines = [f"Найдено {len(matches)} объявлений:"]
    for card in matches:
        line = f"{card['title']} ({card.get('year') or 'год?'}), цена: {card.get('price') or '—'}"
        if card.get("mileage"):
            line += f", пробег: {card['mileage']} км"
        line += f"\n{card['url']}"
        reply_lines.append(line)
    await update.message.reply_text("\n\n".join(reply_lines))


def run_bot() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or DEFAULT_BOT_TOKEN
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    job_queue = JobQueue()

    application = (
        Application.builder()
        .token(token)
        .job_queue(job_queue)
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("filters", set_filters))
    application.add_handler(CommandHandler("search", search))
    application.add_handler(CommandHandler("showfilters", show_filters))

    LOGGER.info("Бот запущен. Ожидание команд...")
    application.run_polling()


if __name__ == "__main__":
    run_bot()
