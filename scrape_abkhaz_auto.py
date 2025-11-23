# scrape_abkhaz_auto.py
"""
Telegram bot for scraping abkhaz-auto.ru listings.

The bot responds to commands to configure filters (brand, model, mileage,
    year, price, and keywords contained in the listing description) and returns
    matches from the configured number of pages in the category.

Environment variable (optional):
    TELEGRAM_BOT_TOKEN - Token obtained from @BotFather. If not provided,
    DEFAULT_BOT_TOKEN below will be used.
"""

from __future__ import annotations

import os

# APScheduler (используется внутри JobQueue) ожидает pytz-таймзону. Эта
# переменная окружения должна быть установлена максимально рано, чтобы tzlocal
# возвращал pytz-объект даже при конфигурации zoneinfo.
os.environ.setdefault("TZLOCAL_USE_DEPRECATED_PYTZ", "1")

import asyncio
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import pytz
import requests
import tzlocal
from apscheduler import util as apscheduler_util
from bs4 import BeautifulSoup
import re

# Принудительно заставляем tzlocal и сам APScheduler возвращать pytz-таймзону до
# инициализации JobQueue внутри Application.builder(), чтобы исключить ошибку
# «Only timezones from the pytz library are supported».
tzlocal.get_localzone = lambda: pytz.UTC  # type: ignore[assignment]
apscheduler_util.get_localzone = lambda: pytz.UTC  # type: ignore[assignment]


# APScheduler по умолчанию вызывает util.astimezone, которое в новых версиях
# tzlocal может вернуть zoneinfo-таймзону. При попытке преобразования APScheduler
# выбрасывает TypeError («Only timezones from the pytz library are supported»).
# Оборачиваем astimezone так, чтобы в подобных случаях возвращался pytz.UTC.
_original_astimezone = apscheduler_util.astimezone


def _astimezone_compat(timezone):  # type: ignore[override]
    try:
        return _original_astimezone(timezone)
    except TypeError:
        return pytz.UTC


apscheduler_util.astimezone = _astimezone_compat  # type: ignore[assignment]

_telegram_spec = importlib.util.find_spec("telegram")
_telegram_ext_spec = importlib.util.find_spec("telegram.ext")
if _telegram_spec is None or _telegram_ext_spec is None:
    raise SystemExit(
        "Требуется установить библиотеку python-telegram-bot (например, `pip install python-telegram-bot>=20`)."
    )

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from telegram.error import TimedOut


# Base configuration
CATEGORY_URL = "https://abkhaz-auto.ru/category/188"
SEEN_FILE = Path("seen_listings.json")
DEFAULT_PAGE_COUNT = 4  # base page + next 3 pages
DEFAULT_BOT_TOKEN = "8277337729:AAHgG2z4cet7VGJJlXnn57kbpoXapkj7Mw0"

# Conversation states
(
    CHOOSING,
    TYPING_VALUE,
) = range(2)


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
    min_mileage: Optional[int] = None
    max_mileage: Optional[int] = None
    min_year: Optional[int] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    description_keywords: List[str] = field(default_factory=list)
    page_count: int = DEFAULT_PAGE_COUNT

    def describe(self) -> str:
        return "\n".join(
            [
                f"Марка: {self.brand or 'не задано'}",
                f"Модель: {self.model or 'не задано'}",
                f"Пробег: {self.min_mileage or '-'} — {self.max_mileage or '-'}",
                f"Минимальный год: {self.min_year or 'не задано'}",
                f"Цена: {self.min_price or '-'} — {self.max_price or '-'}",
                f"Объём двигателя: {self.min_volume or '-'} — {self.max_volume or '-'}",
                f"Ключевые слова в описании: {', '.join(self.description_keywords) if self.description_keywords else 'не задано'}",
                f"Страниц для поиска: {self.page_count}",
            ]
        )


def _ensure_absolute_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    if url.startswith("http"):
        return url
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return f"https://abkhaz-auto.ru{url}"
    return url


async def _safe_reply_text(message, text: str, **kwargs):
    """Отправляет сообщение с повторной попыткой при тайм-ауте."""

    if not message:
        return None
    try:
        return await message.reply_text(text, **kwargs)
    except TimedOut:
        LOGGER.warning("Таймаут при отправке сообщения, повторяем с более высоким таймаутом")
        try:
            return await message.reply_text(text, **kwargs)
        except TimedOut:
            LOGGER.error("Не удалось отправить сообщение из-за тайм-аута Telegram")
            return None
    except Exception as exc:  # pragma: no cover - сетевые сбои
        LOGGER.error("Ошибка при отправке сообщения: %s", exc)
        return None


async def _safe_reply_photo(message, photo_url: str, caption: str) -> None:
    """Безопасно отправляет фото с подписью, с повторной попыткой при тайм-ауте."""

    if not message:
        return
    try:
        await message.reply_photo(photo=photo_url, caption=caption)
        return
    except TimedOut:
        LOGGER.warning("Таймаут при отправке фото, повторяем")
        try:
            await message.reply_photo(photo=photo_url, caption=caption)
            return
        except TimedOut:
            LOGGER.error("Не удалось отправить фото из-за тайм-аута Telegram")
            return
    except Exception as exc:  # pragma: no cover - сетевые сбои
        LOGGER.error("Ошибка при отправке фото: %s", exc)
        return


async def _reply_in_chunks(message, lines: list[str], chunk_size: int = 8) -> None:
    """Делит длинный ответ на части, чтобы снизить риск тайм-аутов."""

    for start in range(0, len(lines), chunk_size):
        chunk = lines[start : start + chunk_size]
        await _safe_reply_text(message, "\n\n".join(chunk))


def _filter_signature(filters: FilterSettings) -> str:
    """Возвращает стабильный ключ для набора фильтров (для кеша seen)."""

    return json.dumps(
        {
            "brand": _normalize_for_match(filters.brand or ""),
            "model": _normalize_for_match(filters.model or ""),
            "min_mileage": filters.min_mileage,
            "max_mileage": filters.max_mileage,
            "min_year": filters.min_year,
            "min_price": filters.min_price,
            "max_price": filters.max_price,
            "min_volume": filters.min_volume,
            "max_volume": filters.max_volume,
            "keywords": sorted(
                [_normalize_for_match(kw) for kw in filters.description_keywords if kw.strip()]
            ),
            "page_count": filters.page_count,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def load_seen() -> dict[str, set[str]]:
    """Загружает карту {signature: set(ids)} для разных наборов фильтров."""

    if SEEN_FILE.exists():
        try:
            raw = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
            # Поддержка старого формата (просто список ID без привязки к фильтрам)
            if isinstance(raw, list):
                return {}
            if isinstance(raw, dict):
                return {key: set(value) for key, value in raw.items() if isinstance(value, list)}
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Не удалось загрузить seen_listings.json: %s", exc)
    return {}


def save_seen(seen_map: dict[str, Iterable[str]]) -> None:
    serializable = {key: list(ids) for key, ids in seen_map.items()}
    SEEN_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Марка", callback_data="set_brand"),
                InlineKeyboardButton("Модель", callback_data="set_model"),
            ],
            [
                InlineKeyboardButton("Пробег", callback_data="set_mileage"),
                InlineKeyboardButton("Мин. год", callback_data="set_year"),
            ],
            [
                InlineKeyboardButton("Цена", callback_data="set_price"),
                InlineKeyboardButton("Ключевые слова", callback_data="set_keywords"),
            ],
            [InlineKeyboardButton("Объём", callback_data="set_volume")],
            [InlineKeyboardButton("Страницы поиска", callback_data="set_pages")],
            [InlineKeyboardButton("Показать фильтры", callback_data="show_filters")],
            [InlineKeyboardButton("Запустить поиск", callback_data="run_search")],
        ]
    )


def build_page_url(page: int) -> str:
    return CATEGORY_URL if page <= 1 else f"{CATEGORY_URL}?page={page}"


def _extract_brand_model_from_title(title: str) -> tuple[Optional[str], Optional[str]]:
    """Выделяет марку и модель из заголовка объявления."""

    if not title:
        return None, None
    parts = title.split(maxsplit=1)
    brand = parts[0] if parts else None
    model = parts[1] if len(parts) > 1 else None
    return brand, model


def _normalized(value: Optional[str]) -> Optional[str]:
    """Возвращает нормализованную строку или None."""

    if not value:
        return None
    normalized = _normalize_for_match(value)
    return normalized or None


def _parse_listing_item(item) -> Optional[dict]:
    """Извлекает данные объявления прямо со страницы списка.

    На абхаз-авто объявления отображаются в блоках вида:
    <li id="box_nc_...">
      <div class="allad_h">Toyota Crown</div>
      <p>2007 г., пробег: 1 км...</p>
      <div class="allad_info">...</div>
    </li>
    Функция пытается собрать марку, модель, год, пробег, цену и краткое
    описание без перехода на детальную страницу.
    """

    detail_link = item.find("a", string=re.compile("подробнее", re.IGNORECASE))
    link = detail_link or item.select_one("a[href]")
    title_el = item.select_one(".catalog__title") or item.select_one(".allad_h") or link
    if not link or not title_el:
        return None

    href = link.get("href") or ""
    listing_id = href.rstrip("/").split("/")[-1]
    title = title_el.get_text(strip=True)

    def parse_price(text: str) -> Optional[int]:
        # Цена на странице обычно обёрнута в catalog__price. Если блок не найден,
        # пробуем вытащить число рядом со словами «руб»/«₽» или «цена».
        # Значения, похожие на год (4 цифры) или слишком маленькие суммы отбрасываем,
        # чтобы не путать стоимость с годом выпуска.
        patterns = [
            r"(?:цена[:\s]*)?([\d\s]{3,})(?=\s*(?:руб|₽|$))",
            r"([\d\s]{3,})",
        ]
        candidates: list[int] = []
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                digits = "".join(ch for ch in match.group(1) if ch.isdigit())
                if digits:
                    value = _parse_int(digits, scale_small=True)
                    # Отбрасываем очевидные годы (до 2100) и слишком маленькие суммы
                    # — в каталоге машины стоят дороже ~10 000.
                    if value is not None and value >= 10_000 and value > 2100:
                        candidates.append(value)
        if candidates:
            # Берём последнее подходящее число, чтобы игнорировать год/пробег в начале строки.
            return candidates[-1]
        return None

    price_el = item.select_one(".catalog__price")
    year_el = item.select_one(".catalog__year")
    description_el = item.select_one(".catalog__description, .description, .allad_info")
    info_text = " ".join(item.select_one("p").stripped_strings) if item.select_one("p") else ""
    block_text = " ".join(item.stripped_strings)

    image_el = (
        item.select_one(".catalog__image img, .catalog__img img, img")
        or item.select_one("img")
    )
    image_url = None
    if image_el:
        for attr in ("data-src", "data-original", "src"):
            candidate = image_el.get(attr)
            if candidate:
                image_url = _ensure_absolute_url(candidate)
                break

    price_source = price_el.get_text(" ", strip=True) if price_el else ""
    if not price_source:
        price_source = info_text or block_text
    price = parse_price(price_source)

    year_text = year_el.get_text(strip=True) if year_el else ""
    if not year_text:
        year_match = re.search(r"\b(19|20)\d{2}\b", info_text or block_text)
        year_text = year_match.group(0) if year_match else ""
    year = int(year_text[:4]) if year_text[:4].isdigit() else None

    mileage_text = _extract_mileage_raw(info_text or block_text)
    mileage = _parse_int(mileage_text, scale_small=True)

    description = description_el.get_text(" ", strip=True) if description_el else ""
    if not description:
        description = info_text

    brand, model = _extract_brand_model_from_title(title)
    brand_norm = _normalized(brand)
    model_norm = _normalized(model)

    volume = _extract_volume(block_text)

    return {
        "id": listing_id,
        "title": title,
        "url": href if href.startswith("http") else f"https://abkhaz-auto.ru{href}",
        "price": price,
        "year": year,
        "mileage": mileage,
        "description": description,
        "brand": brand,
        "model": model,
        "brand_norm": brand_norm,
        "model_norm": model_norm,
        "image_url": image_url,
        "volume": volume,
    }


def fetch_listings(page_count: int = DEFAULT_PAGE_COUNT) -> list[dict]:
    listings: list[dict] = []
    seen_ids: set[str] = set()
    effective_pages = page_count if page_count > 0 else DEFAULT_PAGE_COUNT
    for page in range(1, effective_pages + 1):
        url = build_page_url(page)
        LOGGER.info("Загрузка страницы %s", url)
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Поддержка старых и новых версток каталога.
        item_selectors = [".catalog__item", "li[id^='box_']", ".allad"]
        for selector in item_selectors:
            for item in soup.select(selector):
                parsed = _parse_listing_item(item)
                if parsed and parsed["id"] not in seen_ids:
                    listings.append(parsed)
                    seen_ids.add(parsed["id"])
            if listings:
                # Если уже нашли объявления по одному из селекторов, не продолжаем
                # перебирать остальные (чтобы избежать дубликатов).
                break
    return listings


def parse_detail_page(url: str) -> dict:
    """Возвращает детали объявления, извлекая бренд, модель, год и пробег.

    Страница может содержать данные как в блоках параметров, так и в тексте.
    Функция ищет значения по подписи (Марка, Модель, Год, Пробег) и возвращает
    словарь вида {brand, model, year, mileage, description}.
    """

    LOGGER.info("Загрузка объявления %s", url)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    def text_lines() -> list[str]:
        raw_text = soup.get_text("\n", strip=True)
        return [line.strip() for line in raw_text.splitlines() if line.strip()]

    def by_label(labels: list[str]) -> Optional[str]:
        lines = text_lines()
        for line in lines:
            lower_line = line.lower()
            for label in labels:
                if lower_line.startswith(label):
                    after = line[len(label):].strip(" -:\t")
                    if ":" in line:
                        after = line.split(":", 1)[1].strip()
                    if after:
                        return after
        return None

    description_el = soup.select_one(
        "p.post_body, .description, .catalog__description, .advert__text, [itemprop='description']"
    )
    description = description_el.get_text(" ", strip=True) if description_el else ""

    price_el = soup.select_one(
        ".advert__price, .price, [itemprop='price'], .catalog__price, .price-value"
    )
    price_text = price_el.get_text(" ", strip=True) if price_el else ""
    if not price_text:
        # Иногда цена прописана словами «Цена: ...» рядом с описанием
        price_match = re.search(r"цена[:\s]*([\d\s]{3,})", soup.get_text(" ", strip=True), re.IGNORECASE)
        price_text = price_match.group(1) if price_match else ""
    price = _parse_int(price_text, scale_small=True) if price_text else None
    if price is not None and price < 10_000:
        price = None

    image_el = soup.select_one(
        ".advert__photo img, .advert__img img, .post-photo img, .slider img, img[itemprop='image']"
    )
    if not image_el:
        og_image = soup.find("meta", property="og:image")
        image_url = _ensure_absolute_url(og_image["content"]) if og_image and og_image.get("content") else None
    else:
        image_url = None
        for attr in ("data-src", "data-original", "src"):
            candidate = image_el.get(attr)
            if candidate:
                image_url = _ensure_absolute_url(candidate)
                break

    # Попытка достать поля из таблицы параметров (часто .advert__info или .params)
    brand = by_label(["марка", "бренд"])
    model = by_label(["модель"])
    year_text = by_label(["год", "год выпуска", "выпуск"])
    mileage_text = by_label(["пробег"])

    text_content = " ".join(text_lines())
    if not mileage_text:
        mileage_text = _extract_mileage_raw(text_content)

    year = int(re.search(r"\b(19|20)\d{2}\b", year_text).group(0)) if year_text and re.search(r"\b(19|20)\d{2}\b", year_text) else None
    mileage = _parse_int(mileage_text, scale_small=True)

    if not description:
        description = text_content

    volume = None
    engine_line = by_label(["двигатель"])
    if engine_line:
        volume = _extract_volume(engine_line)
    if volume is None:
        volume = _extract_volume(text_content)

    return {
        "brand": brand,
        "model": model,
        "year": year,
        "mileage": mileage,
        "description": description,
        "brand_norm": _normalized(brand),
        "model_norm": _normalized(model),
        "image_url": image_url,
        "price": price,
        "volume": volume,
    }


def _extract_mileage(text: str) -> Optional[int]:
    lowered = text.lower()
    for marker in ["пробег", "км", "km"]:
        idx = lowered.find(marker)
        if idx != -1:
            start = max(0, idx - 12)
            window = text[start: idx]
            digits = "".join(ch for ch in window if ch.isdigit())
            if digits.isdigit():
                return _parse_int(digits, scale_small=True)
    return None


def _extract_mileage_raw(text: str) -> Optional[str]:
    lowered = text.lower()
    for marker in ["пробег", "км", "km"]:
        idx = lowered.find(marker)
        if idx != -1:
            start = max(0, idx - 20)
            window = text[start:idx]
            digits = "".join(ch for ch in window if ch.isdigit())
            if digits:
                return digits
    return None


def _parse_int(value: Optional[str], *, scale_small: bool = False) -> Optional[int]:
    """Парсит число из строки.

    Если включён ``scale_small`` и количество цифр меньше 4, число умножается
    на 1000 (для обработки компактных значений цены/пробега вроде "123").
    """

    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits.isdigit():
        return None

    number = int(digits)
    if scale_small and len(digits) < 4:
        number *= 1000
    return number


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Парсит число с плавающей точкой из строки, используя точку или запятую."""

    if not value:
        return None
    match = re.search(r"\d+(?:[\.,]\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _extract_volume(text: str) -> Optional[float]:
    """Извлекает объём двигателя (в литрах) из произвольного текста."""

    if not text:
        return None
    match = re.search(r"(\d+[\.,]?\d*)\s*л", text, re.IGNORECASE)
    if match:
        return _parse_float(match.group(1))
    # Иногда объём следует после слова «двигатель» без «л»
    engine_match = re.search(r"двигатель[^\d]*(\d+[\.,]?\d*)", text, re.IGNORECASE)
    if engine_match:
        return _parse_float(engine_match.group(1))
    return None


def _parse_range_int(text: str, *, scale_small: bool = False) -> tuple[Optional[int], Optional[int]]:
    """Парсит диапазон целых чисел формата "мин-макс"."""

    if not text:
        return None, None
    parts = re.split(r"[-–]", text)
    min_value = _parse_int(parts[0], scale_small=scale_small) if parts else None
    max_value = _parse_int(parts[1], scale_small=scale_small) if len(parts) > 1 else None
    return min_value, max_value


def _parse_range_float(text: str) -> tuple[Optional[float], Optional[float]]:
    """Парсит диапазон чисел с плавающей точкой формата "мин-макс"."""

    if not text:
        return None, None
    parts = re.split(r"[-–]", text)
    min_value = _parse_float(parts[0]) if parts else None
    max_value = _parse_float(parts[1]) if len(parts) > 1 else None
    return min_value, max_value


def _normalize_for_match(text: str) -> str:
    """Упрощённая нормализация для поиска по латинице/кириллице.

    Переводит кириллицу в приближенную латиницу, убирает лишние
    символы и приводит к нижнему регистру, чтобы запрос "Toyota"
    совпадал с "Тойота" в заголовках и описаниях.
    """

    translit = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "c",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }

    normalized = []
    for ch in text.lower():
        mapped = translit.get(ch, ch)
        if mapped.isalnum() or mapped.isspace():
            normalized.append(mapped)
    return " ".join("".join(normalized).split())


def _match_value(candidate: Optional[str], needle: str) -> bool:
    """Сравнивает значение с фильтром, нормализуя текст."""

    if not candidate:
        return False
    candidate_norm = _normalize_for_match(candidate)
    needle_norm = _normalize_for_match(needle)
    if not candidate_norm or not needle_norm:
        return False
    return needle_norm in candidate_norm


def matches_filters(card: dict, filters: FilterSettings) -> bool:
    """
    Проверяет объявление по заданным фильтрам.

    Детали объявления загружаются только при необходимости (пробег,
    ключевые слова или если марка/модель не найдены в заголовке), что
    позволяет находить совпадения по марке даже при ошибках парсинга
    детальной страницы.
    """

    title = card.get("title", "")
    description = card.get("description", "") or ""
    mileage: Optional[int] = card.get("mileage")
    detail_brand: Optional[str] = card.get("brand")
    detail_model: Optional[str] = card.get("model")
    detail_brand_norm: Optional[str] = card.get("brand_norm")
    detail_model_norm: Optional[str] = card.get("model_norm")
    detail_year: Optional[int] = card.get("year")
    detail_price: Optional[int] = card.get("price")
    detail_volume: Optional[float] = card.get("volume")
    image_url: Optional[str] = card.get("image_url")
    title_brand, title_model = _extract_brand_model_from_title(title)
    fetched_details = False

    def ensure_details(force: bool = False) -> None:
        nonlocal description, mileage, detail_brand, detail_model, detail_year, detail_price, detail_brand_norm, detail_model_norm, image_url, fetched_details, detail_volume
        if fetched_details:
            return
        already_have_core = (
            description
            or mileage is not None
            or detail_brand is not None
            or detail_model is not None
            or detail_year is not None
            or detail_price is not None
            or detail_volume is not None
        )
        if not force and already_have_core:
            return
        try:
            details = parse_detail_page(card.get("url", ""))
            mileage = details.get("mileage", mileage)
            description = details.get("description") or description
            detail_brand = details.get("brand") or detail_brand
            detail_model = details.get("model") or detail_model
            detail_brand_norm = details.get("brand_norm") or detail_brand_norm
            detail_model_norm = details.get("model_norm") or detail_model_norm
            detail_year = details.get("year") if details.get("year") is not None else detail_year
            detail_price = details.get("price") if details.get("price") is not None else detail_price
            detail_volume = details.get("volume") if details.get("volume") is not None else detail_volume
            image_url = image_url or details.get("image_url")
            if detail_price is not None:
                card["price"] = detail_price
            if detail_volume is not None:
                card["volume"] = detail_volume
            fetched_details = True
        except Exception as exc:  # pragma: no cover - сетевые ошибки
            LOGGER.warning(
                "Не удалось загрузить детали объявления %s: %s",
                card.get("url"),
                exc,
            )
            fetched_details = True

    if filters.brand:
        ensure_details()
        brand_filter = _normalized(filters.brand)
        ensure_details()
        brand_candidates = [
            card.get("brand_norm") or _normalized(card.get("brand")),
            _normalized(title_brand),
            detail_brand_norm or _normalized(detail_brand),
        ]
        if brand_filter and not any(
            candidate and brand_filter in candidate for candidate in brand_candidates
        ):
            ensure_details(force=True)
            brand_candidates = [
                card.get("brand_norm") or _normalized(card.get("brand")),
                _normalized(title_brand),
                detail_brand_norm or _normalized(detail_brand),
            ]
            if not any(
                candidate and brand_filter in candidate for candidate in brand_candidates
            ):
                return False

    if filters.model:
        ensure_details()
        model_filter = _normalized(filters.model)
        ensure_details()
        model_candidates = [
            card.get("model_norm") or _normalized(card.get("model")),
            _normalized(title_model),
            detail_model_norm or _normalized(detail_model),
        ]
        if model_filter and not any(
            candidate and model_filter in candidate for candidate in model_candidates
        ):
            ensure_details(force=True)
            model_candidates = [
                card.get("model_norm") or _normalized(card.get("model")),
                _normalized(title_model),
                detail_model_norm or _normalized(detail_model),
            ]
            if not any(
                candidate and model_filter in candidate for candidate in model_candidates
            ):
                return False

    if filters.min_price is not None or filters.max_price is not None:
        if card.get("price") is None:
            ensure_details(force=True)
        price_value = card.get("price") if card.get("price") is not None else detail_price
        if price_value is not None:
            if filters.min_price is not None and price_value < filters.min_price:
                return False
            if filters.max_price is not None and price_value > filters.max_price:
                return False
    if filters.min_year is not None:
        effective_year = detail_year if detail_year is not None else card.get("year")
        if effective_year is None:
            ensure_details(force=True)
            effective_year = detail_year if detail_year is not None else card.get("year")
        if effective_year is not None and effective_year < filters.min_year:
            return False

    if filters.min_mileage is not None or filters.max_mileage is not None:
        if mileage is None:
            ensure_details(force=True)
        if mileage is not None:
            if filters.min_mileage is not None and mileage < filters.min_mileage:
                return False
            if filters.max_mileage is not None and mileage > filters.max_mileage:
                return False

    if filters.min_volume is not None or filters.max_volume is not None:
        if detail_volume is None:
            ensure_details(force=True)
        volume_value = card.get("volume") if card.get("volume") is not None else detail_volume
        if volume_value is not None:
            if filters.min_volume is not None and volume_value < filters.min_volume:
                return False
            if filters.max_volume is not None and volume_value > filters.max_volume:
                return False

    if filters.description_keywords:
        ensure_details(force=True)
        for keyword in filters.description_keywords:
            if not _match_value(description, keyword):
                return False

    card["mileage"] = mileage
    card["description"] = description
    if detail_brand or title_brand:
        card["brand"] = detail_brand or title_brand
        card["brand_norm"] = _normalized(card["brand"])
    if detail_model or title_model:
        card["model"] = detail_model or title_model
        card["model_norm"] = _normalized(card["model"])
    if detail_year is not None:
        card["year"] = detail_year
    if image_url:
        card["image_url"] = image_url
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
        elif key in {"mileage", "max_mileage", "min_mileage"}:
            min_m, max_m = _parse_range_int(value, scale_small=True)
            if min_m is not None:
                settings.min_mileage = min_m
            if max_m is not None:
                settings.max_mileage = max_m
        elif key == "min_year":
            if value.isdigit():
                settings.min_year = int(value)
        elif key in {"price", "max_price", "min_price"}:
            min_p, max_p = _parse_range_int(value, scale_small=True)
            if min_p is not None:
                settings.min_price = min_p
            if max_p is not None:
                settings.max_price = max_p
        elif key in {"volume", "engine", "engine_volume"}:
            min_v, max_v = _parse_range_float(value)
            if min_v is not None:
                settings.min_volume = min_v
            if max_v is not None:
                settings.max_volume = max_v
        elif key == "keywords":
            settings.description_keywords = [word.strip() for word in value.split(",") if word.strip()]
        elif key in {"pages", "page_count"}:
            if value.isdigit():
                settings.page_count = max(1, int(value))
    return settings


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.setdefault("filters", FilterSettings())
    await _safe_reply_text(
        update.message,
        "Привет! Я буду искать объявления на abkhaz-auto.ru.\n"
        "Можешь воспользоваться кнопками ниже для настройки фильтров или командой /filters.\n"
        "После выбора фильтров нажми «Запустить поиск». Можно указать, сколько страниц смотреть.",
        reply_markup=main_menu_keyboard(),
    )
    return CHOOSING


async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.setdefault("filters", FilterSettings())
    message = update.effective_message
    if message:
        await _safe_reply_text(
            message, "Выберите, что настроить:", reply_markup=main_menu_keyboard()
        )
    return CHOOSING


async def set_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = parse_filter_args(context.args)
    context.user_data["filters"] = settings
    await _safe_reply_text(update.message, "Фильтры обновлены:\n" + settings.describe())


async def show_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: FilterSettings = context.user_data.get("filters") or FilterSettings()
    message = update.effective_message
    if message:
        await _safe_reply_text(message, "Текущие фильтры:\n" + settings.describe())


async def search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: FilterSettings = context.user_data.get("filters") or FilterSettings()
    message = update.effective_message
    if not message:
        return
    await _safe_reply_text(message, "Начинаю поиск объявлений...")
    listings = fetch_listings(settings.page_count)
    matches: list[dict] = []
    for card in listings:
        try:
            if matches_filters(card, settings):
                matches.append(card)
        except Exception as exc:  # pragma: no cover - network parsing errors
            LOGGER.warning("Ошибка при обработке объявления %s: %s", card.get("url"), exc)
            continue
    if not matches:
        await _safe_reply_text(message, "Ничего не найдено по текущим фильтрам.")
        return
    await _safe_reply_text(
        message,
        f"Найдено {len(matches)} объявлений. Отправляю результаты по одному сообщению...",
    )
    for card in matches:
        caption = _format_listing(card)
        photo_url = card.get("image_url")
        if photo_url:
            await _safe_reply_photo(message, photo_url, caption)
        else:
            await _safe_reply_text(message, caption)


def _ensure_filter_storage(context: ContextTypes.DEFAULT_TYPE) -> FilterSettings:
    settings: FilterSettings = context.user_data.get("filters") or FilterSettings()
    context.user_data["filters"] = settings
    return settings


def _format_listing(card: dict) -> str:
    parts = [card.get("title") or "Объявление", card.get("url") or ""]
    year = card.get("year") or "год?"
    price = card.get("price") or "—"
    mileage = card.get("mileage")
    description = card.get("description") or ""
    details_lines = [f"Год: {year}", f"Цена: {price}"]
    if mileage is not None:
        details_lines.append(f"Пробег: {mileage} км")
    if description:
        details_lines.append(description[:400])
    parts.insert(1, "\n".join(details_lines))
    return "\n\n".join([p for p in parts if p])


async def handle_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query:
        return CHOOSING
    await query.answer()
    action = query.data or ""
    settings = _ensure_filter_storage(context)

    prompts = {
        "set_brand": "Введите марку (например, Toyota):",
        "set_model": "Введите модель (например, Camry):",
        "set_mileage": "Введите диапазон пробега, например 10000-165000:",
        "set_year": "Введите минимальный год выпуска (например, 2010):",
        "set_price": "Введите диапазон цены, например 400000-1500000:",
        "set_volume": "Введите диапазон объёма двигателя, например 1.8-4.7:",
        "set_keywords": "Введите ключевые слова через запятую (например, левый руль, газ):",
        "set_pages": "Введите количество страниц для поиска (например, 3):",
    }

    if action in prompts:
        context.user_data["pending_field"] = action
        await _safe_reply_text(query.message, prompts[action])
        return TYPING_VALUE

    if action == "show_filters":
        await show_filters(update, context)
        await _safe_reply_text(
            query.message,
            "Можешь продолжить настройку или запустить поиск:",
            reply_markup=main_menu_keyboard(),
        )
        return CHOOSING

    if action == "run_search":
        await search(update, context)
        await _safe_reply_text(
            query.message,
            "Продолжить настройку фильтров?",
            reply_markup=main_menu_keyboard(),
        )
        return CHOOSING

    # Неизвестное действие — показать меню
    await _safe_reply_text(
        query.message,
        "Неизвестная команда. Выберите действие на клавиатуре:",
        reply_markup=main_menu_keyboard(),
    )
    return CHOOSING


async def handle_value(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    pending_field = context.user_data.get("pending_field")
    settings = _ensure_filter_storage(context)
    text = (update.message.text or "").strip()

    if not pending_field:
        await _safe_reply_text(
            update.message,
            "Не удалось определить, что вы хотите изменить. Выберите действие на клавиатуре:",
            reply_markup=main_menu_keyboard(),
        )
        return CHOOSING

    if pending_field == "set_brand":
        settings.brand = text or None
        reply = f"Марка установлена: {settings.brand or 'не задано'}"
    elif pending_field == "set_model":
        settings.model = text or None
        reply = f"Модель установлена: {settings.model or 'не задано'}"
    elif pending_field == "set_mileage":
        min_m, max_m = _parse_range_int(text, scale_small=True)
        settings.min_mileage, settings.max_mileage = min_m, max_m
        reply = f"Пробег: {settings.min_mileage or '-'} — {settings.max_mileage or '-'}"
    elif pending_field == "set_year":
        settings.min_year = int(text) if text.isdigit() else None
        reply = f"Минимальный год: {settings.min_year or 'не задано'}"
    elif pending_field == "set_price":
        min_p, max_p = _parse_range_int(text, scale_small=True)
        settings.min_price, settings.max_price = min_p, max_p
        reply = f"Цена: {settings.min_price or '-'} — {settings.max_price or '-'}"
    elif pending_field == "set_volume":
        min_v, max_v = _parse_range_float(text)
        settings.min_volume, settings.max_volume = min_v, max_v
        reply = f"Объём двигателя: {settings.min_volume or '-'} — {settings.max_volume or '-'}"
    elif pending_field == "set_keywords":
        settings.description_keywords = [word.strip() for word in text.split(",") if word.strip()]
        reply = (
            "Ключевые слова: "
            + (", ".join(settings.description_keywords) if settings.description_keywords else "не задано")
        )
    elif pending_field == "set_pages":
        digits = "".join(ch for ch in text if ch.isdigit())
        settings.page_count = max(1, int(digits)) if digits else settings.page_count
        reply = f"Страниц для поиска: {settings.page_count}"
    else:
        reply = "Неизвестный фильтр"

    context.user_data.pop("pending_field", None)
    await _safe_reply_text(update.message, reply)
    await _safe_reply_text(
        update.message, "Далее?", reply_markup=main_menu_keyboard()
    )
    return CHOOSING


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Логирует исключения и информирует пользователя при сбое отправки."""

    LOGGER.error("Ошибка при обработке обновления", exc_info=context.error)
    if hasattr(update, "effective_message") and update.effective_message:
        await _safe_reply_text(
            update.effective_message,
            "Произошла ошибка при обращении к Telegram. Попробуйте ещё раз.",
        )


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

    application = Application.builder().token(token).build()

    conversation = ConversationHandler(
        entry_points=[CommandHandler("start", start), CommandHandler("menu", show_menu)],
        states={
            CHOOSING: [CallbackQueryHandler(handle_menu_choice)],
            TYPING_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_value)],
        },
        fallbacks=[CommandHandler("menu", show_menu), CommandHandler("start", start)],
        name="filter_conversation",
        persistent=False,
    )

    application.add_handler(conversation)
    application.add_handler(CommandHandler("filters", set_filters))
    application.add_handler(CommandHandler("search", search))
    application.add_handler(CommandHandler("showfilters", show_filters))
    application.add_error_handler(handle_error)

    LOGGER.info("Бот запущен. Ожидание команд...")
    application.run_polling()


if __name__ == "__main__":
    run_bot()
