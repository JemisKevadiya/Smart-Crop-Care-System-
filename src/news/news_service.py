"""Live agriculture news for Indian farmers, read from publishers' public RSS feeds.

    report = get_farmer_news()            # NewsReport: items (newest first) + per-feed errors
    report.items[0].title, .source, .published, .summary, .url, .category

Sources: the agriculture sections of Indian news publishers, through the RSS feeds
they publish for readers (no scraping, no API key). The feed list is configurable
with NEWS_FEEDS in .env. Every headline comes from a feed; nothing is generated.

`category` is the topic this module assigns from keywords in the headline and
summary (see TOPICS); `source_category` is the feed's own category, if it has one.
"""

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests

from src.env import get_setting

TIMEOUT_SECONDS = 10
MAX_FEED_BYTES = 5_000_000
MAX_AGE_DAYS = 30           # older items are not "current" news
MAX_SUMMARY_CHARS = 300
USER_AGENT = "SmartCropCare/1.0 (student project; agriculture news reader)"
UNAVAILABLE = "Live farmer news is temporarily unavailable. Please try again later."
NO_NEWS = "No recent agriculture news available."


@dataclass(frozen=True)
class Feed:
    publisher: str
    url: str
    agriculture_only: bool   # True: the feed is an agriculture section, every item is kept


# Agriculture sections of Indian publishers. Their robots.txt allows these feed paths.
DEFAULT_FEEDS = (
    Feed("The Economic Times",
         "https://economictimes.indiatimes.com/news/economy/agriculture/rssfeeds/1202099874.cms",
         agriculture_only=True),
    Feed("The Hindu BusinessLine",
         "https://www.thehindubusinessline.com/economy/agri-business/feeder/default.rss",
         agriculture_only=True),
    Feed("Krishi Jagran", "https://krishijagran.com/feeds/rss/", agriculture_only=False),
)

# Words that make an item relevant to farmers (used for mixed feeds such as Krishi Jagran).
FARMER_KEYWORDS = (
    "agri", "agriculture", "agricultural", "farming", "farmer", "farmers", "farm", "crop",
    "crops", "kharif", "rabi", "sowing", "harvest", "fertiliser", "fertilizer", "urea", "dap",
    "pesticide", "insecticide", "fungicide", "irrigation", "soil", "msp", "mandi", "kisan",
    "krishi", "horticulture", "seed", "seeds", "paddy", "wheat", "rice", "pulses", "cotton",
    "sugarcane", "dairy", "livestock", "monsoon", "tractor", "agritech", "procurement",
    "icar", "imd", "pm-kisan", "food grain", "foodgrain", "oilseed", "oilseeds", "vegetable",
    "vegetables", "fruit", "fruits", "plantation", "organic", "millet", "millets", "spice",
)

# Topic -> keywords. The first topic with a match wins; otherwise "Agriculture".
TOPICS = {
    "Government & Schemes": ("scheme", "yojana", "pm-kisan", "pm kisan", "subsidy", "ministry",
                             "government", "govt", "policy", "budget", "cabinet", "minister",
                             "insurance", "pmfby", "loan waiver", "credit card"),
    "Market & MSP": ("msp", "mandi", "price", "prices", "market", "procurement", "export",
                     "exports", "import", "imports", "trade", "rate", "rates", "apmc",
                     "commodity", "futures", "auction"),
    "Crop & Disease": ("disease", "pest", "pests", "blight", "fungus", "fungal", "infestation",
                       "locust", "virus", "wilt", "rust", "rot", "armyworm", "whitefly",
                       "bollworm", "pesticide", "fungicide", "insecticide", "fertiliser",
                       "fertilizer", "urea", "dap", "yield", "sowing", "seed", "seeds"),
    "Weather & Agriculture": ("monsoon", "rain", "rains", "rainfall", "imd", "weather",
                              "drought", "flood", "floods", "heatwave", "heat", "cyclone",
                              "frost", "temperature", "climate", "dry spell", "thunderstorm",
                              "thunderstorms", "hailstorm"),
    "Farming Technology": ("technology", "tech", "drone", "drones", "digital", "app", "ai",
                           "artificial intelligence", "startup", "startups", "innovation",
                           "precision", "sensor", "satellite", "machinery", "tractor",
                           "platform", "research", "icar"),
}
DEFAULT_TOPIC = "Agriculture"
ALL_TOPICS = (DEFAULT_TOPIC, *TOPICS)


class NewsError(Exception):
    """Base class; str(error) is a message suitable for showing to the user."""


class NewsUnavailableError(NewsError):
    """No feed could be read."""


@dataclass
class NewsItem:
    title: str
    source: str
    published: datetime | None
    summary: str
    url: str
    category: str                       # topic assigned from keywords (see TOPICS)
    source_category: str | None = None  # the feed's own category, if any

    def to_dict(self, summary_chars=MAX_SUMMARY_CHARS):
        return {"title": self.title, "source": self.source,
                "published": self.published.isoformat() if self.published else None,
                "summary": _shorten(self.summary, summary_chars), "url": self.url,
                "category": self.category}


@dataclass
class NewsReport:
    items: list[NewsItem]
    errors: dict[str, str] = field(default_factory=dict)    # publisher -> problem
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --- configuration ---------------------------------------------------------------------

def get_feeds():
    """DEFAULT_FEEDS, or NEWS_FEEDS from .env: "Publisher|url|all" entries separated by ;

    The third part is optional: "all" keeps every item, otherwise items are filtered by
    farmer keywords.
    """
    value = get_setting("NEWS_FEEDS")
    if not value:
        return DEFAULT_FEEDS
    feeds = []
    for entry in value.split(";"):
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) >= 2 and parts[0] and _is_web_url(parts[1]):
            feeds.append(Feed(parts[0], parts[1], len(parts) > 2 and parts[2].lower() == "all"))
    return tuple(feeds) or DEFAULT_FEEDS


# --- text helpers ----------------------------------------------------------------------

def _is_web_url(url):
    parsed = urlparse(url or "")
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _clean(text):
    """Plain text from feed HTML: tags removed, entities decoded, whitespace collapsed."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text).replace("�", "'")
    return re.sub(r"\s+", " ", text).strip()


def _shorten(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "..."


def _has_word(text, words):
    """True if any of `words` appears in `text` as a whole word (case-insensitive)."""
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def is_farmer_news(title, summary):
    return _has_word(f"{title} {summary}", FARMER_KEYWORDS)


def classify(title, summary):
    """Topic for an item: the first TOPICS entry whose keywords appear, else Agriculture.

    The headline is checked first, so it decides when headline and summary disagree.
    """
    for text in (title, f"{title} {summary}"):
        for topic, words in TOPICS.items():
            if _has_word(text, words):
                return topic
    return DEFAULT_TOPIC


def crop_terms(crop):
    """Search words for a crop name from the model's class list.

    "Corn (maize)" -> corn, maize; "Pepper, bell" -> bell pepper, pepper;
    "Cherry (including sour)" -> cherry. No synonyms are invented.
    """
    if not crop:
        return []
    base, _, extra = crop.partition("(")
    terms = []
    if "," in base:
        name, _, kind = base.partition(",")
        terms += [f"{kind.strip()} {name.strip()}", name.strip()]
    else:
        terms.append(base.strip())
    extra = extra.rstrip(")").strip()
    if extra and not extra.lower().startswith("including"):
        terms.append(extra)
    return [t.lower() for t in terms if t]


def mentions_crop(item, crop):
    terms = crop_terms(crop)
    if not terms:
        return False
    # Plurals: tomato -> tomatoes, cherry -> cherries, apple -> apples.
    forms = []
    for t in terms:
        forms += [t, t + "s", t + "es"] + ([t[:-1] + "ies"] if t.endswith("y") else [])
    return _has_word(f"{item.title} {item.summary}", forms)


def prioritize(items, crop):
    """Items that mention the crop first (order otherwise kept)."""
    if not crop_terms(crop):
        return list(items)
    return sorted(items, key=lambda item: not mentions_crop(item, crop))


def headlines_for_context(items, crop=None, limit=10, crop_items=3):
    """A short, varied selection for the assistant (items are already newest first).

    Up to `crop_items` items about the crop, then the newest item of every topic, then
    the newest of the rest, so a question about any topic can use a real headline.
    """
    chosen = [i for i in items if crop and mentions_crop(i, crop)][:crop_items]
    for topic in ALL_TOPICS:
        first = next((i for i in items if i.category == topic), None)
        if first is not None and first not in chosen:
            chosen.append(first)
    chosen += [i for i in items if i not in chosen]
    chosen = chosen[:limit]
    crop_first = prioritize(items, crop)
    return sorted(chosen, key=crop_first.index)


# --- fetching and parsing --------------------------------------------------------------

def _parse_date(text):
    if not text:
        return None
    try:
        value = parsedate_to_datetime(text.strip())
    except (TypeError, ValueError, IndexError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def parse_feed(content, feed):
    """NewsItems from RSS 2.0 bytes. Raises NewsError if the document is not an RSS feed."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise NewsError(f"{feed.publisher} sent a feed that could not be read.") from exc
    if root.find("channel") is None:
        raise NewsError(f"{feed.publisher} did not send an RSS feed.")
    items = []
    for node in root.iter("item"):
        title = _clean(node.findtext("title"))
        url = (node.findtext("link") or "").strip()
        if not title or not _is_web_url(url):
            continue
        summary = _clean(node.findtext("description"))
        if summary.lower() == title.lower():
            summary = ""
        source_category = _clean(node.findtext("category")) or None
        items.append(NewsItem(title=title, source=feed.publisher,
                              published=_parse_date(node.findtext("pubDate")),
                              summary=summary, url=url,
                              category=classify(title, summary),
                              source_category=source_category))
    return items


def fetch_feed(feed, timeout=TIMEOUT_SECONDS):
    try:
        response = requests.get(feed.url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.Timeout as exc:
        raise NewsError(f"{feed.publisher} did not respond within {timeout} seconds.") from exc
    except requests.ConnectionError as exc:
        raise NewsError(f"Could not connect to {feed.publisher}.") from exc
    except requests.RequestException as exc:
        raise NewsError(f"Request to {feed.publisher} failed.") from exc
    if not response.ok:
        raise NewsError(f"{feed.publisher} returned an error (HTTP {response.status_code}).")
    if len(response.content) > MAX_FEED_BYTES:
        raise NewsError(f"{feed.publisher} sent an unexpectedly large feed.")
    return parse_feed(response.content, feed)


def get_farmer_news(feeds=None, now=None, max_age_days=MAX_AGE_DAYS, limit=100):
    """Recent farmer-relevant news from all feeds, newest first, without duplicates.

    Feeds that fail are listed in report.errors; NewsUnavailableError is raised only if
    every feed fails. An empty item list means the feeds worked but had no recent news.
    """
    feeds = get_feeds() if feeds is None else feeds
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    items, errors = [], {}
    for feed in feeds:
        try:
            fetched = fetch_feed(feed)
        except NewsError as exc:
            errors[feed.publisher] = str(exc)
            continue
        items += [i for i in fetched
                  if (feed.agriculture_only or is_farmer_news(i.title, i.summary))
                  and (i.published is None or cutoff <= i.published <= now + timedelta(days=1))]
    if feeds and len(errors) == len(feeds):
        raise NewsUnavailableError(UNAVAILABLE)

    seen, unique = set(), []
    for item in items:
        key = re.sub(r"\W+", " ", item.title.lower()).strip()
        if key in seen or item.url in seen:
            continue
        seen.update((key, item.url))
        unique.append(item)
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    unique.sort(key=lambda i: i.published or oldest, reverse=True)
    return NewsReport(items=unique[:limit], errors=errors)
