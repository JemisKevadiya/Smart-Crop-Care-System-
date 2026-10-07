"""Live Farmer News panel, used on the dashboard and on the Farmer news page.

    show_news("dashboard", page_size=3)

News is information only. It is stored in the crop context for the assistant, but
it never changes the disease prediction, the confidence or the advice.
"""

import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import streamlit as st

from src.context import get_session_context
from src.news import (
    ALL_TOPICS,
    NO_NEWS,
    UNAVAILABLE,
    NewsUnavailableError,
    get_farmer_news,
    headlines_for_context,
    mentions_crop,
    prioritize,
)

CACHE_SECONDS = 15 * 60
ALL = "All"
TOPIC_STYLE = {      # topic -> (icon, badge colour)
    "Agriculture": (":material/agriculture:", "green"),
    "Government & Schemes": (":material/account_balance:", "blue"),
    "Market & MSP": (":material/currency_rupee:", "orange"),
    "Crop & Disease": (":material/coronavirus:", "red"),
    "Weather & Agriculture": (":material/partly_cloudy_day:", "violet"),
    "Farming Technology": (":material/precision_manufacturing:", "gray"),
}
IST = timezone(timedelta(hours=5, minutes=30))


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Loading the latest farmer news...")
def fetch_news():
    """Cached for a few minutes so reruns do not call the feeds again.

    Raises NewsUnavailableError when no feed can be read; exceptions are not cached,
    so the next run tries again.
    """
    return get_farmer_news()


def load_news():
    """The current NewsReport (or None if unavailable), also saved in the crop context."""
    context = get_session_context()
    try:
        report = fetch_news()
    except NewsUnavailableError:
        report = None
    except Exception:  # noqa: BLE001 - news must never break the rest of the app
        report = None
    if report is None:
        context.update_from_news(None)
        return None
    crop = context.crop
    # A varied selection (crop, every topic, newest) so the assistant can answer topic
    # questions with real headlines.
    context.update_from_news(
        replace(report, items=headlines_for_context(report.items, crop)), crop)
    return report


def md_escape(text):
    """Escape characters that Streamlit markdown would otherwise interpret."""
    return re.sub(r"([\\`*_\[\]<>#$~|])", r"\\\1", text)


def _when(published, now=None):
    if published is None:
        return None
    now = now or datetime.now(timezone.utc)
    local = published.astimezone(IST)
    age = now - published
    if age < timedelta(hours=1):
        ago = f"{max(1, int(age.total_seconds() // 60))} min ago"
    elif age < timedelta(days=1):
        ago = f"{int(age.total_seconds() // 3600)} h ago"
    else:
        ago = f"{age.days} day{'s' if age.days != 1 else ''} ago"
    return f"{local:%d %b %Y, %H:%M} IST ({ago})"


def show_item(item, crop=None, key=None):
    icon, color = TOPIC_STYLE.get(item.category, TOPIC_STYLE["Agriculture"])
    with st.container(border=True):
        with st.container(horizontal=True):
            st.badge(item.category, icon=icon, color=color)
            if crop and mentions_crop(item, crop):
                st.badge(f"Mentions {crop}", icon=":material/eco:", color="green")
        st.markdown(f"**{md_escape(item.title)}**")
        meta = [f"Source: {item.source}"]
        when = _when(item.published)
        if when:
            meta.append(f"Published: {when}")
        st.caption(" · ".join(meta))
        if item.summary:
            st.markdown(md_escape(item.summary if len(item.summary) <= 260
                            else item.summary[:260].rsplit(" ", 1)[0] + "..."))
        st.link_button("Read full article", item.url, icon=":material/open_in_new:", key=key)


def show_news(prefix, page_size=10):
    """Refresh button, category and search filters, and news cards."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption("Latest agriculture updates for farmers, from Indian news publishers.")
        if st.button("Refresh news", icon=":material/refresh:", key=f"{prefix}_refresh"):
            fetch_news.clear()
            st.session_state.pop(f"{prefix}_shown", None)

    report = load_news()
    if report is None:
        st.warning(UNAVAILABLE, icon=":material/cloud_off:")
        return
    if not report.items:
        st.info(NO_NEWS, icon=":material/newspaper:")
        return

    topics = [ALL, *[t for t in ALL_TOPICS if any(i.category == t for i in report.items)]]
    topic = st.pills("Category", topics, default=ALL, required=True, key=f"{prefix}_topic")
    query = st.text_input("Search news", placeholder="e.g. wheat, urea, monsoon",
                          key=f"{prefix}_query").strip().lower()

    crop = get_session_context().crop
    items = prioritize(report.items, crop)
    if topic and topic != ALL:
        items = [i for i in items if i.category == topic]
    if query:
        items = [i for i in items if query in f"{i.title} {i.summary}".lower()]
    if crop and any(mentions_crop(i, crop) for i in items):
        st.caption(f"News mentioning **{crop}** (from your latest disease detection) is "
                   "shown first.")

    if not items:
        st.info("No news matches this category or search.", icon=":material/search_off:")
    shown = st.session_state.get(f"{prefix}_shown", page_size)
    for n, item in enumerate(items[:shown]):
        show_item(item, crop, key=f"{prefix}_article_{n}")
    if len(items) > shown:
        if st.button(f"Show more ({len(items) - shown} more)", icon=":material/expand_more:",
                     key=f"{prefix}_more"):
            st.session_state[f"{prefix}_shown"] = shown + page_size
            st.rerun()

    fetched = report.fetched_at.astimezone(IST)
    notes = [f"Sources: {', '.join(sorted({i.source for i in report.items}))}",
             f"updated {fetched:%H:%M} IST"]
    st.caption(" · ".join(notes) + ". Topics are assigned automatically from keywords in "
               "each headline, so a few may be imprecise.")
    if report.errors:
        st.caption("Could not reach: " + "; ".join(report.errors.values()))
