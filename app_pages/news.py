"""Farmer news page: live agriculture news from Indian publishers' RSS feeds."""

import streamlit as st

from src.ui import show_news

st.subheader("Live Farmer News", icon=":material/newspaper:")
show_news("news_page", page_size=10)
st.caption("Headlines and summaries come from the publishers' own feeds; open the article "
           "for the full story. News is general information and does not change the disease "
           "result or the crop-care advice.")
