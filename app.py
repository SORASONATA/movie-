import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from collections import Counter
from urllib.parse import urlencode

# =============== CONFIG ===============
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "b964b8682fdd7c07019be4031b66f77c")  # แนะนำใช้ st.secrets
POSTER_PLACEHOLDER = "https://via.placeholder.com/260x390?text=No+Image"
PKL_PATH = "movie_data_with_director_filled.pkl"   # ต้องมีคอลัมน์ director แล้ว

st.set_page_config(page_title="Movie Recs", layout="wide")

# =============== GLOBAL STYLE ===============
st.markdown(
    """
    <style>
      /* ลด padding หน้าให้แน่นขึ้นเล็กน้อย */
      .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

      /* หัวข้อหลัก */
      h1 { letter-spacing: .2px; }

      /* การ์ดรูปโปสเตอร์ */
      .movie-card {
        background: #ffffff10;
        border: 1px solid rgba(200,200,200,.25);
        border-radius: 16px;
        padding: 10px 10px 14px 10px;
        transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
      }
      .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
        border-color: rgba(200,200,200,.45);
      }

      /* รูปโปสเตอร์ให้โค้งมน */
      .movie-card img { border-radius: 12px; }

      /* ชื่อเรื่องใต้รูป */
      .movie-title {
        margin-top: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.25rem;
      }

      /* แคปชันผู้กำกับ */
      .movie-meta {
        color: #7f8b99;
        font-size: 0.85rem;
        margin-top: 2px;
      }

      /* ปุ่มลิงก์ (เราจะใช้ลิงก์ครอบรูป) */
      .poster-link {
        text-decoration: none !important;
      }

      /* ชิป/ป้ายคำ */
      .chip {
        display: inline-block;
        padding: 3px 10px;
        font-size: 12px;
        border-radius: 999px;
        border: 1px solid rgba(150,150,150,.35);
        margin: 0 6px 6px 0;
        background: rgba(255,255,255,.04);
        white-space: nowrap;
      }

      /* กล่องรายละเอียด */
      .detail-card {
        border: 1px solid rgba(200,200,200,.25);
        border-radius: 16px;
        padding: 16px;
      }

      /* กล่องแนะนำด้านบน */
      .top-tip {
        border-left: 4px solid #5b9cff;
        background: #5b9cff15;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: .95rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =============== DATA LOAD ===============
@st.cache_data(show_spinner=False)
def load_data(pkl_path: str):
    with open(pkl_path, "rb") as f:
        movies, cosine_sim = pickle.load(f)
    # ป้องกันค่าว่าง
    for c in ["title", "overview", "tags", "director"]:
        if c in movies.columns:
            movies[c] = movies[c].fillna("")
    return movies, cosine_sim

movies, cosine_sim = load_data(PKL_PATH)

# =============== TMDB HELPERS ===============
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        r = requests.get(url, timeout=12)
        data = r.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return POSTER_PLACEHOLDER

def poster_link(markdown_img: str, movie_id: int) -> str:
    """
    ห่อรูปด้วยลิงก์ภายในหน้า (ใช้ query params ?mid=)
    ช่วยให้ "คลิกโปสเตอร์" แล้วเปิดรายละเอียดได้ทันที
    """
    qs = urlencode({"mid": int(movie_id)})
    return f"[{markdown_img}](?{qs})"

def grid_show_movies(df: pd.DataFrame, per_row: int = 5, show_director: bool = True):
    if df.empty:
        st.info("ไม่พบรายการ")
        return
    n = len(df)
    for i in range(0, n, per_row):
        cols = st.columns(per_row)
        for col, j in zip(cols, range(i, min(i + per_row, n))):
            row = df.iloc[j]
            poster_url = fetch_poster(int(row["movie_id"]))
            with col:
                with st.container():
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                    # รูปเป็นลิงก์ → เปิด detail ด้วย ?mid=
                    md_img = f"![{row['title']}]({poster_url})"
                    st.markdown(poster_link(md_img, int(row["movie_id"])), unsafe_allow_html=True)

                    # ชื่อเรื่อง + ผู้กำกับ
                    st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
                    if show_director:
                        director = (row.get("director","") or "").strip() or "-"
                        st.markdown(f'<div class="movie-meta">🎬 {director}</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

# =============== RECOMMENDER ===============
def get_recommendations_by_title(title: str, k: int = 10) -> pd.DataFrame:
    hits = movies.index[movies["title"] == title]
    if len(hits) == 0:
        return pd.DataFrame(columns=movies.columns)
    idx = hits[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top = [i for i, _ in sim_scores[1:k+1]]
    return movies.iloc[top]

def get_similar_by_id(movie_id: int, k: int = 10) -> pd.DataFrame:
    hits = movies.index[movies["movie_id"] == movie_id]
    if len(hits) == 0:
        return pd.DataFrame(columns=movies.columns)
    idx = hits[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top = [i for i, _ in sim_scores[1:k+1]]
    return movies.iloc[top]

# =============== SEARCH OPTIONS ===============
movie_opts = pd.DataFrame({"label": movies["title"], "type": "Movie"})
director_series = movies["director"].replace("", np.nan).dropna()
dir_opts = pd.DataFrame({"label": director_series, "type": "Director"}).drop_duplicates()
options = pd.concat([movie_opts, dir_opts], ignore_index=True).drop_duplicates(subset=["label","type"])

# =============== TOP BAR ===============
st.title("🎬 Movie Recommendation System — Single Search")

# แถบแนะนำสั้น ๆ
st.markdown(
    '<div class="top-tip">Tip: คลิกโปสเตอร์เพื่อเปิดหน้า “ข้อมูลหนัง” ด้านล่าง หรือแชร์ลิงก์ที่มี  ให้เพื่อนเปิดเข้าหน้าเดียวกันได้</div>',
    unsafe_allow_html=True
)

# ช่องค้นหาเดียว
c_search, c_btn = st.columns([4, 1])
with c_search:
    selected_label = st.selectbox("🔍 พิมพ์หรือเลือกชื่อภาพยนตร์ / ผู้กำกับ", options["label"].values, index=0)
if selected_label:
    sel_type = options.loc[options["label"] == selected_label, "type"].values[0]
else:
    sel_type = None

with c_btn:
    do_search = st.button("Search", use_container_width=True)

# =============== RESULTS (LEFT) + INFO (RIGHT) ===============
colA, colB = st.columns([1.2, 0.8])

with colA:
    if do_search:
        # เมื่อค้นหาใหม่ ให้เคลียร์ mid จาก URL และ state
        st.session_state.pop("page_movie_id", None)
        st.query_params.clear()

        if sel_type == "Movie":
            st.subheader(f"🎯 Recommendations for *{selected_label}*")
            recs = get_recommendations_by_title(selected_label, k=10)
            grid_show_movies(recs, per_row=5, show_director=True)

        elif sel_type == "Director":
            st.subheader(f"🎬 Movies directed by *{selected_label}*")
            director_movies = movies[movies["director"] == selected_label].copy()

            # === สรุปแนว/แท็กที่ผู้กำกับทำบ่อย ===
            all_tags = " ".join(director_movies["tags"].astype(str))
            tag_counts = Counter(all_tags.split())
            blacklist = {"the","a","an","and","of","to","in","for","on","by","with","is","at","from"}
            top_tags = [t for t,_ in tag_counts.most_common(30) if t.lower() not in blacklist][:12]

            if top_tags:
                st.caption("แนว/คีย์เวิร์ดที่ทำบ่อย:")
                st.markdown("".join([f'<span class="chip">{t}</span>' for t in top_tags]), unsafe_allow_html=True)

            grid_show_movies(director_movies, per_row=5, show_director=False)

with colB:
    st.info("ลอง: เลือกผู้กำกับที่ชอบ → ดูแนวที่ทำบ่อย → คลิกโปสเตอร์เพื่อดูรายละเอียดและ ‘More like this’", icon="💡")

# =============== MOVIE DETAIL PAGE ===============
# รองรับเปิดจากลิงก์: /?mid=123
qp = st.query_params
if "mid" in qp:
    try:
        st.session_state["page_movie_id"] = int(qp.get("mid"))
    except Exception:
        pass

if "page_movie_id" in st.session_state:
    movie_id = int(st.session_state["page_movie_id"])
    row = movies.loc[movies["movie_id"] == movie_id]
    if not row.empty:
        row = row.iloc[0]
        st.markdown("---")
        st.header(row["title"])

        # การ์ดรายละเอียด
        with st.container():
            st.markdown('<div class="detail-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(fetch_poster(movie_id), width=260)
            with c2:
                st.markdown(f"**Director:** {row.get('director','-') or '-'}")
                st.markdown("**Overview:**")
                st.write(row.get("overview","-") or "-")
                st.markdown("**Tags:**")
                tags_line = (row.get("tags","") or "").strip()
                if tags_line:
                    tags_html = "".join([f'<span class="chip">{t}</span>' for t in tags_line.split()])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.caption("-")
            st.markdown('</div>', unsafe_allow_html=True)

        # More like this
        st.subheader("📌 More like this")
        more_like = get_similar_by_id(movie_id, k=10)
        grid_show_movies(more_like, per_row=5, show_director=True)
    else:
        st.warning("ไม่พบข้อมูลภาพยนตร์ในชุดข้อมูล")
