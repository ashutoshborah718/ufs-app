import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import re

# ==========================================
# 1. LIVE DATA SCRAPER & FEATURE EXTRACTOR
# ==========================================
class PlayerDataFetcher:
    @staticmethod
    def search_wikipedia(player_name: str) -> dict:
        session = requests.Session()
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{player_name} footballer",
            "format": "json"
        }
        res = session.get(search_url, params=params).json()
        if not res.get("query", {}).get("search"):
            return None
        
        page_title = res["query"]["search"][0]["title"]
        
        parse_params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json"
        }
        page_data = session.get(search_url, params=parse_params).json()
        raw_html = page_data.get("parse", {}).get("text", {}).get("*", "")
        
        return PlayerDataFetcher.parse_wiki_html(raw_html, page_title)

    @staticmethod
    def parse_wiki_html(html: str, player_name: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()

        # 1. Detect Position
        pos = "FW"
        if re.search(r'\b(midfielder|winger)\b', text, re.IGNORECASE):
            pos = "MF"
        if re.search(r'\b(defender|centre-back|full-back)\b', text, re.IGNORECASE):
            pos = "DF"
        if re.search(r'\b(goalkeeper)\b', text, re.IGNORECASE):
            pos = "GK"

        # 2. Extract Career Dates & Longevity
        years = re.findall(r'\b(19\d\d|20[0-2]\d)\b', text)
        years = [int(y) for y in years if 1930 <= int(y) <= 2026]
        start_year = min(years) if years else 2005
        end_year = max(years) if years else 2024
        longevity = max(1, min(24, end_year - start_year))

        # 3. Detect Era & Measurement Uncertainty
        is_modern = start_year >= 2005
        sigma_base = 0.08 if is_modern else (0.15 if start_year >= 1990 else 0.25)

        # 4. Count Key Honors & Trophies via keyword scan
        wc_wins = len(re.findall(r'FIFA World Cup(?:\s*winner|\s*\(\d{4}\)|\s*champion)', text, re.I))
        ucl_wins = len(re.findall(r'UEFA Champions League(?:\s*winner|\s*\(\d{4}\))', text, re.I))
        ballon_dor = len(re.findall(r'Ballon d\'Or(?:\s*winner|\s*\(\d{4}\)|\s*:\s*\d{4})', text, re.I))

        # 5. Extract Goals & Caps
        goals = 0
        appearances = 0
        info_table = soup.find("table", class_="infobox")
        if info_table:
            rows = info_table.find_all("tr")
            for row in rows:
                if "Total" in row.get_text():
                    cols = row.find_all(["td", "th"])
                    nums = [re.sub(r'[^\d]', '', c.get_text()) for c in cols]
                    valid = [int(n) for n in nums if n.isdigit()]
                    if len(valid) >= 2:
                        appearances, goals = valid[-2], valid[-1]

        goals_per_game = goals / max(1, appearances) if appearances > 50 else 0.45

        return {
            "name": player_name,
            "position": pos,
            "start_year": start_year,
            "end_year": end_year,
            "longevity": longevity,
            "sigma": sigma_base,
            "goals": goals,
            "apps": appearances,
            "gpg": goals_per_game,
            "wc": min(3, wc_wins),
            "ucl": min(6, ucl_wins),
            "ballon_dor": min(8, ballon_dor)
        }

    @staticmethod
    def derive_ufs_z_scores(meta: dict) -> np.ndarray:
        gpg = meta.get("gpg", 0.40)
        pos = meta.get("position", "FW")

        if pos == "FW":
            pk_z = min(5.2, max(1.5, (gpg - 0.25) * 5.5 + 2.0))
        elif pos == "MF":
            pk_z = min(4.8, max(1.5, (gpg - 0.12) * 6.0 + 2.5))
        elif pos == "DF":
            pk_z = min(4.6, 2.8 + (meta["ucl"] * 0.25))
        else:
            pk_z = min(4.6, 2.7 + (meta["ucl"] * 0.25))

        cv_z = min(4.8, 1.8 + (meta["longevity"] / 20.0) * 2.8)
        cl_z = min(4.9, 2.5 + (meta["ucl"] * 0.3) + (meta["wc"] * 0.5))
        t_z = min(4.8, 2.0 + (meta["wc"] * 0.6) + (meta["ucl"] * 0.4))
        e_z = (pk_z * 0.6) + (cl_z * 0.4) - 0.2
        d_z = max(1.5, pk_z - 0.3)
        s_z = max(2.0, min(5.0, pk_z * 0.95))
        h_z = min(5.0, 2.0 + (meta["ballon_dor"] * 0.35))

        return np.array([pk_z, cv_z, cl_z, t_z, e_z, d_z, s_z, h_z])


# ==========================================
# 2. UFS 4.0 BAYESIAN MONTE CARLO ENGINE
# ==========================================
class UFS4Engine:
    PILLARS = [
        "Peak (Pk)", "Career (CV)", "Clutch (Cl)", "Team (T)",
        "Elevation (E)", "Dominance (D)", "Skill (S)", "Honours (H)"
    ]

    def __init__(self, dirichlet_conc=50.0):
        self.w0 = np.array([0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05])
        self.alpha = self.w0 * dirichlet_conc
        self.mu_0 = 0.0
        self.tau = 1.0
        self.center = 3.0
        self.scale = 0.5

    def simulate(self, players: list, n_samples=20_000, seed=42):
        np.random.seed(seed)
        n_p = len(players)
        n_k = len(self.PILLARS)

        w_samples = np.random.dirichlet(self.alpha, size=n_samples)
        player_samples = np.zeros((n_p, n_samples, n_k))
        shrunk_means = np.zeros((n_p, n_k))

        for idx, p in enumerate(players):
            prec_obs = 1.0 / (p["sigma"] ** 2)
            prec_prior = 1.0 / (self.tau ** 2)
            post_prec = prec_obs + prec_prior

            z_hat = (p["z_scores"] * prec_obs + self.mu_0 * prec_prior) / post_prec
            post_std = np.sqrt(1.0 / post_prec)
            shrunk_means[idx] = z_hat

            player_samples[idx] = np.random.normal(
                loc=z_hat,
                scale=post_std,
                size=(n_samples, n_k)
            )

        composite_z = np.einsum('psk,sk->ps', player_samples, w_samples)
        composite_ufs = 100.0 / (1.0 + np.exp(-(composite_z - self.center) / self.scale))

        pairwise = np.zeros((n_p, n_p))
        for i in range(n_p):
            for j in range(n_p):
                if i != j:
                    pairwise[i, j] = np.mean(composite_z[i] > composite_z[j])
                else:
                    pairwise[i, j] = 0.50

        return {
            "composite_ufs": composite_ufs,
            "shrunk_means": shrunk_means,
            "pairwise": pairwise
        }


# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Universal Footballer Score (UFS 4.0)", layout="wide")

st.title("⚽ Universal Footballer Score (UFS 4.0)")
st.markdown(
    "*A stochastic Bayesian model that searches player career data live, "
    "evaluates performance across 8 pillars, and computes Monte Carlo win probabilities.*"
)

st.sidebar.header("⚙️ Model Parameters")
mc_iterations = st.sidebar.slider("Monte Carlo Iterations", 5_000, 30_000, 15_000, step=5_000)
dirichlet_alpha = st.sidebar.slider("Dirichlet Weight Certainty (α)", 10.0, 100.0, 50.0, step=10.0)

if "roster" not in st.session_state:
    st.session_state.roster = [
        {"name": "Lionel Messi", "position": "FW", "sigma": 0.08,
         "z_scores": np.array([4.65, 4.55, 4.20, 4.30, 4.40, 4.50, 4.60, 4.40])},
        {"name": "Cristiano Ronaldo", "position": "FW", "sigma": 0.08,
         "z_scores": np.array([4.30, 4.50, 4.40, 4.25, 4.00, 4.10, 3.85, 4.20])},
        {"name": "Pelé", "position": "FW", "sigma": 0.25,
         "z_scores": np.array([4.40, 4.25, 4.30, 4.50, 4.10, 4.35, 4.00, 3.90])},
        {"name": "Diego Maradona", "position": "FW", "sigma": 0.22,
         "z_scores": np.array([4.35, 3.80, 4.45, 4.10, 4.40, 4.20, 4.30, 3.80])}
    ]

st.subheader("🔍 Add Any Player via Live Web Search")
col_input, col_btn = st.columns([3, 1])

with col_input:
    query = st.text_input("Enter player name (e.g., Zinedine Zidane, Paolo Maldini, Erling Haaland):")

with col_btn:
    st.write(" ")
    search_clicked = st.button("Fetch & Calculate", type="primary")

if search_clicked and query.strip():
    with st.spinner(f"Fetching stats for '{query}'..."):
        data = PlayerDataFetcher.search_wikipedia(query)
        if data:
            z_scores = PlayerDataFetcher.derive_ufs_z_scores(data)
            new_player = {
                "name": data["name"],
                "position": data["position"],
                "sigma": data["sigma"],
                "z_scores": z_scores
            }
            st.session_state.roster = [p for p in st.session_state.roster if p["name"] != new_player["name"]]
            st.session_state.roster.append(new_player)
            st.success(f"Added **{data['name']}** ({data['position']}) | Era: {data['start_year']}-{data['end_year']}")
        else:
            st.error("Player not found. Try entering their full name.")

engine = UFS4Engine(dirichlet_conc=dirichlet_alpha)
results = engine.simulate(st.session_state.roster, n_samples=mc_iterations)

summary = []
for idx, p in enumerate(st.session_state.roster):
    ufs_dist = results["composite_ufs"][idx]
    summary.append({
        "Player": p["name"],
        "Position": p["position"],
        "Data Uncertainty (σ)": f"±{p['sigma']}",
        "Mean UFS": round(np.mean(ufs_dist), 2),
        "Median UFS": round(np.median(ufs_dist), 2),
        "95% Credible Interval": f"[{np.percentile(ufs_dist, 2.5):.1f} – {np.percentile(ufs_dist, 97.5):.1f}]"
    })

df_summary = pd.DataFrame(summary).sort_values(by="Mean UFS", ascending=False).reset_index(drop=True)

col_table, col_dist = st.columns([1.1, 0.9])

with col_table:
    st.subheader("📊 UFS 4.0 Leaderboard")
    st.dataframe(df_summary, use_container_width=True)

with col_dist:
    st.subheader("📈 Posterior Density (UFS Distribution)")
    fig_kde = go.Figure()
    for idx, p in enumerate(st.session_state.roster):
        fig_kde.add_trace(go.Box(
            y=results["composite_ufs"][idx],
            name=p["name"],
            boxpoints=False
        ))
    fig_kde.update_layout(yaxis_title="UFS (0-100)", showlegend=False, height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_kde, use_container_width=True)

st.subheader("🎯 Pairwise Dominance Matrix: P(Row > Column)")
names = [p["name"] for p in st.session_state.roster]
df_pairwise = pd.DataFrame(
    np.round(results["pairwise"] * 100, 1),
    index=names,
    columns=names
)

fig_heatmap = px.imshow(
    df_pairwise,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto",
    labels=dict(x="Opponent", y="Player", color="Win Prob %")
)
fig_heatmap.update_layout(height=400)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.subheader("🕸️️ Pillar Profile Breakdown (Z-Scores)")
radar_fig = go.Figure()
for idx, p in enumerate(st.session_state.roster):
    radar_fig.add_trace(go.Scatterpolar(
        r=results["shrunk_means"][idx],
        theta=UFS4Engine.PILLARS,
        fill='toself',
        name=p["name"]
    ))
radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    showlegend=True,
    height=450
)
st.plotly_chart(radar_fig, use_container_width=True)
