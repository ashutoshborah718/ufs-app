import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import re

# ==========================================
# 0. PRE-INDEXED POPULAR PLAYERS DATABASE
# ==========================================
TOP_PLAYERS_INDEX = [
    # Modern & Active
    "Erling Haaland", "Kylian Mbappé", "Kevin De Bruyne", "Harry Kane", 
    "Karim Benzema", "Vinícius Júnior", "Luka Modrić", "Toni Kroos", 
    "Robert Lewandowski", "Mohamed Salah", "Neymar Jr", "Virgil van Dijk",
    "Rodri", "Jude Bellingham", "Antoine Griezmann", "Alisson Becker",
    # All-Time Legends
    "Lionel Messi", "Cristiano Ronaldo", "Pelé", "Diego Maradona",
    "Zinedine Zidane", "Paolo Maldini", "Ronaldo Nazário", "Ronaldinho",
    "Thierry Henry", "Johan Cruyff", "Franz Beckenbauer", "Manuel Neuer",
    "Andrés Iniesta", "Xavi Hernández", "Sergio Ramos", "Gianluigi Buffon",
    "Iker Casillas", "Lev Yashin", "Roberto Carlos", "Kaká", "Custom (Type Name)"
]

# ==========================================
# 1. LIVE DATA SCRAPER & FEATURE EXTRACTOR
# ==========================================
class PlayerDataFetcher:
    HEADERS = {
        "User-Agent": "UniversalFootballerScoreApp/4.0 (contact: admin@footballanalytics.org) python-requests"
    }

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
        
        try:
            res = session.get(search_url, params=params, headers=PlayerDataFetcher.HEADERS, timeout=10)
            if res.status_code != 200:
                return None
            search_json = res.json()
            search_results = search_json.get("query", {}).get("search", [])
            if not search_results:
                return None
            
            page_title = search_results[0]["title"]
            
            parse_params = {
                "action": "parse",
                "page": page_title,
                "prop": "text",
                "format": "json"
            }
            page_res = session.get(search_url, params=parse_params, headers=PlayerDataFetcher.HEADERS, timeout=10)
            if page_res.status_code != 200:
                return None
            page_data = page_res.json()
            raw_html = page_data.get("parse", {}).get("text", {}).get("*", "")
            
            return PlayerDataFetcher.parse_wiki_html(raw_html, page_title)
        except Exception:
            return None

    @staticmethod
    def parse_wiki_html(html: str, player_name: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()

        # 1. Robust Position Extraction from Infobox
        pos = "FW"
        infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
        
        pos_found = False
        if infobox:
            for tr in infobox.find_all("tr"):
                header = tr.find(["th", "td"])
                if header and "position" in header.get_text().lower():
                    td = tr.find_all(["td", "th"])[-1]
                    pos_text = td.get_text().lower()
                    if any(k in pos_text for k in ["striker", "forward", "winger", "centre-forward"]):
                        pos = "FW"
                        pos_found = True
                    elif any(k in pos_text for k in ["midfielder", "attacking mid", "defensive mid"]):
                        pos = "MF"
                        pos_found = True
                    elif any(k in pos_text for k in ["defender", "centre-back", "full-back"]):
                        pos = "DF"
                        pos_found = True
                    elif "goalkeeper" in pos_text:
                        pos = "GK"
                        pos_found = True
                    break
        
        # Fallback if position row not detected in infobox
        if not pos_found:
            lead_para = ""
            for p in soup.find_all("p"):
                if len(p.get_text()) > 60:
                    lead_para = p.get_text().lower()
                    break
            if any(k in lead_para for k in ["striker", "forward", "winger"]):
                pos = "FW"
            elif any(k in lead_para for k in ["midfielder", "playmaker"]):
                pos = "MF"
            elif any(k in lead_para for k in ["defender", "centre-back"]):
                pos = "DF"
            elif "goalkeeper" in lead_para:
                pos = "GK"

        # 2. Career Longevity & Era
        years = re.findall(r'\b(19\d\d|20[0-2]\d)\b', text)
        years = [int(y) for y in years if 1935 <= int(y) <= 2026]
        start_year = min(years) if years else 2015
        end_year = max(years) if years else 2025
        longevity = max(2, min(24, end_year - start_year))

        # Modern vs Classic era uncertainty (tight for tracking era)
        is_modern = start_year >= 2008
        sigma_base = 0.08 if is_modern else (0.15 if start_year >= 1990 else 0.25)

        # 3. Trophies & Awards
        wc_wins = len(re.findall(r'FIFA World Cup(?:\s*winner|\s*\(\d{4}\)|\s*champion)', text, re.I))
        ucl_wins = len(re.findall(r'UEFA Champions League(?:\s*winner|\s*\(\d{4}\))', text, re.I))
        ballon_dor = len(re.findall(r'Ballon d\'Or(?:\s*winner|\s*\(\d{4}\)|\s*:\s*\d{4})', text, re.I))
        league_titles = len(re.findall(r'(?:Premier League|La Liga|Serie A|Bundesliga|Ligue 1)(?:\s*winner|\s*\(\d{4}(?:–\d{2,4})?\))', text, re.I))

        # 4. Senior Appearances & Goals
        goals = 0
        appearances = 0
        if infobox:
            for row in infobox.find_all("tr"):
                if "total" in row.get_text().lower():
                    cols = row.find_all(["td", "th"])
                    nums = [re.sub(r'[^\d]', '', c.get_text()) for c in cols]
                    valid = [int(n) for n in nums if n.isdigit()]
                    if len(valid) >= 2:
                        appearances, goals = valid[-2], valid[-1]

        goals_per_game = goals / max(1, appearances) if appearances > 30 else 0.50

        # Special overrides for active generational outliers
        if "haaland" in player_name.lower():
            pos = "FW"
            goals_per_game = max(goals_per_game, 0.88)
        elif "mbapp" in player_name.lower():
            pos = "FW"
            goals_per_game = max(goals_per_game, 0.72)
        elif "de bruyne" in player_name.lower():
            pos = "MF"

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
            "leagues": min(12, league_titles),
            "ballon_dor": min(8, ballon_dor)
        }

    @staticmethod
    def derive_ufs_z_scores(meta: dict) -> np.ndarray:
        gpg = meta.get("gpg", 0.45)
        pos = meta.get("position", "FW")
        longevity = meta.get("longevity", 8)
        ucl = meta.get("ucl", 0)
        wc = meta.get("wc", 0)
        leagues = meta.get("leagues", 0)
        b_dor = meta.get("ballon_dor", 0)

        # 1. Peak Z-Score (Pk)
        if pos == "FW":
            # GPG > 0.85 (Haaland, Messi) maps directly into 4.5+ sigma peak
            pk_z = min(5.3, max(2.0, 2.2 + (gpg * 3.4)))
        elif pos == "MF":
            pk_z = min(4.9, max(2.0, 3.2 + (ucl * 0.25) + (gpg * 2.0)))
        elif pos == "DF":
            pk_z = min(4.8, 3.2 + (ucl * 0.25) + (leagues * 0.08))
        else: # GK
            pk_z = min(4.8, 3.1 + (ucl * 0.25) + (wc * 0.4))

        # 2. Career Value (CV) - rewards longevity, with grace period for active peaks
        cv_z = min(4.9, 2.5 + (longevity / 18.0) * 2.2)

        # 3. Clutch (Cl)
        cl_z = min(5.0, 2.8 + (ucl * 0.35) + (wc * 0.55))

        # 4. Team Success (T)
        t_z = min(5.0, 2.3 + (wc * 0.7) + (ucl * 0.4) + (leagues * 0.12))

        # 5. Elevation (E)
        e_z = (pk_z * 0.55) + (cl_z * 0.45) - 0.1

        # 6. Era Dominance (D)
        d_z = max(2.0, pk_z - 0.25)

        # 7. Skill/Technique (S)
        s_z = max(2.5, min(5.0, pk_z * 0.94))

        # 8. Honours (H)
        h_z = min(5.0, 2.2 + (b_dor * 0.4) + (ucl * 0.15))

        return np.array([pk_z, cv_z, cl_z, t_z, e_z, d_z, s_z, h_z])


# ==========================================
# 2. UFS 4.0 BAYESIAN ENGINE
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

    def simulate(self, players: list, n_samples=15_000, seed=42):
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
    "*A stochastic Bayesian model with autocomplete search that evaluates player performance across 8 pillars, "
    "adjusts for data uncertainty, and simulates Monte Carlo win probabilities.*"
)

st.sidebar.header("⚙️ Model Parameters")
mc_iterations = st.sidebar.slider("Monte Carlo Iterations", 5_000, 30_000, 15_000, step=5_000)
dirichlet_alpha = st.sidebar.slider("Dirichlet Weight Certainty (α)", 10.0, 100.0, 50.0, step=10.0)

# Default roster
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

# Autocomplete Player Selector
st.subheader("🔍 Add Any Player (Type to Autocomplete)")
col_select, col_custom, col_btn = st.columns([2.5, 2, 1])

with col_select:
    selected_option = st.selectbox(
        "Search famous players (type letter to filter):",
        options=TOP_PLAYERS_INDEX,
        index=0
    )

with col_custom:
    if selected_option == "Custom (Type Name)":
        custom_name = st.text_input("Type any player name:")
        target_name = custom_name
    else:
        target_name = selected_option
        st.info(f"Selected: **{target_name}**")

with col_btn:
    st.write(" ")
    search_clicked = st.button("Fetch & Calculate", type="primary")

if search_clicked and target_name.strip():
    with st.spinner(f"Parsing stats & honours for '{target_name}'..."):
        data = PlayerDataFetcher.search_wikipedia(target_name)
        if data:
            z_scores = PlayerDataFetcher.derive_ufs_z_scores(data)
            new_player = {
                "name": data["name"],
                "position": data["position"],
                "sigma": data["sigma"],
                "z_scores": z_scores
            }
            # Remove duplicate if exists, append updated
            st.session_state.roster = [p for p in st.session_state.roster if p["name"] != new_player["name"]]
            st.session_state.roster.append(new_player)
            st.success(f"Added **{data['name']}** ({data['position']}) | Era: {data['start_year']}-{data['end_year']} | Peak: {data['gpg']:.2f} g/g")
        else:
            st.error(f"Could not parse player '{target_name}'. Try typing their official Wikipedia name.")

# Run Simulation
engine = UFS4Engine(dirichlet_conc=dirichlet_alpha)
results = engine.simulate(st.session_state.roster, n_samples=mc_iterations)

# Leaderboard Output
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

st.subheader("🕸 Pillar Profile Breakdown (Z-Scores)")
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
