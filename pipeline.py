"""Pronos Foot : télécharge les données, entraîne le modèle, écrit le site (docs/index.html).

Lancé chaque jour par GitHub Actions. Clé football-data.org : variable d'environnement FD_KEY.
Mode test hors ligne : PRONOS_OFFLINE=<dossier de fichiers factices>.
"""
import io, os, re, json, math, time, difflib, unicodedata, itertools, warnings, datetime as dt
from collections import defaultdict, deque
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------- réglages
SEUIL = 0.75                 # probabilité minimale d'un pronostic (0.70 à 0.80)
MAX_OFFRE = 15
SEUIL_SUR = 0.95
CUT = pd.Timestamp("2024-07-01")   # test = matchs après cette date
N_FORME = 8
MIN_HIST = 5
ELO_K, ELO_HA = 20.0, 60.0
FD_KEY = os.environ.get("FD_KEY", "")
OFFLINE = os.environ.get("PRONOS_OFFLINE")
OUT = os.environ.get("PRONOS_OUT", ".")
BENIN = ZoneInfo("Africa/Porto-Novo")

UK_SEASONS = ["2223", "2324", "2425", "2526", "2627"]
UK_DIVS = {"E0": "Premier League", "F1": "Ligue 1", "SP1": "La Liga", "I1": "Serie A",
           "D1": "Bundesliga", "N1": "Eredivisie", "B1": "Jupiler League",
           "P1": "Liga Portugal", "T1": "Süper Lig"}
FD_COMPS = {"CL": "Champions League", "BSA": "Brasileirão"}
FD_SEASONS = [2023, 2024, 2025, 2026]
AM = {"ARG": ("Liga Argentina", "arg.1"), "MEX": ("Liga MX", "mex.1")}
AM_FROM = pd.Timestamp("2018-01-01")
# nom, identifiant ESPN, groupe de modèle, jour de la semaine (None = tous les jours), début de l'historique
ESPN_HIST = {"EL": ("Europa League", "uefa.europa", "eu", 3, dt.date(2019, 7, 1)),
             "ECL": ("Conference League", "uefa.europa.conf", "eu", 3, dt.date(2019, 7, 1)),
             "COL": ("Liga Colombia", "col.1", "am", None, dt.date(2023, 7, 1))}
ESPN_MAX_CALLS = 1500
MIN_HIST_EU = 3
ALIASES = {}   # corrections manuelles : "nom ESPN": "nom dans l'historique"
INCONNUS = []

TARGETS = ["H", "A", "X1", "X2", "BT", "O15", "O25"]
EVN = ["Victoire domicile", "Victoire extérieur", "1X (domicile ou nul)", "X2 (nul ou extérieur)",
       "Les deux équipes marquent", "Pas les deux équipes qui marquent",
       "Plus de 1.5 buts", "Plus de 2.5 buts", "Moins de 2.5 buts"]
BASE = ["d", "pe", "pdraw", "h_gf", "h_ga", "h_bt", "h_o25", "a_gf", "a_ga", "a_bt", "a_o25", "xh", "xa"]
ODDS = ["qH", "qD", "qA", "qO"]
CC = ["h_cf", "h_ca", "a_cf", "a_ca", "h_cd", "a_cd", "xc", "xk"]
LINES = {"corners": [8.5, 9.5, 10.5, 11.5], "cartons": [2.5, 3.5, 4.5, 5.5]}
SEUIL_CC = 0.70
MAX_CC = 10


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- téléchargement
def http_get(url, headers=None):
    for essai in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200:
                return r
            log("  HTTP", r.status_code, url)
            if r.status_code == 429:
                time.sleep(30)
            else:
                return None
        except requests.RequestException as e:
            log("  erreur réseau", e)
            time.sleep(5)
    return None


def pick(df, names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def uk_frame(df, div, fin):
    df = df.copy()
    df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()]
    d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    if not fin and "Time" in df.columns:
        def conv(date, t):
            try:
                ts = pd.Timestamp(f"{date:%Y-%m-%d} {t}")
                return ts.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)
            except Exception:
                return pd.NaT
        kick = [conv(a, b) if pd.notna(a) and isinstance(b, str) else pd.NaT for a, b in zip(d, df["Time"])]
        has_t = pd.Series([pd.notna(k) for k in kick], index=df.index)
        d = pd.Series([k if pd.notna(k) else a for k, a in zip(kick, d)], index=df.index)
    else:
        has_t = pd.Series(False, index=df.index)
    out = pd.DataFrame({
        "src": "uk", "div": div, "lg": UK_DIVS[div], "dt": d, "has_t": has_t,
        "home": df["HomeTeam"].astype(str), "away": df["AwayTeam"].astype(str),
        "hg": pick(df, ["FTHG"]) if fin else np.nan, "ag": pick(df, ["FTAG"]) if fin else np.nan,
        "oh": pick(df, ["B365H", "AvgH"]), "od": pick(df, ["B365D", "AvgD"]), "oa": pick(df, ["B365A", "AvgA"]),
        "oo": pick(df, ["B365>2.5", "Avg>2.5"]), "ou": pick(df, ["B365<2.5", "Avg<2.5"]),
        "hc": pick(df, ["HC"]), "ac": pick(df, ["AC"]), "hy": pick(df, ["HY"]), "ay": pick(df, ["AY"]),
        "hr": pick(df, ["HR"]), "ar": pick(df, ["AR"]),
    })
    out = out[out["dt"].notna()]
    if fin:
        out = out[out["hg"].notna() & out["ag"].notna()]
    out["fin"] = fin
    return out


def read_csv_bytes(b):
    df = pd.read_csv(io.BytesIO(b), encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).replace("ï»¿", "").replace("\ufeff", "").strip() for c in df.columns]
    return df


def load_uk():
    frames = []
    for s in UK_SEASONS:
        for div in UK_DIVS:
            if OFFLINE:
                p = f"{OFFLINE}/uk_{s}_{div}.csv"
                if not os.path.exists(p):
                    continue
                raw = open(p, "rb").read()
            else:
                r = http_get(f"https://www.football-data.co.uk/mmz4281/{s}/{div}.csv")
                if r is None:
                    log("  manquant :", div, s)
                    continue
                raw = r.content
            try:
                frames.append(uk_frame(read_csv_bytes(raw), div, True))
            except Exception as e:
                log("  fichier illisible", div, s, e)
    # matchs à venir
    try:
        if OFFLINE:
            raw = open(f"{OFFLINE}/fixtures.csv", "rb").read()
        else:
            r = http_get("https://www.football-data.co.uk/fixtures.csv")
            raw = r.content if r is not None else None
        if raw:
            fx = read_csv_bytes(raw)
            for div, g in fx.groupby("Div"):
                if div in UK_DIVS:
                    frames.append(uk_frame(g, div, False))
    except Exception as e:
        log("  fixtures illisible", e)
    return frames


STOP = {"fc", "cf", "fk", "sc", "cd", "ca", "club", "de", "del", "la", "el", "los", "las", "c", "a"}


def norm_tokens(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().replace(".", "")
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in STOP]
    return sorted(toks)


def match_name(name, known):
    if name in ALIASES:
        return ALIASES[name]
    if name in known:
        return name
    t = norm_tokens(name)
    scored = []
    for k in known:
        kt = norm_tokens(k)
        r = difflib.SequenceMatcher(None, " ".join(t), " ".join(kt)).ratio()
        if t and kt and (set(t) <= set(kt) or set(kt) <= set(t)):
            r = max(r, 0.85)
        scored.append((r, k))
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.75 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08):
        return scored[0][1]
    return None


def load_am():
    frames = []
    for code, (lname, slug) in AM.items():
        if OFFLINE:
            p_ = f"{OFFLINE}/new_{code}.csv"
            raw = open(p_, "rb").read() if os.path.exists(p_) else None
        else:
            r = http_get(f"https://www.football-data.co.uk/new/{code}.csv")
            raw = r.content if r is not None else None
        if not raw:
            log("  manquant :", code)
            continue
        df = read_csv_bytes(raw)
        d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed") + pd.Timedelta(hours=12)
        h = pd.DataFrame({"src": "am", "div": code, "lg": lname, "dt": d, "has_t": False,
                          "home": df["Home"].astype(str), "away": df["Away"].astype(str),
                          "hg": pd.to_numeric(df["HG"], errors="coerce"), "ag": pd.to_numeric(df["AG"], errors="coerce"),
                          "fin": True})
        h = h[h["dt"].notna() & h["hg"].notna() & h["ag"].notna() & (h["dt"] >= AM_FROM)]
        frames.append(h)
        known = set(h.tail(800)["home"]) | set(h.tail(800)["away"])
        # matchs à venir (ESPN, non officiel)
        base = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
        today = dt.date.today()
        urls = [base] + [f"{base}?dates={today + dt.timedelta(days=i):%Y%m%d}" for i in range(3)]
        events = {}
        for u in urls:
            if OFFLINE:
                p_ = f"{OFFLINE}/espn_{slug}.json"
                data = json.load(open(p_)) if os.path.exists(p_) else {}
            else:
                r = http_get(u)
                time.sleep(0.5)
                data = r.json() if r is not None else {}
            for e in data.get("events", []):
                try:
                    if e["status"]["type"]["name"] != "STATUS_SCHEDULED":
                        continue
                    c = e["competitions"][0]["competitors"]
                    home = [x for x in c if x["homeAway"] == "home"][0]["team"]["displayName"]
                    away = [x for x in c if x["homeAway"] == "away"][0]["team"]["displayName"]
                    events[e["id"]] = (pd.to_datetime(e["date"], utc=True).tz_localize(None), home, away)
                except Exception:
                    continue
        rows, ok = [], 0
        for t, home, away in events.values():
            mh, ma = match_name(home, known), match_name(away, known)
            if mh is None or ma is None:
                for n, m in ((home, mh), (away, ma)):
                    if m is None:
                        INCONNUS.append(n)
                        proches = difflib.get_close_matches(n, sorted(known), n=3, cutoff=0.0)
                        log(f"    {code} nom non reconnu : {n!r}  plus proches dans l'historique : {proches}")
                continue
            ok += 1
            rows.append({"src": "am", "div": code, "lg": lname, "dt": t, "has_t": True, "home": mh, "away": ma,
                         "hg": np.nan, "ag": np.nan, "fin": False})
        log(f"  {code} : {len(h)} matchs passés, {len(events)} à venir trouvés, {ok} reconnus")
        if rows:
            frames.append(pd.DataFrame(rows))
    return frames


def espn_get(url):
    r = http_get(url)
    time.sleep(0.4)
    try:
        return r.json() if r is not None else None
    except Exception:
        return None


def espn_events(data, statuses):
    out = []
    for e in (data or {}).get("events", []):
        try:
            if e["status"]["type"]["name"] not in statuses:
                continue
            c = e["competitions"][0]["competitors"]
            h = [x for x in c if x["homeAway"] == "home"][0]
            a = [x for x in c if x["homeAway"] == "away"][0]
            out.append((e["id"], pd.to_datetime(e["date"], utc=True).tz_localize(None),
                        h["team"]["displayName"], a["team"]["displayName"], h.get("score"), a.get("score")))
        except Exception:
            continue
    return out


def load_eu():
    """Europa League et Conference League : historique ESPN gardé en cache dans data/ (mis à jour chaque jour)."""
    ddir = os.path.join(OUT, "data")
    os.makedirs(ddir, exist_ok=True)
    cache, dates_p = os.path.join(ddir, "espn_eu.csv"), os.path.join(ddir, "espn_eu_dates.json")
    hist = pd.read_csv(cache, parse_dates=["dt"]) if os.path.exists(cache) else pd.DataFrame(columns=["dt", "div", "home", "away", "hg", "ag"])
    done = json.load(open(dates_p)) if os.path.exists(dates_p) else {}
    today = dt.date.today()
    frames, rows_new = [], []
    calls = [0]
    for div, (name, slug, src_, wd, start) in ESPN_HIST.items():
        base = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
        checked = set(done.get(div, []))
        if not OFFLINE:
            d = start
            while wd is not None and d.weekday() != wd:
                d += dt.timedelta(days=1)
            step = 7 if wd is not None else 1
            appels = 0
            while d < today and calls[0] < ESPN_MAX_CALLS:
                if d.isoformat() not in checked:
                    data = espn_get(base + f"?dates={d:%Y%m%d}")
                    appels += 1
                    calls[0] += 1
                    if data is not None:
                        for _, t, h, a, sh, sa in espn_events(data, {"STATUS_FULL_TIME"}):
                            try:
                                rows_new.append({"dt": t, "div": div, "home": h, "away": a, "hg": int(sh), "ag": int(sa)})
                            except (TypeError, ValueError):
                                pass
                        checked.add(d.isoformat())
                d += dt.timedelta(days=step)
            log(f"  {name} : {appels} jours interrogés (reste {max(0, (today - d).days)} jours à rattraper)")
            done[div] = sorted(checked)
        # matchs à venir
        events = {}
        for u in [base] + [f"{base}?dates={today + dt.timedelta(days=i):%Y%m%d}" for i in range(3)]:
            if OFFLINE:
                p_ = f"{OFFLINE}/espn_{slug}.json"
                data = json.load(open(p_)) if os.path.exists(p_) else {}
            else:
                data = espn_get(u)
            for i_, t, h, a, _, _ in espn_events(data, {"STATUS_SCHEDULED"}):
                events[i_] = (t, h, a)
        frames.append(pd.DataFrame([{"src": src_, "div": div, "lg": name, "dt": t, "has_t": True, "home": h, "away": a,
                                     "hg": np.nan, "ag": np.nan, "fin": False} for t, h, a in events.values()]))
    if rows_new:
        hist = pd.concat([hist, pd.DataFrame(rows_new)], ignore_index=True)
        hist["dt"] = pd.to_datetime(hist["dt"])
        hist = hist.drop_duplicates(["dt", "div", "home", "away"])
        hist.to_csv(cache, index=False)
    hist["dt"] = pd.to_datetime(hist["dt"])
    if not OFFLINE:
        json.dump(done, open(dates_p, "w"))
    if len(hist):
        h = hist.copy()
        h["src"], h["lg"] = h["div"].map({k: v[2] for k, v in ESPN_HIST.items()}), h["div"].map({k: v[0] for k, v in ESPN_HIST.items()})
        h["has_t"], h["fin"] = True, True
        frames.append(h)
    log("  Coupes ESPN : ", len(hist), "matchs passés en cache")
    return [f for f in frames if len(f)]


def load_fd():
    rows = []
    for code, name in FD_COMPS.items():
        for season in FD_SEASONS:
            if OFFLINE:
                p = f"{OFFLINE}/api_{code}_{season}.json"
                if not os.path.exists(p):
                    continue
                data = json.load(open(p))
            else:
                if not FD_KEY:
                    log("  pas de clé FD_KEY : ", code, "ignoré")
                    break
                r = http_get(f"https://api.football-data.org/v4/competitions/{code}/matches?season={season}",
                             {"X-Auth-Token": FD_KEY})
                time.sleep(7)  # limite gratuite : 10 appels par minute
                if r is None:
                    continue
                data = r.json()
            for m in data.get("matches", []):
                st = m.get("status")
                t = pd.to_datetime(m["utcDate"], utc=True).tz_localize(None)
                if st == "FINISHED":
                    ft = m["score"]["fullTime"]
                    if ft["home"] is None:
                        continue
                    hg, ag, fin = ft["home"], ft["away"], True
                elif st in ("SCHEDULED", "TIMED"):
                    hg, ag, fin = np.nan, np.nan, False
                else:
                    continue
                if not m["homeTeam"].get("name") or not m["awayTeam"].get("name"):
                    continue
                rows.append(dict(src="fd", div=code, lg=name, dt=t, has_t=True,
                                 home=m["homeTeam"]["name"], away=m["awayTeam"]["name"],
                                 hg=hg, ag=ag, oh=np.nan, od=np.nan, oa=np.nan, oo=np.nan, ou=np.nan, fin=fin,
                                 hc=np.nan, ac=np.nan, hy=np.nan, ay=np.nan, hr=np.nan, ar=np.nan))
    return [pd.DataFrame(rows)] if rows else []


# ----------------------------------------------------------------- variables
def build(df):
    df = df.sort_values("dt").reset_index(drop=True)
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=N_FORME))
    feats, feats_cc = [], []
    minh = MIN_HIST_EU if len(df) and df["src"].iloc[0] == "eu" else MIN_HIST
    for r in df.itertuples():
        kh, ka = (r.div, r.home), (r.div, r.away)
        hh, ha = hist[kh], hist[ka]
        f = None
        if len(hh) >= minh and len(ha) >= minh:
            diff = elo[kh] + ELO_HA - elo[ka]
            pe = 1 / (1 + 10 ** (-diff / 400))
            def stats(h):
                a = np.array(h, dtype=float)
                return a[:, 0].mean(), a[:, 1].mean(), ((a[:, 0] > 0) & (a[:, 1] > 0)).mean(), ((a[:, 0] + a[:, 1]) >= 3).mean()
            sh, sa = stats(hh), stats(ha)
            f = [diff / 100, pe, 0.28 - 0.14 * abs(2 * pe - 1), *sh, *sa, sh[0] + sa[1], sa[0] + sh[1]]
        feats.append(f)
        cc = [np.nan] * len(CC)
        if f is not None:
            a, b = np.array(hh, dtype=float), np.array(ha, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = [np.nanmean(a[:, 2]), np.nanmean(a[:, 3]), np.nanmean(b[:, 2]), np.nanmean(b[:, 3]),
                     np.nanmean(a[:, 4]), np.nanmean(b[:, 4])]
            cc = v + [v[0] + v[3] + v[2] + v[1], v[4] + v[5]]
        feats_cc.append(cc)
        if r.fin:
            s = 1.0 if r.hg > r.ag else 0.5 if r.hg == r.ag else 0.0
            pe = 1 / (1 + 10 ** (-(elo[kh] + ELO_HA - elo[ka]) / 400))
            elo[kh] += ELO_K * (s - pe)
            elo[ka] -= ELO_K * (s - pe)
            hh.append((r.hg, r.ag, r.hc, r.ac, r.hy + r.hr, r.ay + r.ar))
            ha.append((r.ag, r.hg, r.ac, r.hc, r.ay + r.ar, r.hy + r.hr))
    ok = np.array([f is not None for f in feats])
    X = pd.DataFrame([f if f is not None else [np.nan] * len(BASE) for f in feats], columns=BASE)
    df = pd.concat([df, X, pd.DataFrame(feats_cc, columns=CC)], axis=1)
    df["ok"] = ok
    df["ok_cc"] = ok & df[CC].notna().all(axis=1)
    # probabilités implicites du bookmaker
    inv = 1 / df[["oh", "od", "oa"]]
    s = inv.sum(axis=1, min_count=3)
    df["qH"], df["qD"], df["qA"] = inv["oh"] / s, inv["od"] / s, inv["oa"] / s
    io_, iu = 1 / df["oo"], 1 / df["ou"]
    df["qO"] = io_ / (io_ + iu)
    df["has_odds"] = df[ODDS].notna().all(axis=1)
    return df


def truth(df):
    hg, ag = df["hg"].values, df["ag"].values
    t = np.column_stack([hg > ag, ag > hg, hg >= ag, ag >= hg, (hg > 0) & (ag > 0), hg + ag >= 2, hg + ag >= 3]).astype(int)
    return t


def events(P):
    return np.column_stack([P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], 1 - P[:, 4], P[:, 5], P[:, 6], 1 - P[:, 6]])


def events_truth(T):
    return np.column_stack([T[:, 0], T[:, 1], T[:, 2], T[:, 3], T[:, 4], 1 - T[:, 4], T[:, 5], T[:, 6], 1 - T[:, 6]])


def min_train(train):
    return 150 if len(train) and str(train["src"].iloc[0]) == "eu" else 300


def fit_predict(train, test, cols):
    if len(train) < min_train(train) or len(test) == 0:
        return None
    T = truth(train)
    P = np.zeros((len(test), len(TARGETS)))
    for j in range(len(TARGETS)):
        if T[:, j].min() == T[:, j].max():
            return None
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        m.fit(train[cols], T[:, j])
        P[:, j] = m.predict_proba(test[cols])[:, 1]
    return P


def fit_goals(train, test, cols):
    """Buts attendus (domicile, extérieur) par régression de Poisson."""
    if len(train) < min_train(train) or len(test) == 0:
        return None
    out = []
    for c in ("hg", "ag"):
        m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
        m.fit(train[cols], train[c].astype(float))
        out.append(m.predict(test[cols]))
    return out[0], out[1]


def score_table(lh, la, nmax=7):
    k = np.arange(nmax)
    f = np.array([math.factorial(int(i)) for i in k], dtype=float)
    ph = np.exp(-lh)[:, None] * lh[:, None] ** k / f
    pa = np.exp(-la)[:, None] * la[:, None] ** k / f
    return ph[:, :, None] * pa[:, None, :]          # (n, buts dom., buts ext.)


def top_scores(lh, la, n=3):
    M = score_table(lh, la)
    res = []
    for mat in M:
        flat = np.argsort(mat.ravel())[::-1][:n]
        res.append([(f"{i // M.shape[2]}-{i % M.shape[2]}", float(mat.ravel()[i])) for i in flat])
    return res


def fit_count(train, test, cols, y):
    if len(train) < 300 or len(test) == 0:
        return None
    m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=500))
    m.fit(train[cols], train[y])
    mu_tr = np.maximum(m.predict(train[cols]), 0.1)
    phi = max(1.0, float(np.mean((train[y].values - mu_tr) ** 2 / mu_tr)))
    return m.predict(test[cols]), phi


def over_prob(mu, phi, line):
    k = int(math.floor(line))
    if phi > 1.05:
        n = mu / (phi - 1)
        return 1 - nbinom.cdf(k, n, n / (n + mu))
    return 1 - poisson.cdf(k, mu)


def choisir_cc(mu, phi, stat, seuil=SEUIL_CC):
    best = None
    for line in LINES[stat]:
        q = float(over_prob(mu, phi, line))
        for side, pr in (("Plus de", q), ("Moins de", 1 - q)):
            if pr >= seuil and (best is None or pr < best[2]):
                best = (side, line, pr)
    return best


def cc_section(df, up):
    out, stat = {"corners": [], "cartons": []}, {}
    base = df[df["ok_cc"] & df["fin"]].copy()
    base["corners"] = base["hc"] + base["ac"]
    base["cartons"] = base["hy"] + base["hr"] + base["ay"] + base["ar"]
    cols = BASE + CC
    ucc = up[up["ok_cc"]]
    for name in out:
        bn = base.dropna(subset=[name])
        tr, te = bn[bn["dt"] < CUT], bn[bn["dt"] >= CUT]
        r = fit_count(tr, te, cols, name)
        if r:
            mu, phi = r
            rows = []
            for m_, d_, y_ in zip(mu, te["dt"].dt.normalize(), te[name]):
                c = choisir_cc(m_, phi, name)
                if c:
                    rows.append((d_, c[2], bool(y_ > c[1]) if c[0] == "Plus de" else bool(y_ < c[1])))
            if rows:
                t = pd.DataFrame(rows, columns=["d", "p", "w"])
                top = t.sort_values("p", ascending=False).groupby("d").head(MAX_CC)
                stat[name] = {"paris": int(len(top)), "taux": round(100 * float(top["w"].mean()), 1)}
                log(f"  test {name} : {len(top)} paris, réussite {stat[name]['taux']} %")
        r = fit_count(bn, ucc, cols, name)
        if r:
            mu, phi = r
            for row, m_ in zip(ucc.itertuples(), mu):
                c = choisir_cc(m_, phi, name)
                if c:
                    kb = row.dt + pd.Timedelta(hours=1)
                    out[name].append({"jour": kb.strftime("%Y-%m-%d"), "heure": kb.strftime("%H:%M") if row.has_t else "",
                                      "ligue": row.lg, "match": f"{row.home} - {row.away}", "home": row.home, "away": row.away,
                                      "pari": f"{c[0]} {c[1]} {name}", "p": round(c[2], 3), "mu": round(float(m_), 1)})
        out[name].sort(key=lambda x: -x["p"])
        out[name] = out[name][:MAX_CC]
    return out, stat


def choisir(E, seuil=SEUIL):
    mask = E >= seuil
    idx = np.where(mask, E, 9).argmin(axis=1)
    return idx, mask.any(axis=1)


FAMILIES = {
    "uk_odds": ("Championnats avec cotes", ["uk"], BASE + ODDS, True),
    "uk": ("Championnats sans cotes", ["uk"], BASE, False),
    "fd": ("Coupes d'Europe et Brésil", ["fd"], BASE, False),
    "eu": ("Europa League et Conference League", ["eu"], BASE, False),
    "am": ("Argentine, Mexique, Colombie", ["am"], BASE, False),
}


def family_mask(df, fam):
    _, srcs, _, need_odds = FAMILIES[fam]
    m = df["src"].isin(srcs) & df["ok"]
    if need_odds:
        m &= df["has_odds"]
    return m


# ----------------------------------------------------------------- principal
def main():
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    now_b = now_utc + pd.Timedelta(hours=1)
    today_b, tomorrow_b = now_b.normalize(), now_b.normalize() + pd.Timedelta(days=1)

    frames = load_uk() + load_fd() + load_am() + load_eu()
    if not frames:
        raise SystemExit("Aucune donnée téléchargée.")
    raw = pd.concat(frames, ignore_index=True)
    raw["dt"] = pd.to_datetime(raw["dt"])
    raw["day"] = raw["dt"].dt.strftime("%Y-%m-%d")
    raw = raw.sort_values("fin", ascending=False).drop_duplicates(["div", "home", "away", "day"], keep="first")
    log("matchs terminés :", int(raw["fin"].sum()), "| à venir :", int((~raw["fin"]).sum()))
    df = pd.concat([build(raw[raw["div"] == d]) for d in raw["div"].unique()], ignore_index=True)

    # --- test sur le passé (modèle entraîné avant CUT, testé après)
    stats = {}
    sc_hit, sc_n, sc_all_hit, sc_all_n = 0, 0, 0, 0
    bt = []
    for fam, (label, _, cols, _) in FAMILIES.items():
        m = family_mask(df, fam) & df["fin"]
        tr, te = df[m & (df["dt"] < CUT)], df[m & (df["dt"] >= CUT)]
        G = fit_goals(tr, te, cols)
        if G is not None:
            ts = top_scores(G[0], G[1], 1)
            ok_ = np.array([f"{int(a)}-{int(b)}" == s[0][0] for a, b, s in zip(te["hg"], te["ag"], ts)])
            pp = np.array([s[0][1] for s in ts])
            sc_all_hit += int(ok_.sum()); sc_all_n += len(ok_)
            tmp = pd.DataFrame({"d": te["dt"].dt.normalize().values, "p": pp, "ok": ok_})
            top = tmp.sort_values("p", ascending=False).groupby("d").head(5)
            sc_hit += int(top["ok"].sum()); sc_n += len(top)
        P = fit_predict(tr, te, cols)
        if P is None:
            continue
        E, Et = events(P), events_truth(truth(te))
        idx, has = choisir(E)
        if has.sum() == 0:
            continue
        win = Et[np.arange(len(te)), idx][has]
        dts, hs, as_ = te["dt"].dt.normalize().values, te["home"].values, te["away"].values
        for i_ in np.where(has)[0]:
            bt.append((dts[i_], hs[i_], as_[i_], float(E[i_, idx[i_]]), bool(Et[i_, idx[i_]])))
        stats[fam] = {"label": label, "matchs": int(len(te)), "paris": int(has.sum()), "taux": round(float(win.mean()) * 100, 1)}
        log(f"  test {label} : {has.sum()} paris sur {len(te)} matchs, réussite {win.mean()*100:.1f} %")

    bt = pd.DataFrame(bt, columns=["d", "home", "away", "p", "w"]).drop_duplicates(["d", "home", "away"])
    stats_all = {"paris": int(len(bt)), "taux": round(100 * float(bt["w"].mean()), 1) if len(bt) else None}
    tk_res = []
    for _, g in bt.groupby("d"):
        cand_ = g.sort_values("p", ascending=False).head(12).to_dict("records")
        best_, bp_ = None, 0.0
        for n_ in (2, 3):
            for combo in itertools.combinations(cand_, n_):
                pp_ = float(np.prod([c["p"] for c in combo]))
                if 1.9 <= 1 / pp_ <= 2.4 and pp_ > bp_:
                    best_, bp_ = combo, pp_
        if best_:
            tk_res.append((bp_, all(c["w"] for c in best_)))
    ticket_stat = {"paris": len(tk_res), "taux": round(100 * float(np.mean([w for _, w in tk_res])), 1) if tk_res else None,
                   "p_moyen": round(100 * float(np.mean([q for q, _ in tk_res])), 1) if tk_res else None}
    log("  test tickets cote 2 :", ticket_stat)

    # --- prédictions des matchs à venir
    up = df[~df["fin"]].copy()
    local = up["dt"] + pd.Timedelta(hours=1)
    keep = np.where(up["has_t"], up["dt"] >= now_utc, local.dt.normalize() >= today_b) & (local.dt.normalize() <= tomorrow_b)
    up = up[keep]
    picks = []
    scores = []
    ignores = 0
    for fam, (label, _, cols, _) in FAMILIES.items():
        trm = family_mask(df, fam) & df["fin"]
        um = up[family_mask(up, fam)]
        if fam == "uk":
            um = up[(up["src"] == "uk") & up["ok"] & ~up["has_odds"]]
        G = fit_goals(df[trm], um, cols)
        if G is not None:
            for row, ts, a, b in zip(um.itertuples(), top_scores(G[0], G[1], 3), G[0], G[1]):
                kb = row.dt + pd.Timedelta(hours=1)
                scores.append({"jour": kb.strftime("%Y-%m-%d"), "heure": kb.strftime("%H:%M") if row.has_t else "",
                               "ligue": row.lg, "match": f"{row.home} - {row.away}", "home": row.home, "away": row.away, "score": ts[0][0],
                               "p": round(ts[0][1], 3), "alt": [[s, round(q, 3)] for s, q in ts[1:]],
                               "buts": round(float(a + b), 1)})
        P = fit_predict(df[trm], um, cols)
        if P is None:
            continue
        E = events(P)
        idx, has = choisir(E)
        for i, (row, e) in enumerate(zip(um.itertuples(), E)):
            if not has[i]:
                continue
            k = idx[i]
            kb = row.dt + pd.Timedelta(hours=1)
            picks.append({"jour": kb.strftime("%Y-%m-%d"), "heure": kb.strftime("%H:%M") if row.has_t else "",
                          "ligue": row.lg, "match": f"{row.home} - {row.away}", "home": row.home, "away": row.away,
                          "pari": EVN[k], "k": int(k), "p": round(float(e[k]), 3), "fam": fam})
    ignores = int((~up["ok"]).sum())
    if INCONNUS:
        log("  noms d'équipes non reconnus (à ajouter dans ALIASES) :", sorted(set(INCONNUS)))
    scores.sort(key=lambda x: -x["p"])
    scores = scores[:5]
    scores_stat = {"paris": sc_n, "taux": round(100 * sc_hit / sc_n, 1) if sc_n else None,
                   "base": round(100 * sc_all_hit / sc_all_n, 1) if sc_all_n else None}
    picks.sort(key=lambda x: -x["p"])
    offre = picks[:MAX_OFFRE]

    # --- ticket cote ~2 (probabilités multipliées, hypothèse d'indépendance)
    ticket = None
    cand = offre[:12]
    best = 0.0
    for n in (2, 3):
        for combo in itertools.combinations(cand, n):
            p = float(np.prod([c["p"] for c in combo]))
            if 1.9 <= 1 / p <= 2.4 and p > best:
                best, ticket = p, {"paris": list(combo), "p": round(p, 3), "cote": round(1 / p, 2)}
    sur = [x for x in picks if x["p"] >= SEUIL_SUR][:2]
    sur_info = None
    if sur:
        p = float(np.prod([x["p"] for x in sur]))
        sur_info = {"paris": sur, "p": round(p, 3), "cote": round(1 / p, 2)}

    try:
        cc, cc_stat = cc_section(df, up)
    except Exception as e:
        log("  cartons/corners ignorés :", repr(e))
        cc, cc_stat = {"corners": [], "cartons": []}, {}

    # --- historique des pronostics (enregistrés avant les matchs) et bilans par onglet
    hist_path = os.path.join(OUT, "data", "historique.csv")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    cols_h = ["cle", "genere", "jour", "ligue", "match", "home", "away", "pari", "k", "p", "resultat", "type", "extra", "reel"]
    H = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame(columns=cols_h)
    H = H.reindex(columns=cols_h)
    H["type"] = H["type"].fillna("jour")
    H["extra"] = H["extra"].fillna("")
    H["reel"] = H["reel"].fillna("")
    gen = now_b.strftime("%Y-%m-%d %H:%M")
    have = set(H["cle"])

    def add(typ, cle, jour, ligue, match, home, away, pari, k, pr, extra=""):
        if cle not in have:
            H.loc[len(H)] = [cle, gen, jour, ligue, match, home, away, pari, k, pr, np.nan, typ, extra, ""]
            have.add(cle)

    for o in offre:
        add("jour", f"{o['jour']}|{o['home']}|{o['away']}|{o['pari']}", o["jour"], o["ligue"], o["match"], o["home"], o["away"],
            o["pari"], o["k"], o["p"])
    for o in scores:
        add("score", f"{o['jour']}|{o['home']}|{o['away']}|score", o["jour"], o["ligue"], o["match"], o["home"], o["away"],
            o["score"], -1, o["p"])
    for typ in ("cartons", "corners"):
        for o in cc[typ]:
            add(typ, f"{o['jour']}|{o['home']}|{o['away']}|{o['pari']}", o["jour"], o["ligue"], o["match"], o["home"], o["away"],
                o["pari"], -1, o["p"])
    if ticket:
        tj = today_b.strftime("%Y-%m-%d")
        if not ((H["type"] == "cote2") & (H["jour"] == tj)).any():
            legs = [f"{o['jour']}|{o['home']}|{o['away']}|{o['pari']}" for o in ticket["paris"]]
            add("cote2", f"{tj}|cote2|" + "+".join(legs), tj, "Ticket", f"Ticket du {today_b:%d/%m}", "", "",
                f"{len(legs)} paris, cote ~{ticket['cote']}", -1, ticket["p"], ";".join(legs))

    res = {}
    for r in df[df["fin"]].itertuples():
        corn = r.hc + r.ac if pd.notna(r.hc) and pd.notna(r.ac) else np.nan
        card = r.hy + r.hr + r.ay + r.ar if pd.notna([r.hy, r.hr, r.ay, r.ar]).all() else np.nan
        for dd in (r.dt, r.dt + pd.Timedelta(hours=1)):
            res[(r.home, r.away, dd.strftime("%Y-%m-%d"))] = (r.hg, r.ag, corn, card)
    for i, r in H.iterrows():
        if pd.notna(r["resultat"]) or r["type"] == "cote2":
            continue
        got = res.get((r["home"], r["away"], r["jour"]))
        if got is None:
            continue
        hg, ag, corn, card = got
        if r["type"] == "jour":
            t = events_truth(truth(pd.DataFrame({"hg": [hg], "ag": [ag]})))[0]
            H.at[i, "resultat"] = int(t[int(r["k"])])
            H.at[i, "reel"] = f"{int(hg)}-{int(ag)}"
        elif r["type"] == "score":
            H.at[i, "resultat"] = int(f"{int(hg)}-{int(ag)}" == r["pari"])
            H.at[i, "reel"] = f"{int(hg)}-{int(ag)}"
        elif r["type"] in ("cartons", "corners"):
            val = card if r["type"] == "cartons" else corn
            if pd.notna(val):
                side, line = r["pari"].split(" ")[0], float(r["pari"].split(" ")[2])
                H.at[i, "resultat"] = int(val > line) if side == "Plus" else int(val < line)
                H.at[i, "reel"] = f"{int(val)} {r['type']}"
    byc = H.set_index("cle")["resultat"]
    for i, r in H[H["type"] == "cote2"].iterrows():
        if pd.notna(r["resultat"]):
            continue
        vals = [byc.get(c, np.nan) for c in r["extra"].split(";")]
        if any(v == 0 for v in vals):
            H.at[i, "resultat"] = 0
        elif all(pd.notna(v) for v in vals):
            H.at[i, "resultat"] = 1
        if any(v == 0 for v in vals) or all(pd.notna(v) for v in vals):
            H.at[i, "reel"] = f"{int(sum(1 for v in vals if v == 1))}/{len(vals)} paris gagnés"
    H.to_csv(hist_path, index=False)

    tj_ = today_b.strftime("%Y-%m-%d")

    def bilan_de(typ):
        sub = H[H["type"] == typ]
        done = sub[sub["resultat"].notna()]
        liste = sub[sub["resultat"].notna() | (sub["jour"] < tj_)].sort_values(["jour", "p"], ascending=[False, False]).head(40)
        return {"n": int(len(done)), "gagnes": int(done["resultat"].astype(float).sum()),
                "recent": [{"jour": r.jour, "match": r.match, "pari": r.pari, "p": float(r.p), "reel": r.reel,
                            "ok": None if pd.isna(r.resultat) else bool(float(r.resultat))}
                           for r in liste.itertuples()],
                "en_attente": int(len(sub) - len(done))}
    bilans = {t: bilan_de(t) for t in ("jour", "cote2", "score", "cartons", "corners")}
    bilan = bilans["jour"]

    data = {"genere": now_b.strftime("%d/%m/%Y à %H:%M"), "seuil": int(SEUIL * 100), "offre": offre,
            "ticket": ticket, "sur": sur_info, "stats": list(stats.values()), "scores": scores, "scores_stat": scores_stat,
            "corners": cc["corners"], "cartons": cc["cartons"], "cc_stat": cc_stat, "bilan": bilan, "bilans": bilans,
            "stats_all": stats_all, "ticket_stat": ticket_stat, "ignores": ignores}
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"), encoding="utf-8").read()
    os.makedirs(os.path.join(OUT, "docs"), exist_ok=True)
    open(os.path.join(OUT, "docs", "index.html"), "w", encoding="utf-8").write(
        tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    json.dump(data, open(os.path.join(OUT, "docs", "offre.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"OK : {len(offre)} pronostics, ticket {'oui' if ticket else 'non'}, sûr {'oui' if sur_info else 'non'}")


if __name__ == "__main__":
    main()
