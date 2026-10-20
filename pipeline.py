"""Pronos Foot : télécharge les données, entraîne le modèle, écrit le site (docs/index.html).

Lancé chaque jour par GitHub Actions. Clé football-data.org : variable d'environnement FD_KEY.
Mode test hors ligne : PRONOS_OFFLINE=<dossier de fichiers factices>.
"""
import io, os, json, time, itertools, datetime as dt
from collections import defaultdict, deque
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
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

TARGETS = ["H", "A", "X1", "X2", "BT", "O15", "O25"]
EVN = ["Victoire domicile", "Victoire extérieur", "1X (domicile ou nul)", "X2 (nul ou extérieur)",
       "Les deux équipes marquent", "Pas les deux équipes qui marquent",
       "Plus de 1.5 buts", "Plus de 2.5 buts", "Moins de 2.5 buts"]
BASE = ["d", "pe", "pdraw", "h_gf", "h_ga", "h_bt", "h_o25", "a_gf", "a_ga", "a_bt", "a_o25", "xh", "xa"]
ODDS = ["qH", "qD", "qA", "qO"]


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
    })
    out = out[out["dt"].notna()]
    if fin:
        out = out[out["hg"].notna() & out["ag"].notna()]
    out["fin"] = fin
    return out


def read_csv_bytes(b):
    return pd.read_csv(io.BytesIO(b), encoding="latin-1", on_bad_lines="skip")


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
                                 hg=hg, ag=ag, oh=np.nan, od=np.nan, oa=np.nan, oo=np.nan, ou=np.nan, fin=fin))
    return [pd.DataFrame(rows)] if rows else []


# ----------------------------------------------------------------- variables
def build(df):
    df = df.sort_values("dt").reset_index(drop=True)
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=N_FORME))
    feats = []
    for r in df.itertuples():
        kh, ka = (r.div, r.home), (r.div, r.away)
        hh, ha = hist[kh], hist[ka]
        f = None
        if len(hh) >= MIN_HIST and len(ha) >= MIN_HIST:
            diff = elo[kh] + ELO_HA - elo[ka]
            pe = 1 / (1 + 10 ** (-diff / 400))
            def stats(h):
                a = np.array(h, dtype=float)
                return a[:, 0].mean(), a[:, 1].mean(), ((a[:, 0] > 0) & (a[:, 1] > 0)).mean(), ((a[:, 0] + a[:, 1]) >= 3).mean()
            sh, sa = stats(hh), stats(ha)
            f = [diff / 100, pe, 0.28 - 0.14 * abs(2 * pe - 1), *sh, *sa, sh[0] + sa[1], sa[0] + sh[1]]
        feats.append(f)
        if r.fin:
            s = 1.0 if r.hg > r.ag else 0.5 if r.hg == r.ag else 0.0
            pe = 1 / (1 + 10 ** (-(elo[kh] + ELO_HA - elo[ka]) / 400))
            elo[kh] += ELO_K * (s - pe)
            elo[ka] -= ELO_K * (s - pe)
            hh.append((r.hg, r.ag))
            ha.append((r.ag, r.hg))
    ok = np.array([f is not None for f in feats])
    X = pd.DataFrame([f if f is not None else [np.nan] * len(BASE) for f in feats], columns=BASE)
    df = pd.concat([df, X], axis=1)
    df["ok"] = ok
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


def fit_predict(train, test, cols):
    if len(train) < 300 or len(test) == 0:
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


def choisir(E, seuil=SEUIL):
    mask = E >= seuil
    idx = np.where(mask, E, 9).argmin(axis=1)
    return idx, mask.any(axis=1)


FAMILIES = {
    "uk_odds": ("Championnats avec cotes", ["uk"], BASE + ODDS, True),
    "uk": ("Championnats sans cotes", ["uk"], BASE, False),
    "fd": ("Coupes d'Europe et Brésil", ["fd"], BASE, False),
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

    frames = load_uk() + load_fd()
    if not frames:
        raise SystemExit("Aucune donnée téléchargée.")
    raw = pd.concat(frames, ignore_index=True)
    raw["day"] = raw["dt"].dt.strftime("%Y-%m-%d")
    raw = raw.sort_values("fin", ascending=False).drop_duplicates(["div", "home", "away", "day"], keep="first")
    log("matchs terminés :", int(raw["fin"].sum()), "| à venir :", int((~raw["fin"]).sum()))
    df = pd.concat([build(raw[raw["div"] == d]) for d in raw["div"].unique()], ignore_index=True)

    # --- test sur le passé (modèle entraîné avant CUT, testé après)
    stats = {}
    for fam, (label, _, cols, _) in FAMILIES.items():
        m = family_mask(df, fam) & df["fin"]
        tr, te = df[m & (df["dt"] < CUT)], df[m & (df["dt"] >= CUT)]
        P = fit_predict(tr, te, cols)
        if P is None:
            continue
        E, Et = events(P), events_truth(truth(te))
        idx, has = choisir(E)
        if has.sum() == 0:
            continue
        win = Et[np.arange(len(te)), idx][has]
        stats[fam] = {"label": label, "matchs": int(len(te)), "paris": int(has.sum()), "taux": round(float(win.mean()) * 100, 1)}
        log(f"  test {label} : {has.sum()} paris sur {len(te)} matchs, réussite {win.mean()*100:.1f} %")

    # --- prédictions des matchs à venir
    up = df[~df["fin"]].copy()
    local = up["dt"] + pd.Timedelta(hours=1)
    keep = np.where(up["has_t"], up["dt"] >= now_utc, local.dt.normalize() >= today_b) & (local.dt.normalize() <= tomorrow_b)
    up = up[keep]
    picks = []
    ignores = 0
    for fam, (label, _, cols, _) in FAMILIES.items():
        trm = family_mask(df, fam) & df["fin"]
        um = up[family_mask(up, fam)]
        if fam == "uk":
            um = up[(up["src"] == "uk") & up["ok"] & ~up["has_odds"]]
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

    # --- historique des pronostics (enregistrés avant les matchs) et bilan
    hist_path = os.path.join(OUT, "data", "historique.csv")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    cols_h = ["cle", "genere", "jour", "ligue", "match", "home", "away", "pari", "k", "p", "resultat"]
    H = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame(columns=cols_h)
    for o in offre:
        cle = f"{o['jour']}|{o['home']}|{o['away']}|{o['pari']}"
        if cle not in set(H["cle"]):
            H.loc[len(H)] = [cle, now_b.strftime("%Y-%m-%d %H:%M"), o["jour"], o["ligue"], o["match"], o["home"], o["away"],
                             o["pari"], o["k"], o["p"], np.nan]
    fin = df[df["fin"]]
    res = {}
    for r in fin.itertuples():
        for dd in (r.dt, r.dt + pd.Timedelta(hours=1)):
            res[(r.home, r.away, dd.strftime("%Y-%m-%d"))] = (r.hg, r.ag)
    for i, r in H.iterrows():
        if pd.isna(r["resultat"]) and (r["home"], r["away"], r["jour"]) in res:
            hg, ag = res[(r["home"], r["away"], r["jour"])]
            t = events_truth(truth(pd.DataFrame({"hg": [hg], "ag": [ag]})))[0]
            H.at[i, "resultat"] = int(t[int(r["k"])])
    H.to_csv(hist_path, index=False)
    done = H[H["resultat"].notna()]
    bilan = {"n": int(len(done)), "gagnes": int(done["resultat"].astype(float).sum()),
             "recent": [{"jour": r.jour, "match": r.match, "pari": r.pari, "ok": bool(float(r.resultat))}
                        for r in done.sort_values("jour", ascending=False).head(12).itertuples()],
             "en_attente": int(H["resultat"].isna().sum())}

    data = {"genere": now_b.strftime("%d/%m/%Y à %H:%M"), "seuil": int(SEUIL * 100), "offre": offre,
            "ticket": ticket, "sur": sur_info, "stats": list(stats.values()), "bilan": bilan, "ignores": ignores}
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html"), encoding="utf-8").read()
    os.makedirs(os.path.join(OUT, "docs"), exist_ok=True)
    open(os.path.join(OUT, "docs", "index.html"), "w", encoding="utf-8").write(
        tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    json.dump(data, open(os.path.join(OUT, "docs", "offre.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"OK : {len(offre)} pronostics, ticket {'oui' if ticket else 'non'}, sûr {'oui' if sur_info else 'non'}")


if __name__ == "__main__":
    main()
