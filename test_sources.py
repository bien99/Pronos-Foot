"""Test des sources de données pour Argentine, Mexique, Colombie, Brésil. Affiche un résumé court."""
import os, io, json, requests, pandas as pd

def get(url, headers=None):
    try:
        return requests.get(url, headers=headers, timeout=40)
    except Exception as e:
        print("   erreur", e)

print("=== 1. Historique football-data.co.uk (dossier new) ===")
for code in ["ARG", "BRA", "MEX", "COL", "USA"]:
    r = get(f"https://www.football-data.co.uk/new/{code}.csv")
    if r is None or r.status_code != 200:
        print(code, "indisponible", None if r is None else r.status_code); continue
    df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
    d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    print(code, len(df), "lignes | du", d.min().date(), "au", d.max().date(), "| colonnes:", ",".join(df.columns[:16]))

print("\n=== 2. Matchs à venir TheSportsDB (clé gratuite 123) ===")
for pays in ["Argentina", "Mexico", "Colombia", "Brazil"]:
    r = get(f"https://www.thesportsdb.com/api/v1/json/123/search_all_leagues.php?c={pays}&s=Soccer")
    if r is None or r.status_code != 200:
        print(pays, "indisponible", None if r is None else r.status_code); continue
    try:
        ls = r.json().get("countries") or []
    except Exception:
        print(pays, "réponse illisible"); continue
    print(pays, [(l["idLeague"], l["strLeague"]) for l in ls][:8])
    for l in ls[:3]:
        e = get(f"https://www.thesportsdb.com/api/v1/json/123/eventsnextleague.php?id={l['idLeague']}")
        try:
            ev = (e.json().get("events") or [])
        except Exception:
            ev = []
        print("   ", l["strLeague"], "->", len(ev), "matchs à venir", [(x.get("dateEvent"), x.get("strTime"), x.get("strHomeTeam"), x.get("strAwayTeam")) for x in ev[:2]])

print("\n=== 3. football-data.org : compétitions accessibles avec ta clé ===")
r = get("https://api.football-data.org/v4/competitions", {"X-Auth-Token": os.environ.get("FD_KEY", "")})
if r is not None and r.status_code == 200:
    print([(c["code"], c["area"]["name"]) for c in r.json()["competitions"]])
else:
    print("indisponible", None if r is None else r.status_code)
