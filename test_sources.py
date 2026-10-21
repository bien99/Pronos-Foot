"""Test 2 : matchs à venir (ESPN) pour Argentine, Mexique, Colombie, Brésil + correspondance des noms."""
import io, datetime as dt, requests, pandas as pd

def get(url):
    try:
        return requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        print("   erreur", e)

LIGUES = {"ARG": "arg.1", "MEX": "mex.1", "COL": "col.1", "BRA": "bra.1"}
today = dt.date.today()
rng = f"{today:%Y%m%d}-{today + dt.timedelta(days=3):%Y%m%d}"
for code, slug in LIGUES.items():
    print(f"=== {code} ({slug}) ===")
    h = get(f"https://www.football-data.co.uk/new/{code}.csv")
    df = pd.read_csv(io.BytesIO(h.content), encoding="latin-1")
    df.columns = [c.replace("ï»¿", "") for c in df.columns]
    noms = set(df.tail(600)["Home"]) | set(df.tail(600)["Away"])
    r = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={rng}")
    if r is None or r.status_code != 200:
        print("ESPN indisponible", None if r is None else r.status_code); continue
    ev = r.json().get("events", [])
    print(len(ev), "matchs à venir (4 jours)")
    ok = 0; tot = 0; mal = []
    for e in ev:
        c = e["competitions"][0]["competitors"]
        home = [x for x in c if x["homeAway"] == "home"][0]["team"]["displayName"]
        away = [x for x in c if x["homeAway"] == "away"][0]["team"]["displayName"]
        for n in (home, away):
            tot += 1
            if n in noms: ok += 1
            else: mal.append(n)
    for e in ev[:3]:
        c = e["competitions"][0]["competitors"]
        print("  ", e["date"], [x["team"]["displayName"] for x in c], e["status"]["type"]["name"])
    print("noms identiques à l'historique :", ok, "/", tot, "| différents :", mal[:6])
