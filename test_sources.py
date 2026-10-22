"""Test 3 : Europa League et Conference League via ESPN (matchs à venir + anciens résultats)."""
import datetime as dt, requests

def get(url):
    try:
        return requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        print("   erreur", e)

def jeudis(depuis, n):
    d = depuis
    while d.weekday() != 3:
        d += dt.timedelta(days=1)
    return [d + dt.timedelta(days=7 * i) for i in range(n)]

today = dt.date.today()
for nom, slug in [("Europa League", "uefa.europa"), ("Conference League", "uefa.europa.conf")]:
    print(f"=== {nom} ({slug}) ===")
    base = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
    r = get(base)
    if r is None or r.status_code != 200:
        print("indisponible", None if r is None else r.status_code); continue
    ev = r.json().get("events", [])
    print("à venir (sans date) :", len(ev), [(e["date"], [c["team"]["displayName"] for c in e["competitions"][0]["competitors"]]) for e in ev[:2]])
    total = 0
    for d in jeudis(dt.date(2025, 9, 18), 8) + jeudis(dt.date(2024, 9, 19), 4):
        x = get(base + f"?dates={d:%Y%m%d}")
        if x is None or x.status_code != 200:
            print("  ", d, "erreur", None if x is None else x.status_code); continue
        e2 = x.json().get("events", [])
        finis = [e for e in e2 if e["status"]["type"]["name"] == "STATUS_FULL_TIME"]
        total += len(finis)
        ex = ""
        if finis:
            c = finis[0]["competitions"][0]["competitors"]
            ex = [(k["team"]["displayName"], k.get("score")) for k in c]
        print("  ", d, len(e2), "matchs,", len(finis), "terminés", ex)
    print("TOTAL matchs terminés trouvés :", total)
