"""Empaqueta el lliurament: submission.zip amb agent.py + paràmetres + models a l'arrel."""
import zipfile

from core import ROOT

FILES = ["agent.py", "params.json", "forecaster_co2.txt", "forecaster_preu.txt"]
out = ROOT / "submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT / "submission" / f, f)
print(out, f"{out.stat().st_size/1e6:.2f} MB")
