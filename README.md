# Navier-Stokes Collapsing-Vortex Streamlit Prototype

This application is an interactive numerical reconstruction of a bounded-energy collapsing-vortex mechanism. It is intended for education and exploratory analysis, not as an independent proof of finite-time blow-up.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Included views

- Axisymmetric collapsing core with radial, azimuthal, and axial velocity
- 3D cylindrical slices
- Continuity and manufactured momentum-force audits
- Reduced annular pulse / Reynolds-stress illustration
- Grid-dependent scaling study for peak speed, energy, radius, and divergence

## Important limitation

The program uses smooth trial profiles based on the reported similarity scaling. It does not implement the complete high-order analytical construction or certify a singularity.
