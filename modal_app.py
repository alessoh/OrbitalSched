"""
Modal deployment for the OrbitalSched FastAPI backend.

Wraps the existing orbitalsched.api.server:app and serves it from Modal's
cloud with a stable public URL that does not change between restarts.

Deploy with:
    modal deploy modal_app.py

After deployment, Modal prints a URL like:
    https://alessoh--orbitalsched-fastapi-app.modal.run
That URL is what you put into the Vercel environment variables.
"""

from __future__ import annotations

import modal

# Build the container image: Python 3.12 plus all the packages our backend uses.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26",
        "scipy>=1.12",
        "gymnasium>=0.29",
        "stable-baselines3>=2.3",
        "torch>=2.3",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.6",
    )
    # Copy the local orbitalsched package and the trained checkpoint into
    # the container image so they are available at runtime.
    .add_local_python_source("orbitalsched")
    .add_local_file(
        "checkpoints/final_policy.zip",
        remote_path="/root/checkpoints/final_policy.zip",
    )
)

app = modal.App(name="orbitalsched", image=image)


@app.function(
    cpu=2,
    memory=2048,
    min_containers=1,
    timeout=3600,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the existing FastAPI app over Modal's HTTPS endpoint."""

    import os

    # Tell the server where to find the checkpoint inside the container.
    os.environ["ORBITALSCHED_CHECKPOINT"] = "/root/checkpoints/final_policy.zip"

    from orbitalsched.api.server import app as fastapi_application

    return fastapi_application