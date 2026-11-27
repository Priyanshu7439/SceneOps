"""Start the optional SceneOps Ray Serve narration deployment."""

from __future__ import annotations

import argparse

from sceneops.serving.ray_serve_app import serve_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--address", default="127.0.0.1:8000")
    args = parser.parse_args()
    serve_model(model_id=args.model_id, address=args.address)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
