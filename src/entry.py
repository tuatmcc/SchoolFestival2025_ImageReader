import argparse
import json
from aruco import ArucoFinder
from camera import Camera
from reader import ImageReader


parser = argparse.ArgumentParser(description="Run the FastAPI application.")
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="settings/config.json",
    help="Path to the configuration file (default: settings/config.json)",
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    finder = ArucoFinder()
    camera = Camera()
    reader = ImageReader(config, camera, finder, debug=args.debug)

    print("Starting image reader...")
    print("Press 'q' to quit.")

    reader.run()
