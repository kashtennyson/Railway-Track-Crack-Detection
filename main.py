import os
import argparse
from src.train import train
from src.evaluate import evaluate
from src.logger import ExperimentLogger

# Supress info and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    parser = argparse.ArgumentParser(description="Railway Track Crack Detection")
    
    # Add arguments
    parser.add_argument(
        "action", 
        choices=["train", "evaluate", "all"],
        help="Action to perform: train the model, evaluate it, or do both."
    )

    args = parser.parse_args()

    try:
        if args.action == "train":
            train()
        elif args.action == "evaluate":
            evaluate()
        elif args.action == "all":
            print("--- Starting Full Pipeline ---")
            train()
            print("\n--- Training Complete. Starting Evaluation ---")
            evaluate()
    finally:
        ExperimentLogger.finish()

if __name__ == "__main__":
    main()