"""Train the answer quality classifier (DeBERTa-v3-small) on labeled_asqa.csv."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.data_split import load_and_split
from src.filtering.learned_filter import train_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    train_df, val_df, test_df = load_and_split("data/asqa/labeled_asqa.csv")

    logger.info("Starting classifier training …")
    model_path = train_classifier(
        train_df=train_df,
        val_df=val_df,
        config_overrides={
            "batch_size": 4,
        },
    )
    logger.info("Training complete. Model saved to %s", model_path)


if __name__ == "__main__":
    main()
