"""Fixed seed for reproducibility."""
import logging
import random

import numpy as np
import sklearn

logger = logging.getLogger(__name__)

GLOBAL_SEED = 42


def seed_file(
    seed: int = GLOBAL_SEED,
    contain_message: bool = False,
    message="",
    verbose: bool = True,
):
    """It seeds the random number generator of the three libraries.

    Args:
      seed (int): int
      contain_message (bool): bool. Defaults to False
      message: If True, print the message.
      verbose: If True, print the message.

    Returns: None
    """
    sklearn.utils.check_random_state(seed)
    np.random.seed(seed)
    random.seed(seed)
    if verbose:
        if not contain_message:
            logger.info(f"Seeding sklearn, numpy and random libraries with the seed {seed}")
        else:
            logger.info(f"{message}")
