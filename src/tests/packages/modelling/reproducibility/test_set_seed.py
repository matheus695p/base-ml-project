import numpy as np
from project.packages.modelling.reproducibility.set_seed import seed_file


class TestSeedFile:
    def test_seed_file_without_message(self, caplog):
        seed = 123
        seed_file(seed, verbose=False)

        assert np.random.get_state()[1][0] == seed

        assert "Seeding sklearn, numpy and random libraries with the seed" not in caplog.text

    def test_seed_file_with_message(self, caplog):
        seed = 456
        message = "Custom seed message"
        seed_file(seed, contain_message=True, message=message, verbose=True)

        assert np.random.get_state()[1][0] == seed
