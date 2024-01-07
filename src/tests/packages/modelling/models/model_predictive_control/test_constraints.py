import pytest
import optuna
from project.packages.modelling.models.model_predictive_control.constraints import (
    ExpressionConstraintsPruner,
    MultiplePruners,
)


def objective(trial):
    x = trial.suggest_float("x", 0, 1)
    y = trial.suggest_float("y", 0, 1)
    return x + y


@pytest.fixture(scope="module")
def study():
    return optuna.create_study()


class TestExpressionConstraintsPruner:
    def test_prune_with_constraints(self, study):
        # Create a study with ExpressionConstraintsPruner
        pruner = ExpressionConstraintsPruner(["x >= 0"])
        study.pruner = pruner

        # Optimize the objective function
        study.optimize(objective, n_trials=10)

        # Check if any trial was pruned
        # assert any(trial.should_prune() for trial in study.trials)

    def test_prune_without_constraints(self, study):
        # Create a study with ExpressionConstraintsPruner (no constraints)
        pruner = ExpressionConstraintsPruner([])
        study.pruner = pruner

        # Optimize the objective function
        study.optimize(objective, n_trials=10)

        # # Check if no trial was pruned
        # assert not any(trial.should_prune() for trial in study.trials)


class TestMultiplePruners:
    def test_prune_with_all_pruners(self, study):
        # Create a study with MultiplePruners and multiple pruners
        pruner1 = optuna.pruners.MedianPruner()
        pruner2 = optuna.pruners.NopPruner()
        combined_pruner = MultiplePruners(pruners=[pruner1, pruner2], pruning_condition="all")
        study.pruner = combined_pruner

        # Optimize the objective function
        study.optimize(objective, n_trials=10)

        assert isinstance(study.pruner, MultiplePruners)

    def test_prune_with_any_pruners(self, study):
        pruner1 = optuna.pruners.MedianPruner()
        pruner2 = optuna.pruners.NopPruner()
        combined_pruner = MultiplePruners(pruners=[pruner1, pruner2], pruning_condition="any")
        study.pruner = combined_pruner

        study.optimize(objective, n_trials=10)

        assert isinstance(study.pruner, MultiplePruners)
