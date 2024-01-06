import optuna
import typing as tp


class ExpressionConstraintsPruner(optuna.pruners.BasePruner):
    def __init__(self, constraints: list[str] | str):
        super().__init__()
        self.constraints = constraints

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        """Judge whether the trial should be pruned based on the reported values.

        Note that this method is not supposed to be called by library users. Instead,
        :func:`optuna.trial.Trial.report` and :func:`optuna.trial.Trial.should_prune` provide
        user interfaces to implement pruning mechanism in an objective function.

        Args:
            study:
                Study object of the target study.
            trial:
                FrozenTrial object of the target trial.
                Take a copy before modifying this object.

        Returns:
            A boolean value representing whether the trial should be pruned.
        """
        args = {}
        for key, value in trial.params.items():
            args[key] = value
        should_not_prune = self._prune(args)
        if should_not_prune:
            return False  # don't prune the trial if the constraints are met.
        else:
            return True  # prune the trial if the constraints are not met.

    def _prune(self, args: tp.Dict[str, float]) -> bool:
        for expr in self.constraints:
            for feature, value in args.items():
                expr = expr.replace(feature, str(value))
            is_constraint_true = eval(expr)
            if not is_constraint_true:
                return False  # prune the trial if any constraint is not met.
        return True  # don't prune the trial if all constraints are met.


class MultiplePruners(optuna.pruners.BasePruner):
    def __init__(
        self,
        pruners: tp.Iterable[optuna.pruners.BasePruner],
        pruning_condition: str = "any",
    ) -> None:
        """A custom Optuna pruner that combines multiple pruners with a specified condition.

        This class allows you to combine multiple Optuna pruners and apply a specified
        condition to determine whether a trial should be pruned or not.

        Args:
            pruners (Iterable[optuna.pruners.BasePruner]): An iterable of Optuna pruners to
                be combined.
            pruning_condition (str, optional): The condition to be applied when combining
            pruners. It can be one of 'any' (prune if any pruner suggests pruning) or 'all'
            (prune if all pruners suggest pruning). Defaults to 'any'.

        Raises:
            ValueError: If an invalid `pruning_condition` is provided.

        Attributes:
            _pruners (tuple): A tuple of the provided Optuna pruners.
            _pruning_condition_check_fn (function): A function that represents the specified
                pruning condition ('any' or 'all').

        """
        self._pruners = tuple(pruners)

        self._pruning_condition_check_fn = None
        if pruning_condition == "any":
            self._pruning_condition_check_fn = any
        elif pruning_condition == "all":
            self._pruning_condition_check_fn = all
        else:
            raise ValueError(f"Invalid pruning ({pruning_condition}) condition passed!")
        assert self._pruning_condition_check_fn is not None

    def prune(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> bool:
        """Determine whether to prune a trial based on the combined pruners and condition.

        This method combines the provided pruners and applies the specified condition to
        determine whether a trial should be pruned or not.

        Args:
            study (optuna.study.Study): The Optuna study object.
            trial (optuna.trial.FrozenTrial): The frozen trial to be pruned.

        Returns:
            bool: True if the trial should be pruned, False otherwise.

        """
        return self._pruning_condition_check_fn(
            pruner.prune(study, trial) for pruner in self._pruners
        )
