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
        return self._pruning_condition_check_fn(
            pruner.prune(study, trial) for pruner in self._pruners
        )
