import optuna


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

    def _prune(self, args):
        for expr in self.constraints:
            for feature, value in args.items():
                expr = expr.replace(feature, str(value))
            is_constraint_true = eval(expr)
            if not is_constraint_true:
                return False  # prune the trial if any constraint is not met.
        return True  # don't prune the trial if all constraints are met.
