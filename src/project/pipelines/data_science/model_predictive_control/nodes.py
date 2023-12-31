from project.packages.modelling.models.model_predictive_control.explorer import (
    ModelPredictiveControlExplorer,
)


def model_predictive_control_exploration(df, model, params, data_schemas, *preprocessors):
    preprocessors = list(preprocessors)
    model_explorer = ModelPredictiveControlExplorer(
        model=model,
        preprocessors=preprocessors,
        params=params,
        data_schemas=data_schemas,
    )
    model_explorer.fit(df)
    return model_explorer
