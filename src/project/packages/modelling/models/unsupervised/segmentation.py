import logging

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from ....python_utils.typing import Vector, Matrix

logger = logging.getLogger(__name__)


class KMeansElbowSelector(BaseEstimator, TransformerMixin, ClusterMixin):
    """KMeansElbowSelector.

    KMeansElbowSelector selects the optimal number of clusters for K-Means clustering
    using the elbow method.

    Args:
        max_clusters : int, optional (default=10)
            The maximum number of clusters to consider when finding the optimal number
            of clusters using the elbow method.

    Attributes:
        optimal_num_clusters : int
            The optimal number of clusters selected using the elbow method.

        fitted : bool
            Indicates whether the model has been fitted.

    Examples:
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    >>> # Create and set up the ElbowClusterSelector instance
    >>> cluster_selector = ElbowClusterSelector(max_clusters=10)

    >>> # Fit the ElbowClusterSelector
    >>> cluster_selector.fit(X)

    >>> # Create a pipeline with StandardScaler and the cluster selector
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipeline = Pipeline([
    ...     ('scaler', StandardScaler()),  # Example pre-processing step
    ...     ('cluster_selector', cluster_selector),
    ... ])

    >>> # Fit the entire pipeline
    >>> pipeline.fit(X)

    >>> # Now you can use the entire pipeline for predictions, e.g., pipeline.predict(X)
    """

    def __init__(self, min_clusters=2, max_clusters=10):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.is_fitted = False
        self.model_args = {"init": "k-means++", "random_state": 42}

    def fit(self, X: Matrix, y: Vector = None) -> "KMeansElbowSelector":
        """Fit the KMeansElbowSelector to the input data.

        This method fits the KMeansElbowSelector to the input data, identifies the optimal
        number of clusters using the elbow method, and stores relevant attributes.

        Args:
            X (Matrix): The input feature matrix.
            y (Vector, optional): The target variable. Defaults to None.

        Returns:
            KMeansElbowSelector: The fitted instance of the class.

        """
        wcss = []  # Within-cluster sum of squares
        graph = []
        for i in range(self.min_clusters, self.max_clusters + 1):
            logger.debug(f"Performing clusters with {i} clusters")
            model = KMeans(n_clusters=i, **self.model_args)
            model.fit(X)
            wcss.append(model.inertia_)
            graph.append([i, model.inertia_])
        self.graph = graph

        # Use the elbow method to find the optimal number of clusters
        self.optimal_num_clusters = self.find_optimal_num_clusters()
        logger.info(f"Optimal number of clusters: {self.optimal_num_clusters}")
        # Fit the K-Means model with the optimal number of clusters
        self.model = KMeans(n_clusters=self.optimal_num_clusters, **self.model_args)
        self.model = self.model.fit(X)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        self.train_data = X
        self._get_cluster_centroids_index()
        self.get_feature_imp_wcss_min()
        self.get_feature_imp_unsup2sup()

        return self

    def predict(self, X: Matrix) -> Vector:
        """Predict cluster labels for input data.

        This method assigns cluster labels to the input data based on the fitted K-Means model.

        Args:
            X (Matrix): The input feature matrix.

        Returns:
            Vector: Cluster labels assigned to the input data.

        """
        # Assign cluster labels to the input data
        cluster_labels = self.model.predict(X)
        return cluster_labels

    def _get_cluster_centroids_index(self):
        """Get the index of data points nearest to cluster centroids.

        This method calculates the data points in the training data that are closest to each
        cluster's centroid and stores their indexes.

        Returns:
            KMeansElbowSelector: The instance of the class with centroid indexes.

        """
        # Get the cluster centroids
        self.centroid_indexes = {}
        if isinstance(self.train_data, pd.DataFrame):
            iterable_data = self.train_data.values
        else:
            iterable_data = self.train_data

        for centroid in self.model.cluster_centers_:
            min_centroid_distance = np.inf

            for i, data_point in enumerate(iterable_data):
                distance = euclidean(centroid, data_point)
                if distance < min_centroid_distance:
                    centroid_index = i
                    min_centroid_distance = distance
            label = self.labels_[centroid_index]
            self.centroid_indexes[f"cluster_id_{label}"] = centroid_index
        logger.info(f"Centroids dictionary -> {self.centroid_indexes}")
        return self

    def get_feature_imp_wcss_min(self) -> pd.DataFrame:
        """Get feature importances using the WCSS method.

        This method calculates feature importances for each cluster using the WCSS method
        and stores the results.

        Returns:
            pd.DataFrame: A DataFrame containing feature importances.

        """
        labels = self.model.n_clusters
        centroids = self.model.cluster_centers_

        if isinstance(self.train_data, pd.DataFrame):
            self.ordered_feature_names = self.train_data.columns
        else:
            self.ordered_feature_names = pd.DataFrame(self.train_data).columns

        centroids = np.vectorize(lambda x: np.abs(x))(centroids)
        sorted_centroid_features_idx = centroids.argsort(axis=1)[:, ::-1]

        dfs = []
        for label, centroid in zip(range(labels), sorted_centroid_features_idx):
            ordered_cluster_feature_weights = centroids[label][sorted_centroid_features_idx[label]]
            ordered_cluster_features = [self.ordered_feature_names[feature] for feature in centroid]

            df_importance = pd.DataFrame(
                [ordered_cluster_features, ordered_cluster_feature_weights]
            ).T
            df_importance.columns = ["feature", f"cluster_id_{label}"]
            df_importance[f"cluster_id_{label}"] = (
                df_importance[f"cluster_id_{label}"]
                / df_importance[f"cluster_id_{label}"].sum()
                * 100
            )
            dfs.append(df_importance.set_index("feature"))

        cluster_feature_weights = pd.concat(dfs, axis=1)
        self.wcss_feature_importances_ = cluster_feature_weights
        self.wcss_feature_importances_["mean_importance"] = self.wcss_feature_importances_[
            list(self.wcss_feature_importances_.columns)
        ].sum(axis=1)
        self.wcss_feature_importances_["mean_importance"] = (
            self.wcss_feature_importances_["mean_importance"]
            / self.wcss_feature_importances_["mean_importance"].sum()
            * 100
        )
        self.wcss_feature_importances_ = self.wcss_feature_importances_.sort_values(
            by="mean_importance", ascending=False
        )

        logger.debug("WCSS feature importances: ")
        logger.debug(self.wcss_feature_importances_.head(20))

        for cluster_id in self.wcss_feature_importances_:
            data = self.wcss_feature_importances_[[cluster_id]]
            data = data.sort_values(by=cluster_id, ascending=False)
            top_predictors = list(data.head(10).index)
            logger.debug(
                f"Method: WCSS importance | Top 10 predictors for cluster {cluster_id}: {top_predictors}"
            )
            logger.debug("\n\n")

        return self.wcss_feature_importances_

    def get_feature_imp_unsup2sup(self) -> pd.DataFrame:
        """Get feature importances using unsupervised-to-supervised conversion.

        This method calculates feature importances for each cluster using an
        unsupervised-to-supervised conversion approach and stores the results.

        Returns:
            pd.DataFrame: A DataFrame containing feature importances.

        """
        if isinstance(self.train_data, pd.DataFrame):
            self.ordered_feature_names = self.train_data.columns
        else:
            self.ordered_feature_names = pd.DataFrame(self.train_data).columns

        X = self.train_data

        dfs = []
        for label in range(self.model.n_clusters):
            binary_enc = np.vectorize(lambda x: 1 if x == label else 0)(self.labels_)
            # overfitt classifier to get feature importance
            clf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=12)
            clf.fit(X, binary_enc)

            sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1]
            ordered_cluster_features = np.take_along_axis(
                np.array(self.ordered_feature_names),
                sorted_feature_weight_idxes,
                axis=0,
            )
            ordered_cluster_feature_weights = np.take_along_axis(
                np.array(clf.feature_importances_), sorted_feature_weight_idxes, axis=0
            )

            df_importance = pd.DataFrame(
                [ordered_cluster_features, ordered_cluster_feature_weights]
            ).T
            df_importance.columns = ["feature", f"cluster_id_{label}"]
            df_importance[f"cluster_id_{label}"] = (
                df_importance[f"cluster_id_{label}"]
                / df_importance[f"cluster_id_{label}"].sum()
                * 100
            )
            dfs.append(df_importance.set_index("feature"))

        cluster_feature_weights = pd.concat(dfs, axis=1)

        self.unsupervised_feature_importance_ = cluster_feature_weights
        self.unsupervised_feature_importance_[
            "mean_importance"
        ] = self.unsupervised_feature_importance_[
            list(self.unsupervised_feature_importance_.columns)
        ].sum(
            axis=1
        )
        self.unsupervised_feature_importance_["mean_importance"] = (
            self.unsupervised_feature_importance_["mean_importance"]
            / self.unsupervised_feature_importance_["mean_importance"].sum()
            * 100
        )
        self.unsupervised_feature_importance_ = self.unsupervised_feature_importance_.sort_values(
            by="mean_importance", ascending=False
        )

        logger.debug("Unsupervised method feature importances: ")
        logger.debug(self.unsupervised_feature_importance_.head(20))

        for cluster_id in self.unsupervised_feature_importance_:
            data = self.unsupervised_feature_importance_[[cluster_id]]
            data = data.sort_values(by=cluster_id, ascending=False)
            top_predictors = list(data.head(10).index)
            logger.debug(
                f"Method: Unsupervised importance | Top 10 predictors for cluster {cluster_id}: {top_predictors}"
            )
            logger.debug("\n\n")

        return self.unsupervised_feature_importance_

    def find_optimal_num_clusters(
        self,
    ):
        """Find the optimal number of clusters using the elbow method.

        This method calculates the optimal number of clusters using the elbow method based
        on the within-cluster sum of squares (WCSS).

        Returns:
            int: The optimal number of clusters.

        """
        graph = pd.DataFrame(self.graph, columns=["number_of_clusters", "inertia"])
        # Calculate the first derivative of the change in wcss
        graph["change_in_wcss"] = graph["inertia"].diff()
        graph["second_derivative"] = graph["change_in_wcss"].diff()
        # Find the index where the second derivative is maximum
        optimal_num_clusters_index = graph["second_derivative"].idxmax()
        # The optimal number of clusters can be obtained from the 'number_of_clusters' column
        optimal_num_clusters = graph.loc[optimal_num_clusters_index, "number_of_clusters"]
        return optimal_num_clusters

    def get_inertia_plot(
        self,
    ) -> px.line:
        """Get an inertia plot.

        This method generates a plot of inertia values for different numbers of clusters
        and highlights the optimal number of clusters selected using the elbow method.

        Returns:
            px.line: A Plotly line chart.

        Raises:
            ValueError: If the model has not been fitted yet.

        """
        if self.is_fitted:
            title = f"k-Means Inertia Plot | Optimal number of cluster is {self.optimal_num_clusters} segments"
            graph = pd.DataFrame(self.graph, columns=["number_of_clusters", "inertia"])
            fig = px.line(
                graph,
                x="number_of_clusters",
                y="inertia",
                title=title,
            )
            fig.add_vline(
                x=self.optimal_num_clusters,
                line_dash="dash",
                line_color="black",
                annotation_text="Optimal number of clusters",
            )
            return fig
        else:
            raise ValueError(
                "The model has not been fitted yet. You should fit before getting inertia plot."
            )
