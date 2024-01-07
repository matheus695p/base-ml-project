import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from project.packages.modelling.models.unsupervised.segmentation import KMeansElbowSelector


class TestKMeansElbowSelector:
    @pytest.fixture
    def kmeans_elbow_selector(self):
        X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
        return KMeansElbowSelector(min_clusters=2, max_clusters=10).fit(X)

    def test_fit(self, kmeans_elbow_selector):
        assert kmeans_elbow_selector.is_fitted

    def test_predict(self, kmeans_elbow_selector):
        X, _ = make_blobs(n_samples=50, centers=4, random_state=42)
        labels = kmeans_elbow_selector.predict(X)
        assert len(labels) == X.shape[0]

    def test_find_optimal_num_clusters(self, kmeans_elbow_selector):
        optimal_num_clusters = kmeans_elbow_selector.find_optimal_num_clusters()
        assert isinstance(optimal_num_clusters, int)
        assert optimal_num_clusters >= kmeans_elbow_selector.min_clusters
        assert optimal_num_clusters <= kmeans_elbow_selector.max_clusters

    def test_get_inertia_plot(self, kmeans_elbow_selector):
        plot = kmeans_elbow_selector.get_inertia_plot()
        assert plot is not None

    def test_get_feature_imp_wcss_min(self, kmeans_elbow_selector):
        feature_importance = kmeans_elbow_selector.get_feature_imp_wcss_min()
        assert isinstance(feature_importance, pd.DataFrame)
        assert not feature_importance.empty

    def test_get_feature_imp_unsup2sup(self, kmeans_elbow_selector):
        feature_importance = kmeans_elbow_selector.get_feature_imp_unsup2sup()
        assert isinstance(feature_importance, pd.DataFrame)
        assert not feature_importance.empty
