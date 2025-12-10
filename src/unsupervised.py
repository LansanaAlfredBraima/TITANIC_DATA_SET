"""
Unsupervised Learning Module for Titanic Dataset
Implements clustering algorithms and dimensionality reduction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class UnsupervisedModels:
    """Class to handle unsupervised learning tasks"""
    
    def __init__(self):
        self.clustering_models = {}
        self.clustering_results = {}
        self.pca_model = None
        self.pca_results = {}
        
    def find_optimal_clusters(self, X, max_clusters=10, method='elbow'):
        """
        Find optimal number of clusters using elbow method or silhouette analysis
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature data
        max_clusters : int
            Maximum number of clusters to test
        method : str
            'elbow' or 'silhouette'
            
        Returns:
        --------
        dict
            Results for different cluster numbers
        """
        results = {
            'n_clusters': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        print(f"\n{'='*60}")
        print(f"FINDING OPTIMAL NUMBER OF CLUSTERS ({method.upper()} METHOD)")
        print(f"{'='*60}")
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['n_clusters'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette'].append(silhouette_score(X, labels))
            results['davies_bouldin'].append(davies_bouldin_score(X, labels))
            results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, "
                  f"Silhouette={results['silhouette'][-1]:.3f}")
        
        return results
    
    def plot_elbow_curve(self, results, save_path=None):
        """
        Plot elbow curve for K-means
        
        Parameters:
        -----------
        results : dict
            Results from find_optimal_clusters
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        axes[0].plot(results['n_clusters'], results['inertia'], 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Silhouette score
        axes[1].plot(results['n_clusters'], results['silhouette'], 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Elbow curve saved to {save_path}")
        
        plt.show()
    
    def perform_kmeans(self, X, n_clusters=3, random_state=42):
        """
        Perform K-means clustering
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature data
        n_clusters : int
            Number of clusters
        random_state : int
            Random seed
            
        Returns:
        --------
        dict
            Clustering results
        """
        print(f"\n{'='*60}")
        print(f"K-MEANS CLUSTERING (k={n_clusters})")
        print(f"{'='*60}")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        results = {
            'model': kmeans,
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters
        }
        
        self.clustering_models['kmeans'] = kmeans
        self.clustering_results['kmeans'] = results
        
        print(f"\n✓ K-means clustering completed")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  Cluster Sizes:")
        for cluster, count in zip(unique, counts):
            print(f"    Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return results
    
    def perform_hierarchical_clustering(self, X, n_clusters=3, linkage_method='ward'):
        """
        Perform hierarchical clustering
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature data
        n_clusters : int
            Number of clusters
        linkage_method : str
            Linkage method ('ward', 'complete', 'average', 'single')
            
        Returns:
        --------
        dict
            Clustering results
        """
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL CLUSTERING (k={n_clusters}, method={linkage_method})")
        print(f"{'='*60}")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hierarchical.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        results = {
            'model': hierarchical,
            'labels': labels,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters,
            'linkage_method': linkage_method
        }
        
        self.clustering_models['hierarchical'] = hierarchical
        self.clustering_results['hierarchical'] = results
        
        print(f"\n✓ Hierarchical clustering completed")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  Cluster Sizes:")
        for cluster, count in zip(unique, counts):
            print(f"    Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        return results
    
    def plot_dendrogram(self, X, save_path=None, max_display=30):
        """
        Plot dendrogram for hierarchical clustering
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature data
        save_path : str, optional
            Path to save the plot
        max_display : int
            Maximum number of samples to display
        """
        print(f"\n{'='*60}")
        print("GENERATING DENDROGRAM")
        print(f"{'='*60}")
        
        # Compute linkage
        Z = linkage(X, method='ward')
        
        plt.figure(figsize=(12, 6))
        dendrogram(Z, truncate_mode='lastp', p=max_display, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster Size', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dendrogram saved to {save_path}")
        
        plt.show()
    
    def apply_pca(self, X, n_components=None, variance_threshold=0.95):
        """
        Apply PCA for dimensionality reduction
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature data
        n_components : int, optional
            Number of components (if None, use variance_threshold)
        variance_threshold : float
            Cumulative variance threshold
            
        Returns:
        --------
        dict
            PCA results
        """
        print(f"\n{'='*60}")
        print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print(f"{'='*60}")
        
        # Determine number of components
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            print(f"  Selected {n_components} components for {variance_threshold*100}% variance")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        
        results = {
            'model': pca,
            'transformed_data': X_pca,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components': n_components,
            'components': pca.components_
        }
        
        self.pca_model = pca
        self.pca_results = results
        
        print(f"\n✓ PCA completed")
        print(f"  Original dimensions: {X.shape[1]}")
        print(f"  Reduced dimensions: {n_components}")
        print(f"  Total variance explained: {results['cumulative_variance'][-1]:.2%}")
        
        print(f"\n  Variance explained by each component:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"    PC{i+1}: {var:.2%}")
        
        return results
    
    def plot_pca_variance(self, pca_results, save_path=None):
        """
        Plot PCA variance explanation
        
        Parameters:
        -----------
        pca_results : dict
            Results from apply_pca
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        n_components = len(pca_results['explained_variance_ratio'])
        components = range(1, n_components + 1)
        
        # Individual variance
        axes[0].bar(components, pca_results['explained_variance_ratio'], color='steelblue')
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[0].set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Cumulative variance
        axes[1].plot(components, pca_results['cumulative_variance'], 'ro-', linewidth=2, markersize=8)
        axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
        axes[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PCA variance plot saved to {save_path}")
        
        plt.show()
    
    def plot_clusters_2d(self, X, labels, title='Cluster Visualization', save_path=None):
        """
        Plot clusters in 2D (uses first 2 features or PCA)
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature data
        labels : array
            Cluster labels
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        # If more than 2 features, use PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_plot = pca.fit_transform(X)
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'
        else:
            X_plot = X
            xlabel = 'Feature 1'
            ylabel = 'Feature 2'
        
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cluster plot saved to {save_path}")
        
        plt.show()
    
    def analyze_clusters(self, df, labels, cluster_name='Cluster'):
        """
        Analyze cluster characteristics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original dataframe with features
        labels : array
            Cluster labels
        cluster_name : str
            Name for the cluster column
            
        Returns:
        --------
        pd.DataFrame
            Cluster analysis summary
        """
        df_analysis = df.copy()
        df_analysis[cluster_name] = labels
        
        print(f"\n{'='*60}")
        print("CLUSTER ANALYSIS")
        print(f"{'='*60}")
        
        # Numerical features summary
        numerical_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        if cluster_name in numerical_cols:
            numerical_cols.remove(cluster_name)
        
        cluster_summary = df_analysis.groupby(cluster_name)[numerical_cols].mean()
        
        print("\n" + "-"*60)
        print("CLUSTER MEANS (Numerical Features)")
        print("-"*60)
        print(cluster_summary.to_string())
        
        return cluster_summary


if __name__ == "__main__":
    print("Unsupervised Learning module loaded successfully!")
    print("Available class: UnsupervisedModels")
    print("Available methods:")
    print("  - find_optimal_clusters()")
    print("  - perform_kmeans()")
    print("  - perform_hierarchical_clustering()")
    print("  - apply_pca()")
    print("  - plot_clusters_2d()")
    print("  - analyze_clusters()")
