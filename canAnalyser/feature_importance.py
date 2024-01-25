from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
import logging
logging.basicConfig(level=logging.INFO)


def plot_decision_boundary(X, y, model, resolution=0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)


class FeatureImportance:
    def __init__(self, feature_vectors, cluster_labels):
        self.feature_vectors = feature_vectors
        self.cluster_labels = cluster_labels

    def calculate_importances(self, n_estimators=100, random_state=0):
        """
        Calculate feature importances using XGBoost classifier.

        Parameters:
        n_estimators (int): The number of boosting rounds.
        random_state (int): Random state for reproducibility.

        Returns:
        DataFrame: A DataFrame containing feature importances.
        """
        xgb = XGBClassifier(n_estimators=n_estimators, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(self.feature_vectors, self.cluster_labels)
        importances = pd.DataFrame({'Feature': self.feature_vectors.columns,
                                    'Importance': xgb.feature_importances_})
        return importances.sort_values(by='Importance', ascending=False)



    def calculate_importances_kfolds(self, n_estimators=100, random_state=0, n_folds=10):
        # Count the number of -1 labels
        num_negative_ones = sum(self.cluster_labels == -1)
        logging.info(f"Number of '-1' labels being removed: {num_negative_ones}")
    
        # Filter out rows where cluster_labels are -1
        valid_indices = self.cluster_labels != -1
        y = self.cluster_labels[valid_indices]
        X = self.feature_vectors.loc[valid_indices]
    
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
        # Initialize array to store feature importances across folds
        feature_importances = np.zeros((n_folds, X.shape[1]))
    
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
    
            xgb = XGBClassifier(random_state=random_state, n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X_train, y_train)
    
            # Store feature importances for this fold
            feature_importances[fold] = xgb.feature_importances_
    
        # Average feature importances across folds
        avg_feature_importances = np.mean(feature_importances, axis=0)
        importances = pd.DataFrame({'Feature': X.columns, 'Importance': avg_feature_importances})
        return importances.sort_values(by='Importance', ascending=False)

    def plot_importances(self, importances, max_features=15, prefix=""):
        """
        Plot the feature importances.

        Parameters:
        importances (DataFrame): DataFrame containing feature importances.
        max_features (int): The maximum number of top features to display.
        """
        fig = plt.figure(figsize=(10, 20))
        sns.barplot(x='Importance', y='Feature', data=importances[:max_features])
        plt.title('Feature Importances')
        plt.xlabel('Relative Importance')
        fig.tight_layout()
        plt.show()
        plt.savefig(f"{prefix}_feature_importance.png")

    def calculate_importances_per_cluster(self, method="cross-validation", n_repeats_kfolds=10, n_estimators=100, random_state=0, max_features=10, prefix=""):
        print("********** INSIDE WRAPPER **********")
        if method == "cross-validation":
            print("********** INSIDE CROSS-VALIDATION **********")
            return self.calculate_importances_per_cluster_kfold(n_folds=n_repeats_kfolds, n_estimators=n_estimators, random_state=0, max_features=max_features, prefix=prefix)

        print("********** INSIDE PERMUTATION **********")
        return self.calculate_importances_per_cluster_permutation(n_repeats=n_repeats_kfolds, n_estimators=n_estimators, random_state=0, max_features=max_features, prefix=prefix)

    def calculate_importances_per_cluster_permutation(self, n_repeats=10, n_estimators=100, random_state=0, max_features=10, prefix=""):
        """
        Calculate and plot feature importances per cluster using permutation feature importance with XGBoost.
    
        Parameters:
        n_repeats (int): Number of times to repeat the permutation.
        n_estimators (int): The number of boosting rounds.
        random_state (int): Random state for reproducibility.
        max_features (int): The maximum number of top features to display in the plot.
        prefix (str): Prefix for the plot file name.
        """
        feature_importances = {}
        n_clusters = len(np.unique(self.cluster_labels))
    
        # Create a figure with subplots - one for each cluster
        fig, axes = plt.subplots(1, n_clusters, figsize=(3*n_clusters, 6))  # Adjust figsize as needed
    
        for cluster in np.unique(self.cluster_labels):
            X = self.feature_vectors
            y = (self.cluster_labels == cluster).astype(int)
    
            xgb = XGBClassifier(n_estimators=n_estimators, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
            xgb.fit(X, y)
    
            # Perform permutation importance
            result = permutation_importance(xgb, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=5)
    
            importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
            sorted_importances = importances_df.sort_values(by='Importance', ascending=False).head(max_features)
            feature_importances[cluster] = sorted_importances
    
            # Plotting
            sns.barplot(ax=axes[cluster], y=sorted_importances['Importance'], x=sorted_importances['Feature'],
                        color='skyblue', edgecolor='black')
            axes[cluster].set_title(f'Cluster {cluster}')
            axes[cluster].set_xlabel('Importance')
            axes[cluster].tick_params(axis='x', rotation=90)
    
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{prefix}_permutation_feature_importance_per_cluster.png")
        #print(feature_importances)
        return feature_importances

    def calculate_importances_per_cluster_kfold(self, n_folds=10, n_estimators=100, random_state=0, max_features=15, prefix=""):
        """
        Calculate and plot feature importances per cluster using cross-validation with XGBoost.

        Parameters:
        n_folds (int): Number of folds for cross-validation.
        n_estimators (int): The number of boosting rounds.
        random_state (int): Random state for reproducibility.
        max_features (int): The maximum number of top features to display in the plot.
        prefix (str): Prefix for the plot file name.
        """
        feature_importances = {}
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        n_clusters = len(np.unique(self.cluster_labels))

        # Create a figure with subplots - one for each cluster
        fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 6))  # Adjust figsize as needed

        for cluster in np.unique(self.cluster_labels):
            print(f"******{cluster}******")
            cluster_feature_importances = []

            for train_index, test_index in kf.split(self.feature_vectors):
                X_train, X_test = self.feature_vectors.iloc[train_index], self.feature_vectors.iloc[test_index]
                y_train = (self.cluster_labels[train_index] == cluster).astype(int)

                xgb = XGBClassifier(n_estimators=n_estimators, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
                xgb.fit(X_train, y_train)
                cluster_feature_importances.append(xgb.feature_importances_)

            avg_importances = np.mean(cluster_feature_importances, axis=0)
            importances_df = pd.DataFrame({'Feature': self.feature_vectors.columns, 'Importance': avg_importances})
            sorted_importances = importances_df.sort_values(by='Importance', ascending=False).head(max_features)
            feature_importances[cluster] = sorted_importances

            # Plotting
            sns.barplot(ax=axes[cluster], y=sorted_importances['Importance'], x=sorted_importances['Feature'],
                        color='skyblue', edgecolor='black')
            axes[cluster].set_title(f'Cluster {cluster}')
            axes[cluster].set_xlabel('Importance')
            axes[cluster].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        plt.show()
        plt.savefig(f"{prefix}_feature_importance_per_cluster.png")
#        print(feature_importances)
        return feature_importances


class FeatureImportance2:
    def __init__(self, feature_vectors, cluster_labels):
        self.feature_vectors = feature_vectors
        self.cluster_labels = cluster_labels

    def calculate_importances(self, n_estimators=100, random_state=0):
        """
        Calculate feature importances using a random forest classifier.

        Parameters:
        n_estimators (int): The number of trees in the forest.
        random_state (int): Random state for reproducibility.

        Returns:
        DataFrame: A DataFrame containing feature importances.
        """
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rf.fit(self.feature_vectors, self.cluster_labels)
        importances = pd.DataFrame({'Feature': self.feature_vectors.columns, 
                                    'Importance': rf.feature_importances_})
        return importances.sort_values(by='Importance', ascending=False)

    def calculate_importances_kfolds(self, n_estimators=100, random_state=0, n_folds=10):

        y = self.cluster_labels
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

		# Initialize array to store feature importances across folds
        feature_importances = np.zeros((n_folds, len(self.feature_vectors.columns)))

        for fold, (train_index, test_index) in enumerate(kf.split(self.feature_vectors)):
            X_train, X_test = self.feature_vectors.iloc[train_index], self.feature_vectors.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            rf = RandomForestClassifier(random_state=0, n_estimators=100)
            rf.fit(X_train, y_train)

            # Store feature importances for this fold
            feature_importances[fold] = rf.feature_importances_

		# Average feature importances across folds
        avg_feature_importances = np.mean(feature_importances, axis=0)
        importances = pd.DataFrame({'Feature': self.feature_vectors.columns, 'Importance': avg_feature_importances})
        return importances.sort_values(by='Importance', ascending=False)


    def plot_importances(self, importances, max_features=10, prefix=""):
        """
        Plot the feature importances.

        Parameters:
        importances (DataFrame): DataFrame containing feature importances.
        max_features (int): The maximum number of top features to display.
        """
        fig = plt.figure(figsize=(10, 20))
        sns.barplot(x='Importance', y='Feature', data=importances[:max_features])
        plt.title('Feature Importances')
        plt.xlabel('Relative Importance')
        fig.tight_layout()
        plt.show()
        plt.savefig(f"{prefix}_feature_importance.png")

    def calculate_importances_per_cluster(self, n_folds=10, n_estimators=100, random_state=0):
        """
        Calculate feature importances per cluster using cross-validation.

        Parameters:
        n_folds (int): Number of folds for cross-validation.
        n_estimators (int): The number of trees in the forest.
        random_state (int): Random state for reproducibility.

        Returns:
        dict: A dictionary where keys are cluster labels and values are DataFrames of feature importances.
        """
        feature_importances = {}
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for cluster in np.unique(self.cluster_labels):
            cluster_feature_importances = []

            for train_index, test_index in kf.split(self.feature_vectors):
                X_train, X_test = self.feature_vectors.iloc[train_index], self.feature_vectors.iloc[test_index]
                y_train = (self.cluster_labels[train_index] == cluster).astype(int)

                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                rf.fit(X_train, y_train)
                cluster_feature_importances.append(rf.feature_importances_)

            avg_importances = np.mean(cluster_feature_importances, axis=0)
            importances_df = pd.DataFrame({'Feature': self.feature_vectors.columns, 'Importance': avg_importances})
            feature_importances[cluster] = importances_df.sort_values(by='Importance', ascending=False)

        return feature_importances

    def plot_feature_importances(self, prefix=""):
        n_clusters = len(np.unique(self.cluster_labels))
        n_folds = 10  # Number of folds for cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    
        # Create a figure with subplots - one for each cluster
        fig, axes = plt.subplots(1, n_clusters, figsize=(25, 6))  # Adjust figsize as needed
    
        for cluster in np.unique(self.cluster_labels):
            y_cluster = (self.cluster_labels == cluster).astype(int)
            
            # Initialize array to store feature importances across folds
            feature_importances = np.zeros((n_folds, len(self.feature_vectors.columns)))
    
            for fold, (train_index, test_index) in enumerate(kf.split(self.feature_vectors)):
                X_train, X_test = self.feature_vectors.iloc[train_index], self.feature_vectors.iloc[test_index]
                y_train, y_test = y_cluster[train_index], y_cluster[test_index]
    
                rf_cluster = RandomForestClassifier(random_state=0, n_estimators=100)
                rf_cluster.fit(X_train, y_train)
    
                # Store feature importances for this fold
                feature_importances[fold] = rf_cluster.feature_importances_
    
            # Average feature importances across folds
            avg_feature_importances = np.mean(feature_importances, axis=0)
            feature_importances_cluster = pd.Series(avg_feature_importances, index=self.feature_vectors.columns)
            sorted_importances_cluster = feature_importances_cluster.sort_values(ascending=False).head(15)
    
            sns.barplot(ax=axes[cluster], y=sorted_importances_cluster.values, x=sorted_importances_cluster.index, 
                        color='skyblue', edgecolor='black')
            axes[cluster].set_title(f'Cluster {cluster}')
            axes[cluster].set_xlabel('Importance')
            axes[cluster].tick_params(axis='x', rotation=90)
    
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{prefix}_feature_importance.png")
