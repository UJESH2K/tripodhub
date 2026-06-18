"""Notebook-friendly code templates for tripodhub labs."""

from __future__ import annotations

from textwrap import dedent


LAB_CODES: dict[int, str] = {
    1: dedent(
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.datasets import fetch_california_housing

        dataset = fetch_california_housing(as_frame=True)
        data = dataset.frame.copy()
        data["MedHouseVal"] = dataset.target

        data.hist(bins=30, figsize=(15, 10))
        plt.suptitle("Histograms of California Housing Features")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 10))
        for index, column in enumerate(data.columns, start=1):
            plt.subplot(3, 3, index)
            sns.boxplot(y=data[column])
            plt.title(column)
        plt.suptitle("Box Plots of California Housing Features")
        plt.tight_layout()
        plt.show()

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).sum()
        print(outliers)
        """
    ).strip(),
    2: dedent(
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.datasets import fetch_california_housing

        dataset = fetch_california_housing(as_frame=True)
        data = dataset.frame.copy()
        data["MedHouseVal"] = dataset.target

        correlation_matrix = data.corr(numeric_only=True)
        print(correlation_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("California Housing Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        sns.pairplot(data.sample(min(500, len(data)), random_state=42))
        plt.show()
        """
    ).strip(),
    3: dedent(
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.datasets import load_iris
        from sklearn.decomposition import PCA

        iris = load_iris(as_frame=True)
        features = iris.data

        pca = PCA(n_components=2, random_state=42)
        transformed = pca.fit_transform(features)

        result = pd.DataFrame(transformed, columns=["PC1", "PC2"])
        result["target"] = iris.target
        print(result.head())

        scatter = plt.scatter(result["PC1"], result["PC2"], c=result["target"], cmap="viridis")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA of Iris Dataset")
        plt.legend(*scatter.legend_elements(), title="Class")
        plt.show()
        """
    ).strip(),
    4: dedent(
        """
        import pandas as pd

        data = pd.DataFrame(
            [
                ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
                ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
                ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
                ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"],
            ],
            columns=["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast", "EnjoySport"],
        )

        positive_examples = data[data["EnjoySport"].str.lower() == "yes"]
        hypothesis = positive_examples.iloc[0, :-1].tolist()

        for _, row in positive_examples.iloc[1:].iterrows():
            for index, value in enumerate(row[:-1]):
                if hypothesis[index] != value:
                    hypothesis[index] = "?"

        final_hypothesis = dict(zip(data.columns[:-1], hypothesis))
        print(final_hypothesis)
        """
    ).strip(),
    5: dedent(
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import accuracy_score
        from sklearn.neighbors import KNeighborsClassifier

        rng = np.random.default_rng(42)
        x_values = np.sort(rng.random(100))
        y_labels = np.where(x_values <= 0.5, 0, 1)

        x_train = x_values[:50].reshape(-1, 1)
        y_train = y_labels[:50]
        x_test = x_values[50:].reshape(-1, 1)
        y_test = y_labels[50:]

        k_values = [1, 2, 3, 4, 5, 20, 30]
        plt.figure(figsize=(14, 10))

        for index, k in enumerate(k_values, start=1):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            print(f"k={k}, accuracy={accuracy_score(y_test, predictions):.2f}")

            plt.subplot(3, 3, index)
            plt.scatter(x_test[:, 0], y_test, label="True", alpha=0.7)
            plt.scatter(x_test[:, 0], predictions, label="Predicted", marker="x")
            plt.title(f"KNN (k={k})")
            plt.xlabel("x")
            plt.ylabel("class")
            plt.ylim(-0.2, 1.2)

        plt.tight_layout()
        plt.show()
        """
    ).strip(),
    6: dedent(
        """
        import matplotlib.pyplot as plt
        import numpy as np

        def locally_weighted_regression(x_train, y_train, query_points, tau=0.5):
            x_train_bias = np.c_[np.ones(len(x_train)), x_train]
            predictions = []

            for point in query_points:
                weights = np.exp(-np.sum((x_train - point) ** 2, axis=1) / (2 * tau**2))
                weighted_x = x_train_bias * weights[:, np.newaxis]
                weighted_y = y_train * weights
                theta, _, _, _ = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)
                point_bias = np.r_[1.0, point]
                predictions.append(point_bias @ theta)

            return np.array(predictions)

        rng = np.random.default_rng(42)
        x_train = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_train = np.sin(x_train[:, 0]) + rng.normal(0, 0.2, size=len(x_train))

        x_query = np.linspace(x_train.min(), x_train.max(), 200).reshape(-1, 1)
        y_pred = locally_weighted_regression(x_train, y_train, x_query, tau=0.5)

        plt.scatter(x_train[:, 0], y_train, label="Training Data", alpha=0.7)
        plt.plot(x_query[:, 0], y_pred, color="red", label="LWR Fit")
        plt.title("Locally Weighted Regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
        """
    ).strip(),
    7: dedent(
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.datasets import fetch_california_housing, fetch_openml
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures

        housing = fetch_california_housing(as_frame=True)
        x_linear = housing.data
        y_linear = housing.target

        x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(
            x_linear, y_linear, test_size=0.2, random_state=42
        )
        linear_model = LinearRegression()
        linear_model.fit(x_train_l, y_train_l)
        linear_predictions = linear_model.predict(x_test_l)
        print("Linear R2:", r2_score(y_test_l, linear_predictions))
        print("Linear RMSE:", mean_squared_error(y_test_l, linear_predictions) ** 0.5)

        try:
            mpg = sns.load_dataset("mpg").dropna()
            x_poly = mpg[["horsepower"]].astype(float)
            y_poly = mpg["mpg"].astype(float)
        except Exception:
            mpg = fetch_openml(name="autoMpg", version=1, as_frame=True, parser="auto").frame.dropna()
            x_poly = mpg[["horsepower"]].astype(float)
            y_poly = mpg["mpg"].astype(float)

        x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(
            x_poly, y_poly, test_size=0.2, random_state=42
        )
        polynomial_model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", LinearRegression()),
            ]
        )
        polynomial_model.fit(x_train_p, y_train_p)
        polynomial_predictions = polynomial_model.predict(x_test_p)
        print("Polynomial R2:", r2_score(y_test_p, polynomial_predictions))
        print("Polynomial RMSE:", mean_squared_error(y_test_p, polynomial_predictions) ** 0.5)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_test_l, linear_predictions, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Linear Regression")

        plt.subplot(1, 2, 2)
        sort_index = np.argsort(x_test_p.iloc[:, 0].to_numpy())
        plt.scatter(x_test_p.iloc[:, 0], y_test_p, alpha=0.6, label="Actual")
        plt.plot(
            x_test_p.iloc[:, 0].to_numpy()[sort_index],
            polynomial_predictions[sort_index],
            color="red",
            label="Polynomial Fit",
        )
        plt.xlabel("Horsepower")
        plt.ylabel("MPG")
        plt.title("Polynomial Regression")
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    ).strip(),
    8: dedent(
        """
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        dataset = load_breast_cancer()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
        )

        model = DecisionTreeClassifier(random_state=42)
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        print("Accuracy:", accuracy_score(y_test, predictions))

        new_sample = x_test[0].reshape(1, -1)
        new_prediction = model.predict(new_sample)[0]
        print("Predicted class:", dataset.target_names[int(new_prediction)])
        """
    ).strip(),
    9: dedent(
        """
        from sklearn.datasets import fetch_olivetti_faces
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB

        dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.25, random_state=42, stratify=dataset.target
        )

        model = GaussianNB()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print("Accuracy:", accuracy_score(y_test, predictions))
        """
    ).strip(),
    10: dedent(
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.datasets import load_breast_cancer
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score

        dataset = load_breast_cancer(as_frame=True)
        features = dataset.data

        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = model.fit_predict(features)
        print("Silhouette score:", silhouette_score(features, clusters))

        reduced = PCA(n_components=2, random_state=42).fit_transform(features)
        result = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        result["cluster"] = clusters

        plt.scatter(result["PC1"], result["PC2"], c=result["cluster"], cmap="viridis")
        plt.title("K-Means Clustering on Breast Cancer Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()
        """
    ).strip(),
}


def get_code(lab_number: int) -> str:
    """Return a copy-pasteable notebook code snippet for a lab."""
    if lab_number not in LAB_CODES:
        raise ValueError("lab_number must be between 1 and 10.")
    return LAB_CODES[lab_number]
