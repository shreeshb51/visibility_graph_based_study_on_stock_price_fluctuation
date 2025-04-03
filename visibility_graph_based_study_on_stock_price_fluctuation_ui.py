import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import seaborn as sns
import streamlit as st

class StockVisibilityAnalyzer:
    """
    A class to analyze stock fluctuations using visibility graphs.
    
    Attributes:
        clean_data (pd.DataFrame): Cleaned and preprocessed stock data.
        window_size (int): Size of the sliding window for feature extraction.
        features_data (pd.DataFrame): Extracted features using sliding windows.
        visibility_graphs (list): List of visibility graphs generated from the data.
        graph_dataset (list): List of tuples containing graphs and their corresponding labels.
        labeled_data (pd.DataFrame): Data labeled with trends (Increasing, Decreasing, No Change).
        column_prefix (str): Prefix for column names in the feature dataset.
    """

    def __init__(self):
        """Initialize the StockVisibilityAnalyzer with default attributes."""
        self.clean_data = None
        self.window_size = None
        self.features_data = None
        self.visibility_graphs = None
        self.graph_dataset = None
        self.labeled_data = None
        self.column_prefix = None

    def load_and_clean_data(self, file_path):
        """
        Load and clean the stock data from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing stock data.

        Returns:
            pd.DataFrame: Cleaned and preprocessed stock data.
        """
        data = pd.read_csv(file_path)
        available_symbols = data['symbol'].unique()

        # Let the user select a company stock
        user_choice = st.sidebar.selectbox("Select the company stock you would like to check:", available_symbols)

        # Filter data for the selected stock
        filtered_data = data[data['symbol'] == user_choice].drop_duplicates(subset='f_date')
        filtered_data['f_date'] = pd.to_datetime(filtered_data['f_date'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['f_date'])

        # Fix missing dates by setting date as index and forward-filling
        filtered_data = filtered_data.set_index('f_date')
        all_dates = pd.date_range(start=filtered_data.index.min(), end=filtered_data.index.max(), freq='D')
        filtered_data = filtered_data.reindex(all_dates).ffill()

        # Remove unnecessary columns and calculate average price
        cols_to_drop = ["Unnamed: 0", "sector", "percent_change"]
        filtered_data = filtered_data.drop([col for col in cols_to_drop if col in filtered_data.columns], axis=1)
        filtered_data['avg'] = filtered_data[['open', 'high', 'low', 'close']].mean(axis=1)

        # Reset index for further processing
        filtered_data.reset_index(inplace=True)
        filtered_data.rename(columns={'index': 'f_date'}, inplace=True)

        self.clean_data = filtered_data
        return self.clean_data

    def create_sliding_window_features(self, window_size):
        """
        Create sliding window features from the cleaned stock data.

        Args:
            window_size (int): Size of the sliding window.

        Returns:
            pd.DataFrame: DataFrame containing sliding window features.
        """
        self.window_size = window_size

        # Extract columns to process
        columns = ['avg', 'volume', 'open', 'high', 'low', 'close']
        rows_count = len(self.clean_data) - window_size + 1

        # Pre-allocate numpy array for better performance
        feature_array = np.zeros((rows_count, len(columns) * window_size))

        # Extract data once for all columns
        column_data = {col: self.clean_data[col].values for col in columns}

        # Fill the feature array efficiently
        for i in range(rows_count):
            col_idx = 0
            for col in columns:
                feature_array[i, col_idx:col_idx + window_size] = column_data[col][i:i+window_size]
                col_idx += window_size

        # Create column names
        column_names = []
        for col in columns:
            column_names.extend([f'{col}_{i+1}' for i in range(window_size)])

        self.features_data = pd.DataFrame(feature_array, columns=column_names)
        return self.features_data

    def generate_visibility_graphs(self, column_prefix, batch_size=100):
        """
        Generate visibility graphs from the sliding window features.

        Args:
            column_prefix (str): Prefix for column names in the feature dataset.
            batch_size (int): Number of graphs to process in each batch.

        Returns:
            list: List of visibility graphs.
        """
        self.column_prefix = column_prefix
        column_names = [f'{column_prefix}_{i+1}' for i in range(self.window_size)]

        # Process in batches to reduce memory pressure
        self.visibility_graphs = []
        num_batches = (len(self.features_data) + batch_size - 1) // batch_size

        progress_bar = st.progress(0)
        status_text = st.empty()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.features_data))

            batch_values = self.features_data.iloc[start_idx:end_idx][column_names].values
            batch_graphs = [self._create_visibility_graph(row) for row in batch_values]
            self.visibility_graphs.extend(batch_graphs)

            progress_bar.progress((batch_idx + 1) / num_batches)
            status_text.text(f"Generating visibility graphs... {batch_idx + 1}/{num_batches} batches completed")

        return self.visibility_graphs

    def _create_visibility_graph(self, values):
        """
        Helper method to create a visibility graph from a row of values.

        Args:
            values (np.array): Array of values to create the visibility graph.

        Returns:
            nx.Graph: Visibility graph.
        """
        try:
            return nx.visibility_graph(values)
        except Exception as e:
            st.error(f"Error creating visibility graph: {e}")
            # Return empty graph as fallback
            G = nx.Graph()
            for i in range(len(values)):
                G.add_node(i)
            return G

    def label_dataset(self):
        """
        Label the dataset with trends (Increasing, Decreasing, No Change).

        Returns:
            pd.DataFrame: Labeled dataset.
        """
        column_names = [f'{self.column_prefix}_{i+1}' for i in range(self.window_size)]
        self.labeled_data = self.features_data[column_names].copy()

        # Get last column values for comparison
        last_col = f'{self.column_prefix}_{self.window_size}'
        last_col_values = self.labeled_data[last_col].values

        # Pre-allocate class array - reversed decreasing/increasing logic to fix the error
        class_array = np.zeros(len(self.labeled_data), dtype=int)

        # Vectorized operations for trend classification
        for i in range(len(self.labeled_data) - 1):
            current_val = last_col_values[i]
            next_val = last_col_values[i + 1]

            if next_val > current_val:
                class_array[i] = 2  # Increasing
            elif next_val == current_val:
                class_array[i] = 1  # No Change
            else:
                class_array[i] = 0  # Decreasing

        # Convert numerical classes to labels
        self.labeled_data['class'] = class_array
        self.labeled_data['class'] = self.labeled_data['class'].map(
            {0: "Decreasing", 1: "No Change", 2: "Increasing"}
        )

        # Associate graphs with classes
        self.graph_dataset = list(zip(self.visibility_graphs, self.labeled_data['class']))

        return self.labeled_data

    def save_graph_dataset(self, filename='graph_dataset_class_labeled.pkl'):
        """
        Save the graph dataset to a pickle file.

        Args:
            filename (str): Name of the file to save the dataset.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.graph_dataset, f)
        st.success(f"Graph dataset saved to {filename}")

    def extract_graph_features(self):
        """
        Extract features from the visibility graphs.

        Returns:
            pd.DataFrame: DataFrame containing graph features and their corresponding labels.
        """
        features = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (graph, _) in enumerate(self.graph_dataset):
            # Initialize feature dictionary with default values
            feature_dict = {
                '#nodes': graph.number_of_nodes(),
                '#edges': graph.number_of_edges(),
                'diameter': 0,
                'radius': 0,
                'centers': 0,
                'avg_degree': 0,
                'degree_variance': 0,
                'avg_shortest_path': 0,
                'edge_connectivity': 0,
                'eigenvector_centrality': 0,
                'clustering_coeff': 0
            }

            try:
                if feature_dict['#nodes'] > 0 and feature_dict['#edges'] > 0:
                    # Calculate degrees for all nodes
                    degrees = [deg for _, deg in graph.degree()]
                    if degrees:
                        feature_dict['avg_degree'] = np.mean(degrees)
                        feature_dict['degree_variance'] = np.var(degrees)

                    # Calculate graph metrics
                    feature_dict['clustering_coeff'] = nx.average_clustering(graph)

                    # Metrics that require connected graphs
                    if nx.is_connected(graph):
                        feature_dict['diameter'] = nx.diameter(graph)
                        feature_dict['radius'] = nx.radius(graph)
                        feature_dict['centers'] = len(nx.center(graph))
                        feature_dict['avg_shortest_path'] = nx.average_shortest_path_length(graph)
                        feature_dict['edge_connectivity'] = nx.edge_connectivity(graph)

                    # Calculate eigenvector centrality with error handling
                    try:
                        centrality_dict = nx.eigenvector_centrality(graph, max_iter=5000, tol=1e-4)
                        feature_dict['eigenvector_centrality'] = np.mean(list(centrality_dict.values()))
                    except (nx.PowerIterationFailedConvergence, ValueError):
                        pass  # Keep the default value

            except (nx.NetworkXError, nx.NetworkXPointlessConcept):
                pass  # Keep the default values

            features.append(feature_dict)

            # Update progress
            progress_bar.progress((idx + 1) / len(self.graph_dataset))
            status_text.text(f"Extracting graph features... {idx + 1}/{len(self.graph_dataset)} graphs processed")

        # Create DataFrame and add class labels
        features_df = pd.DataFrame(features)
        features_df['class'] = [class_label for _, class_label in self.graph_dataset]

        return features_df

class Visualizer:
    """
    A class to visualize visibility graphs and feature distributions.
    """

    @staticmethod
    def plot_visibility_graphs(visibility_graphs, num_samples, node_size, font_size, node_color, edge_color, font_color, layout):
        """
        Plot a sample of visibility graphs.

        Args:
            visibility_graphs (list): List of visibility graphs.
            num_samples (int): Number of graphs to plot.
            node_size (int): Size of nodes in the graph.
            font_size (int): Font size for node labels.
            node_color (str): Color of nodes.
            edge_color (str): Color of edges.
            font_color (str): Color of node labels.
            layout (str): Layout type for the graph (circular, kamada_kawai, spring).
        """
        sampled_graphs = visibility_graphs[:num_samples]

        for i, G in enumerate(sampled_graphs):
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.margins(0.20)

            # Choose layout based on user selection
            if layout == "circular":
                pos = nx.circular_layout(G)
            elif layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            else:  # Default to spring layout
                pos = nx.spring_layout(G)

            nx.draw_networkx_nodes(G, pos, alpha=0.3, node_size=node_size, node_color=node_color)
            nx.draw_networkx_labels(G, pos, font_size=font_size, font_color=font_color, font_weight='normal')
            nx.draw_networkx_edges(G, pos, alpha=0.6, width=1, edge_color=edge_color, style='solid',
                                  arrows=True, arrowstyle='-')

            plt.title(f"Visibility Graph {i+1}", fontsize=12)
            plt.axis("equal")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    @staticmethod
    def plot_feature_distributions(features_df):
        """
        Plot feature distributions for each class.

        Args:
            features_df (pd.DataFrame): DataFrame containing graph features and their corresponding labels.
        """
        # Define features to plot, ensuring they match the column names in features_df
        graph_features = [
            '#edges', 'radius', 'diameter', 'centers', 'avg_degree', 'degree_variance',
            'avg_shortest_path', 'edge_connectivity', 'eigenvector_centrality', 'clustering_coeff'
        ]

        class_colors = {
            "Decreasing": 'blue',
            "No Change": 'green',
            "Increasing": 'red'
        }

        for feature in graph_features:
            # Skip features not in the DataFrame
            if feature not in features_df.columns:
                st.warning(f"Feature '{feature}' not found in the data.")
                continue

            fig, ax = plt.subplots(figsize=(10, 10))

            for class_label, color in class_colors.items():
                class_data = features_df[features_df['class'] == class_label]
                if len(class_data) > 0:
                    sns.kdeplot(class_data[feature],
                               label=f'Class {class_label}',
                               fill=False,
                               color=color,
                               alpha=0.9)

            if len(plt.gca().get_legend_handles_labels()[0]) > 0:
                plt.legend()

            plt.title(f'Distribution of {feature.upper()} by Class')
            plt.xlabel(feature.lower())
            plt.ylabel('Density')
            plt.grid(True)
            st.pyplot(fig)
            plt.close()

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Study of Stocks Fluctuation via Visibility Graph", layout="wide")
    st.title("📈 Study of Stocks Fluctuation via Visibility Graph")

    analyzer = StockVisibilityAnalyzer()

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        analyzer.load_and_clean_data(uploaded_file)

        data_length = len(analyzer.clean_data)
        window_size = st.sidebar.number_input(
            "Enter the days for moving average:",
            min_value=2, max_value=data_length, value=min(30, data_length),
            help=f"Window size must be between 2 and {data_length} days."
        )

        field_options = ['close', 'open', 'high', 'avg', 'volume', 'low']
        column_prefix = st.sidebar.selectbox("Select the field to explore:", field_options)

        # Calculate maximum number of visibility graphs
        max_visibility_graphs = data_length - window_size + 1
        num_samples = st.sidebar.number_input(
            "Enter the number of visibility graph samples:",
            min_value=1, max_value=max_visibility_graphs, value=min(5, max_visibility_graphs),
            help=f"Number of samples must be between 1 and {max_visibility_graphs}."
        )

        # Graph customization options
        st.sidebar.write("Customize Graph Appearance:")
        layout = st.sidebar.selectbox("Graph Layout", ["circular", "kamada_kawai", "spring"])
        node_size = st.sidebar.slider("Node size", min_value=10, max_value=500, value=300)
        font_size = st.sidebar.slider("Font size", min_value=5, max_value=20, value=10)
        node_color = st.sidebar.color_picker("Node color", "#00ff00")
        edge_color = st.sidebar.color_picker("Edge color", "#0000ff")
        font_color = st.sidebar.color_picker("Font color", "#000000")

        if st.sidebar.button("Study"):
            with st.spinner("Processing data..."):
                analyzer.create_sliding_window_features(window_size)

                st.write("Generating visibility graphs...")
                analyzer.generate_visibility_graphs(column_prefix)

                st.write("Labeling dataset...")
                analyzer.label_dataset()
                analyzer.save_graph_dataset()

                st.write("Displaying sample visibility graphs...")
                Visualizer.plot_visibility_graphs(
                    analyzer.visibility_graphs, num_samples,
                    node_size, font_size, node_color, edge_color, font_color, layout
                )

                st.write("Analyzing feature distributions...")
                features_df = analyzer.extract_graph_features()
                Visualizer.plot_feature_distributions(features_df)

        if st.sidebar.button("Reset"):
            st.rerun()

if __name__ == "__main__":
    main()
