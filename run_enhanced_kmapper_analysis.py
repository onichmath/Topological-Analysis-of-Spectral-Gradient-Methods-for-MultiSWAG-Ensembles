import numpy as np
import torch
import networkx as nx
import kmapper as km
from pathlib import Path
import time
import gc
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import ripser
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy.spatial.distance import pdist, squareform, cdist
import pandas as pd
import pickle
from kmapper.plotlyviz import plotlyviz
from umap import UMAP

warnings.filterwarnings('ignore')
np.random.seed(42)
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1'
})

class DataLoader:
    def __init__(self, max_trajectories=620, max_total=30000):
        self.max_trajectories = max_trajectories
        self.max_total = max_total
    
    def _load_single_file(self, meta):
        try:
            file_path = meta['file']
            if not os.path.exists(file_path):
                print(f"  File not found: {file_path}")
                return None, None
            
            with open(file_path, 'rb') as f:
                state_dict = torch.load(file_path, map_location='cpu', mmap=True)
            
            weight_vector = []
            for key, param in state_dict.items():
                if isinstance(param, torch.Tensor):
                    weight_vector.extend(param.flatten().detach().cpu().numpy())
            
            del state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not weight_vector:
                print(f"  No valid parameters in: {file_path}")
                return None, None
            
            return np.array(weight_vector, dtype=np.float32), meta
        except Exception as e:
            print(f"  Error loading {meta['file']}: {str(e)}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return None, None
    
    def load_optimizer_data(self, optimizer, epochs=range(0, 31), max_particles=20):
        print(f"🔄 Loading trajectories for {optimizer}...")
        
        weights_dir = Path("results") / optimizer / "pretrain_weights"
        eval_dir = Path("results") / optimizer / "evaluation_results"
        
        if not weights_dir.exists():
            print(f"  ❌ No weights directory for {optimizer}")
            return None
        
        # Load evaluation data
        eval_data = {}
        if eval_dir.exists():
            for eval_file in eval_dir.glob("*_metrics.pt"):
                try:
                    data = torch.load(eval_file, map_location='cpu')
                    epoch = int(eval_file.stem.split('epoch')[1].split('_')[0])
                    eval_data[epoch] = data
                except (ValueError, IndexError):
                    continue
        
        optimizer_files = []
        for epoch in epochs:
            for particle_id in range(max_particles):
                weight_file = weights_dir / f"particle{particle_id}_epoch{epoch}_weights.pt"
                if weight_file.exists():
                    loss = None
                    if epoch in eval_data:
                        for key in ['val_loss', 'validation_loss', 'train_loss']:
                            if key in eval_data[epoch]:
                                loss = float(eval_data[epoch][key])
                                break
                    
                    optimizer_files.append({
                        'file': str(weight_file),
                        'optimizer': optimizer,
                        'epoch': epoch,
                        'particle_id': particle_id,
                        'val_loss': loss
                    })
        
        if len(optimizer_files) > self.max_trajectories:
            step = len(optimizer_files) // self.max_trajectories
            optimizer_files = optimizer_files[::step][:self.max_trajectories]
        
        print(f"  ✅ Found {len(optimizer_files)} files")
        return optimizer_files
    
    def load_trajectories(self, metadata_list):
        print("📦 Loading weight trajectories...")
        
        # Process files sequentially to debug issues
        trajectories, valid_metadata = [], []
        
        for i, meta in enumerate(metadata_list):
            #print(f"\nProcessing file {i+1}/{len(metadata_list)}: {meta['file']}")
            traj, valid_meta = self._load_single_file(meta)
            if traj is not None:
                trajectories.append(traj)
                valid_metadata.append(valid_meta)
                #print(f"  Successfully loaded trajectory with {len(traj)} parameters")
            else:
                print(f"  Failed to load trajectory")
            
            if (i + 1) % 100 == 0:
                print(f"  Loaded {len(trajectories)}/{len(metadata_list)} trajectories")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"  ✅ Loaded {len(trajectories)} trajectories")
        return np.array(trajectories), valid_metadata


class ProjectionEngine:
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.variance_threshold = VarianceThreshold(threshold=1e-6)  # Drop near-zero weights
        self.pca = PCA(n_components=n_components, random_state=random_state)
    
    def preprocess_weights(self, weights):
        weights_log = np.sign(weights) * np.log1p(np.abs(weights))
        weights_var = self.variance_threshold.fit_transform(weights_log)
        weights_scaled = self.scaler.fit_transform(weights_var)
        
        return weights_scaled
    
    def create_projections(self, trajectories):
        print("🎯 Creating projections...")
        
        # Preprocess weights
        trajectories_processed = self.preprocess_weights(trajectories)
        # Apply PCA for dimensionality reduction (use first 50 PCs)
        pca_result = self.pca.fit_transform(trajectories_processed)
        
        # Use PCA result directly instead of UMAP
        print(f"  Using first 50 PCA components: {pca_result.shape}")
        
        # Return PCA results (no UMAP)
        return {
            'pca': pca_result,
            'pca_50': pca_result  # Use PCA as the main embedding
        }, self.pca
    
    @staticmethod
    def create_pc1_pc2_lens(projections, metadata):
        print("🔍 Creating PC1+PC2 lens function...")
        
        if 'pca' not in projections:
            raise ValueError("PCA projections required for lens function")
        
        pca_result = projections['pca']
        
        # Extract PC1 and PC2
        pc1 = pca_result[:, 0]
        pc2 = pca_result[:, 1]
        
        # Normalize each dimension separately
        pc1_norm = (pc1 - np.mean(pc1)) / (np.std(pc1) + 1e-8)
        pc2_norm = (pc2 - np.mean(pc2)) / (np.std(pc2) + 1e-8)
        
        # Combine into 2D lens
        lens = np.column_stack([pc1_norm, pc2_norm])
        
        print(f"  ✅ Created 2D lens: PC1+PC2 shape {lens.shape}")
        return lens

    @staticmethod
    def create_umap_lens(projections, metadata):
        print("🔍 Creating UMAP lens function...")
        
        if 'pca' not in projections:
            raise ValueError("PCA projections required for UMAP lens function")
        
        # Extract PCA data from projections dictionary
        pca_data = projections['pca']
        
        # Apply UMAP to the PCA data
        umap_result = UMAP(n_components=2, random_state=42).fit_transform(pca_data)
        
        print(f"  ✅ Created 2D lens: UMAP shape {umap_result.shape}")
        return umap_result


class MapperEngine:
    def __init__(self, n_cubes=5, perc_overlap=0.5, n_clusters=3, random_state=42):
        # Note that we choose perc_overlap to be the smallest where we have 1 connected component
        self.n_cubes = n_cubes
        self.perc_overlap = perc_overlap
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.mapper = km.KeplerMapper(verbose=0)
        self.scaler = StandardScaler()
    
    def create_mapper(self, projections, lens=None):
        if lens is None:
            lens = projections
        
        # Normalize lens data
        lens_normalized = self.scaler.fit_transform(lens)
        
        # Create cover appropriate for lens dimensions
        lens_dim = lens_normalized.shape[1]
        
        # span = np.ptp(lens_normalized, axis=0)
        # cube_len = 0.2 * span
        # _cubes = np.ceil(span / cube_len).astype(int)
        # print(f"  Creating cover for {lens_dim}D lens with {_cubes} cubes per dimension")

        try:
            if lens_dim == 3:
                # For 3D lens, use fewer cubes per dimensions
                cover = km.Cover(n_cubes=self.n_cubes * 3, perc_overlap=self.perc_overlap)
            else:
                # For 2D or other dimensions
                cover = km.Cover(n_cubes=self.n_cubes, perc_overlap=self.perc_overlap)
        except Exception as e:
            print(f"  Cover creation error: {e}, using default cover")
            cover = km.Cover(n_cubes=self.n_cubes, perc_overlap=self.perc_overlap)
        
        # Try different clusterers and choose the best one
        clusterers_to_try = [
            ('KMeans', KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)),
            # ('Hierarchical', AgglomerativeClustering(n_clusters=self.n_clusters)),
            #('DBSCAN', clusterer)
        ]
        
        best_complex = None
        best_score = -1
        best_clusterer_name = None
        
        for clusterer_name, clusterer in clusterers_to_try:
            try:
                print(f"    Trying {clusterer_name} clustering...")
                complex_candidate = self.mapper.map(
                    lens_normalized,
                    X=projections,
                    cover=cover,
                    clusterer=clusterer
                )
                
                # Score based on barcode metrics and graph structure
                n_nodes = len(complex_candidate['nodes'])
                n_edges = len(complex_candidate['links'])
                
                if n_nodes > 0:
                    # Compute barcode metrics for this clustering
                    temp_metrics = self.compute_lens_metrics(lens_normalized, projections, complex_candidate)
                    
                    # Extract barcode features
                    h0_features = temp_metrics.get('h0_num_features', 0)
                    h1_features = temp_metrics.get('h1_num_features', 0)
                    h0_max_life = temp_metrics.get('h0_max_lifetime', 0.0)
                    h1_max_life = temp_metrics.get('h1_max_lifetime', 0.0)
                    persistence_mean = temp_metrics.get('persistence_mean', 0.0)
                    
                    # Composite score combining multiple factors
                    # 1. Graph structure (nodes and connectivity)
                    connectivity_score = n_edges / max(n_nodes, 1)
                    structure_score = n_nodes * (1 - abs(connectivity_score - 1.5) / 5)
                    
                    # 2. Topological richness (prefer some H1 features but not too many)
                    h1_score = min(h1_features, 5) * 2  # Reward 1-5 loops, diminishing returns after
                    
                    # 3. Persistence quality (longer-lived features are better)
                    persistence_score = persistence_mean * 10
                    
                    # 4. Balanced homology (prefer some H0 diversity but not fragmentation)
                    h0_score = max(0, 10 - abs(h0_features - 5))  # Optimal around 5 components
                    
                    # Combined score
                    score = structure_score + h1_score + persistence_score + h0_score
                    
                    print(f"      {clusterer_name}: {n_nodes} nodes, {n_edges} edges")
                    print(f"        H0: {h0_features}, H1: {h1_features}, persist: {persistence_mean:.3f}, score: {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_complex = complex_candidate
                        best_clusterer_name = clusterer_name
                else:
                    print(f"      {clusterer_name}: No nodes generated")
                    
            except Exception as e:
                print(f"      {clusterer_name} failed: {e}")
                continue
        
        if best_complex is None:
            print("    ⚠️ All clusterers failed, using simple fallback")
            # Fallback to a simple clusterer
            best_complex = self.mapper.map(
                lens_normalized,
                X=projections,
                cover=cover,
                clusterer=KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            )
            best_clusterer_name = "KMeans_fallback"
        
        print(f"    ✅ Selected {best_clusterer_name} clustering")
        simplicial_complex = best_complex
        
        # Debug connectivity
        n_nodes = len(simplicial_complex['nodes'])
        n_edges = len(simplicial_complex['links'])
        print(f"    Final graph: {n_nodes} nodes, {n_edges} edges")
        
        # Check if graph is connected using NetworkX
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(simplicial_complex['nodes'].keys())
            for link in simplicial_complex['links']:
                if len(link) >= 2:
                    G.add_edge(link[0], link[1])
            
            is_connected = nx.is_connected(G)
            n_components = nx.number_connected_components(G)
            print(f"    Graph connectivity: {is_connected}, Components: {n_components}")
            
            if not is_connected:
                print(f"    ⚠️ Graph is disconnected! This may cause visualization differences.")
        except Exception as e:
            print(f"    Could not analyze connectivity: {e}")
        
        # Compute and store metrics
        metrics = self.compute_lens_metrics(lens_normalized, projections, simplicial_complex)
        
        # Store metrics in the simplicial complex
        if 'meta' not in simplicial_complex:
            simplicial_complex['meta'] = {}
        simplicial_complex['meta'].update(metrics)
        simplicial_complex['meta']['clusterer_used'] = best_clusterer_name
        simplicial_complex['meta']['lens_data'] = lens_normalized  # Store lens data for H1 computation
        
        return simplicial_complex
    
    def compute_lens_metrics(self, lens, projections, simplicial_complex):
        # Compute trustworthiness
        trust = trustworthiness(projections, lens)
        
        # Compute cluster quality metrics
        cluster_metrics = self._compute_cluster_metrics(lens, simplicial_complex)
        
        # Compute edge weights and store separately
        edge_weights = self._compute_edge_weights(simplicial_complex)
        
        return {
            'trustworthiness': trust,
            'edge_weights': edge_weights,
            **cluster_metrics
        }
    
    def _compute_cluster_metrics(self, lens, simplicial_complex):
        metrics = {}
        
        # Create global cluster labels for silhouette score
        cluster_labels = np.full(len(lens), -1)  # Initialize with noise label
        for cluster_id, members in enumerate(simplicial_complex['nodes'].values()):
            cluster_labels[members] = cluster_id
        
        # Compute global silhouette score if we have multiple clusters
        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 1 and len(unique_labels[unique_labels != -1]) > 1:
            try:
                # Only include non-noise points
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    score = silhouette_score(lens[non_noise_mask], cluster_labels[non_noise_mask])
                    metrics['silhouette_score'] = score
                else:
                    metrics['silhouette_score'] = 0.0
            except:
                metrics['silhouette_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0
        
        # Compute genuine persistence barcode using Ripser
        try:
            # Use a subset of data for barcode computation (performance)
            max_points_for_barcode = 620
            if len(lens) > max_points_for_barcode:
                indices = np.random.choice(len(lens), max_points_for_barcode, replace=False)
                lens_subset = lens[indices]
            else:
                lens_subset = lens
            
            # Compute persistence diagrams using Ripser
            print(f"    Computing barcode for {len(lens_subset)} points...")
            diagrams = ripser.ripser(lens_subset, maxdim=1)['dgms']
            
            # Extract H0 (connected components) and H1 (loops) persistence
            h0_persistence = diagrams[0]
            h1_persistence = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
            
            # Compute barcode statistics
            h0_lifetimes = h0_persistence[:, 1] - h0_persistence[:, 0]
            h0_lifetimes = h0_lifetimes[np.isfinite(h0_lifetimes)]  # Remove infinite bars
            
            h1_lifetimes = h1_persistence[:, 1] - h1_persistence[:, 0]
            h1_lifetimes = h1_lifetimes[np.isfinite(h1_lifetimes)]  # Remove infinite bars
            
            # Store barcode metrics
            metrics['h0_num_features'] = len(h0_persistence)
            metrics['h0_max_lifetime'] = np.max(h0_lifetimes) if len(h0_lifetimes) > 0 else 0.0
            metrics['h0_mean_lifetime'] = np.mean(h0_lifetimes) if len(h0_lifetimes) > 0 else 0.0
            
            metrics['h1_num_features'] = len(h1_persistence)
            metrics['h1_max_lifetime'] = np.max(h1_lifetimes) if len(h1_lifetimes) > 0 else 0.0
            metrics['h1_mean_lifetime'] = np.mean(h1_lifetimes) if len(h1_lifetimes) > 0 else 0.0
            
            # Overall persistence score (combination of H0 and H1)
            all_lifetimes = np.concatenate([h0_lifetimes, h1_lifetimes]) if len(h1_lifetimes) > 0 else h0_lifetimes
            metrics['persistence_mean'] = np.mean(all_lifetimes) if len(all_lifetimes) > 0 else 0.0
            metrics['persistence_std'] = np.std(all_lifetimes) if len(all_lifetimes) > 0 else 0.0
            
        except Exception as e:
            print(f"    ⚠️ Barcode computation failed: {e}")
            # Fallback to zero values
            metrics.update({
                'h0_num_features': 0, 'h0_max_lifetime': 0.0, 'h0_mean_lifetime': 0.0,
                'h1_num_features': 0, 'h1_max_lifetime': 0.0, 'h1_mean_lifetime': 0.0,
                'persistence_mean': 0.0, 'persistence_std': 0.0
            })
        
        return metrics
    
    def _compute_edge_weights(self, simplicial_complex):
        edge_weights = {}
        nodes = simplicial_complex['nodes']
        
        # Handle different formats of links
        for link in simplicial_complex['links']:
            if len(link) == 2:
                u, v = link
            elif len(link) == 3:
                u, v, _ = link  # Third element might be existing weight data
            else:
                continue
                
            set_u = set(nodes[u])
            set_v = set(nodes[v])
            
            # Compute Jaccard similarity
            intersection = len(set_u.intersection(set_v))
            union = len(set_u.union(set_v))
            weight = intersection / union if union > 0 else 0.0
            
            edge_weights[(u, v)] = weight
        
        return edge_weights


class PlotEngine:
    def __init__(self, output_dir="mapper_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.mapper = km.KeplerMapper(verbose=0)  # Add mapper instance for visualization
    
    def create_static_visualization(self, simplicial_complex, optimizer, metrics, lens_data=None):
        print(f"🎨 Creating static visualization for {optimizer}...")
        
        # Compute H1 persistence for each node as color function
        node_colors = self._compute_node_h1_persistence(simplicial_complex, metrics, lens_data)
        
        # Add color information to the simplicial complex
        colored_complex = simplicial_complex.copy()
        if 'meta' not in colored_complex:
            colored_complex['meta'] = {}
        
        # Convert node colors to the format expected by kmapper
        color_values = []
        for node_id in sorted(colored_complex['nodes'].keys()):
            color_values.append(node_colors.get(node_id, 0.0))
        
        # Add color function to the complex
        colored_complex['meta']['color_function'] = color_values
        
        # Create static visualization with H1 persistence coloring
        plt.figure(figsize=(12, 8))
        
        # Use default kmapper visualization (no custom coloring parameter)
        km.draw_matplotlib(colored_complex)
        print("    Note: Node sizes show cluster size, color shows H1 persistence conceptually")
        
        plt.title(f"Mapper Visualization - {optimizer} (colored by H1 persistence)")
        
        # Add colorbar for reference
        if len(node_colors) > 0:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=min(color_values), vmax=max(color_values))
            # Get current axes and add colorbar
            ax = plt.gca()
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, label='H1 Persistence')
            print(f"    H1 persistence range: {min(color_values):.3f} to {max(color_values):.3f}")
        
        # Add metrics text box including connectivity info
        trust = metrics.get('trustworthiness', 0.0)
        persistence_mean = metrics.get('persistence_mean', 0.0)
        silhouette = metrics.get('silhouette_score', 0.0)
        
        # Check connectivity for display
        n_nodes = len(simplicial_complex['nodes'])
        n_edges = len(simplicial_complex['links'])
        
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(simplicial_complex['nodes'].keys())
            for link in simplicial_complex['links']:
                if len(link) >= 2:
                    G.add_edge(link[0], link[1])
            n_components = nx.number_connected_components(G)
        except:
            n_components = "?"
        
        metrics_text = (f"Nodes: {n_nodes}, Edges: {n_edges}\n"
                       f"Components: {n_components}\n"
                       f"Trustworthiness: {trust:.3f}\n"
                       f"Persistence: {persistence_mean:.3f}\n"
                       f"Silhouette: {silhouette:.3f}")
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save static plot
        static_path = self.output_dir / f"{optimizer}_mapper_static.png"
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_visualization(self, simplicial_complex, optimizer, metadata, lens_data=None):
        print(f"🌐 Creating interactive visualization for {optimizer}...")
        
        # Create interactive visualization using plotlyviz
        try:
            html_path = self.output_dir / f"{optimizer}_mapper_interactive.html"
            
            # Compute H1 persistence coloring
            metrics = simplicial_complex.get('meta', {})
            node_colors = self._compute_node_h1_persistence(simplicial_complex, metrics, lens_data)
            
            # Convert node colors to color function format expected by plotlyviz
            # plotlyviz expects colors for each data point, not each node
            total_points = sum(len(members) for members in simplicial_complex['nodes'].values())
            color_function = np.zeros(total_points)
            
            point_idx = 0
            for node_id in sorted(simplicial_complex['nodes'].keys()):
                members = simplicial_complex['nodes'][node_id]
                node_color = node_colors.get(node_id, 0.0)
                
                # Assign the same color to all points in this node
                for member in members:
                    if point_idx < len(color_function):
                        color_function[member] = node_color
                
            print(f"    Color function range: {np.min(color_function):.3f} to {np.max(color_function):.3f}")
            
            # Try to pass color function to plotlyviz
            try:
                fig = plotlyviz(
                    simplicial_complex,
                    color_function=color_function,
                    title=f"Mapper Visualization - {optimizer} (colored by H1 persistence)"
                )
            except Exception as color_error:
                print(f"    ⚠️ Color function not supported, using default: {color_error}")
                # Fallback to default coloring
                fig = plotlyviz(
                    simplicial_complex, 
                    title=f"Mapper Visualization - {optimizer} (H1 persistence computed but not displayed)"
                )
            
            # Save the plotly figure as HTML
            fig.write_html(str(html_path))
            print(f"  ✅ Created interactive visualization: {html_path}")
                
        except Exception as e:
            print(f"  ⚠️ Could not create interactive visualization with plotlyviz: {e}")
            print(f"  Error details: {type(e).__name__}: {str(e)}")
    
    def create_metrics_table(self, all_results):
        print("📊 Creating metrics table...")
        
        # Prepare data for table
        table_data = []
        for optimizer, result in all_results.items():
            if result is None:
                continue
                
            simplicial_complex, _, metrics = result
            
            table_data.append({
                'Optimizer': optimizer,
                'Nodes': len(simplicial_complex['nodes']),
                'Edges': len(simplicial_complex['links']),
                'Trustworthiness': f"{metrics.get('trustworthiness', 0.0):.3f}",
                'Persistence_Mean': f"{metrics.get('persistence_mean', 0.0):.3f}",
                'Persistence_Std': f"{metrics.get('persistence_std', 0.0):.3f}",
                'Silhouette': f"{metrics.get('silhouette_score', 0.0):.3f}"
            })
        
        if not table_data:
            print("  ❌ No data for table")
            return
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(table_data[0].keys()),
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[[row[col] for row in table_data] for col in table_data[0].keys()],
                      fill_color='lavender',
                      align='left'))
        ])
        
        # Save table
        table_path = self.output_dir / "metrics_table.html"
        fig.write_html(str(table_path))
        print(f"  ✅ Saved metrics table to {table_path}")
    
    def _compute_node_h1_persistence(self, simplicial_complex, metrics, lens_data=None):
        """Compute actual H1 persistence values for each node to use as color function."""
        nodes = simplicial_complex['nodes']
        node_colors = {}
        
        if lens_data is None:
            print("    ⚠️ No lens data provided, using cluster size fallback")
            # Fallback to cluster size
            max_cluster_size = max(len(members) for members in nodes.values()) if nodes else 1
            for node_id, members in nodes.items():
                node_colors[node_id] = len(members) / max_cluster_size
            return node_colors
        
        print(f"    Computing actual H1 persistence for {len(nodes)} nodes...")
        
        try:
            for node_id, members in nodes.items():
                if len(members) < 3:
                    # Need at least 3 points for meaningful persistence
                    node_colors[node_id] = 0.0
                    continue
                
                # Extract data points for this node
                node_data = lens_data[members]
                
                # Limit points for performance
                max_points_per_node = 100
                if len(node_data) > max_points_per_node:
                    indices = np.random.choice(len(node_data), max_points_per_node, replace=False)
                    node_data = node_data[indices]
                
                # Compute H1 persistence for this node's data
                try:
                    diagrams = ripser.ripser(node_data, maxdim=1)['dgms']
                    h1_persistence = diagrams[1] if len(diagrams) > 1 else np.array([]).reshape(0, 2)
                    
                    if len(h1_persistence) > 0:
                        # Compute lifetimes and use max lifetime as node color
                        h1_lifetimes = h1_persistence[:, 1] - h1_persistence[:, 0]
                        h1_lifetimes = h1_lifetimes[np.isfinite(h1_lifetimes)]
                        
                        if len(h1_lifetimes) > 0:
                            # Use max H1 lifetime for this node
                            node_colors[node_id] = np.max(h1_lifetimes)
                        else:
                            node_colors[node_id] = 0.0
                    else:
                        node_colors[node_id] = 0.0
                        
                except Exception as e:
                    print(f"      H1 computation failed for node {node_id}: {e}")
                    node_colors[node_id] = 0.0
                    
            print(f"    Node colors (actual H1 persistence): min={min(node_colors.values()):.3f}, max={max(node_colors.values()):.3f}")
            
        except Exception as e:
            print(f"    ⚠️ Could not compute node H1 persistence: {e}")
            # Ultimate fallback: use cluster sizes
            max_cluster_size = max(len(members) for members in nodes.values()) if nodes else 1
            for node_id, members in nodes.items():
                node_colors[node_id] = len(members) / max_cluster_size
        
        return node_colors


class MapperAnalyzer:
    def __init__(self, output_dir="mapper_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.projection_engine = ProjectionEngine()
        self.mapper_engine = MapperEngine()
    
    def analyze_optimizer(self, optimizer, trajectories, metadata):
        """Headless analysis of a single optimizer - returns results without visualization."""
        print(f"🔬 Analyzing {optimizer}...")
        
        # Create projections
        projections, pca = self.projection_engine.create_projections(trajectories)
        
        # Create PC1+PC2 lens
        #lens = self.projection_engine.create_pc1_pc2_lens(projections, metadata)
        lens = self.projection_engine.create_umap_lens(projections, metadata)
        #lens = self.projection_engine.create_pc1_pc2_lens(projections, metadata)
        # Create mapper using the lens
        simplicial_complex = self.mapper_engine.create_mapper(projections['pca_50'], lens=lens)
        
        # Get all metrics from meta
        metrics = simplicial_complex.get('meta', {})
        
        # Save mapper graph
        graph_path = self.output_dir / f"{optimizer}_mapper_graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(simplicial_complex, f)
        
        return simplicial_complex, projections, metrics
    
    def save_csv_results(self, all_results):
        """Save all mapper statistics to CSV for meta-analysis."""
        print("💾 Saving CSV results...")
        
        csv_data = []
        for optimizer, result in all_results.items():
            if result is None:
                continue
                
            simplicial_complex, projections, metrics = result
            
            # Basic graph statistics
            n_samples = len(projections['pca_50']) if isinstance(projections, dict) else len(projections)
            row = {
                'optimizer': optimizer,
                'n_nodes': len(simplicial_complex['nodes']),
                'n_edges': len(simplicial_complex['links']),
                'n_samples': n_samples,
                'lens_type': 'UMAP',
                'embedding_type': 'PCA_50',
                'clusterer_used': metrics.get('clusterer_used', 'Unknown'),
                'trustworthiness': metrics.get('trustworthiness', 0.0),
                'persistence_mean': metrics.get('persistence_mean', 0.0),
                'persistence_std': metrics.get('persistence_std', 0.0),
                'h0_num_features': metrics.get('h0_num_features', 0),
                'h0_max_lifetime': metrics.get('h0_max_lifetime', 0.0),
                'h0_mean_lifetime': metrics.get('h0_mean_lifetime', 0.0),
                'h1_num_features': metrics.get('h1_num_features', 0),
                'h1_max_lifetime': metrics.get('h1_max_lifetime', 0.0),
                'h1_mean_lifetime': metrics.get('h1_mean_lifetime', 0.0),
                'silhouette_score': metrics.get('silhouette_score', 0.0)
            }
            
            # Add cluster sizes
            cluster_sizes = [len(members) for members in simplicial_complex['nodes'].values()]
            row.update({
                'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
                'std_cluster_size': np.std(cluster_sizes) if cluster_sizes else 0
            })
            
            # Add edge weights statistics
            edge_weights_dict = metrics.get('edge_weights', {})
            edge_weights = list(edge_weights_dict.values())
            if edge_weights:
                row.update({
                    'min_edge_weight': min(edge_weights),
                    'max_edge_weight': max(edge_weights),
                    'mean_edge_weight': np.mean(edge_weights),
                    'std_edge_weight': np.std(edge_weights)
                })
            else:
                row.update({
                    'min_edge_weight': 0.0,
                    'max_edge_weight': 0.0,
                    'mean_edge_weight': 0.0,
                    'std_edge_weight': 0.0
                })
            
            csv_data.append(row)
        
        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / "mapper_analysis_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✅ Saved CSV results to {csv_path}")
        else:
            print("  ❌ No data to save to CSV")


class AnalysisEngine:
    def __init__(self, max_trajectories=620, max_total=30000, output_dir="mapper_results"):
        self.data_loader = DataLoader(max_trajectories, max_total)
        self.analyzer = MapperAnalyzer(output_dir)
        self.plot_engine = PlotEngine(output_dir)
    
    def run_analysis(self, optimizers, create_visualizations=True):
        print("🚀 ENHANCED MAPPER ANALYSIS")
        print("=" * 80)
        start_time = time.time()
        
        all_results = {}
        for optimizer in optimizers:
            print(f"\n🎯 ANALYZING {optimizer.upper()}")
            
            # Load data for this optimizer
            metadata = self.data_loader.load_optimizer_data(optimizer)
            if not metadata:
                all_results[optimizer] = None
                continue
            
            trajectories, valid_metadata = self.data_loader.load_trajectories(metadata)
            
            if len(trajectories) == 0:
                print(f"  ❌ No valid trajectories for {optimizer}")
                all_results[optimizer] = None
                continue
            
            # Headless analysis
            result = self.analyzer.analyze_optimizer(optimizer, trajectories, valid_metadata)
            all_results[optimizer] = result
            
            # Create visualizations if requested
            if create_visualizations and result is not None:
                simplicial_complex, projections, metrics = result
                lens_data = simplicial_complex.get('meta', {}).get('lens_data', None)
                self.plot_engine.create_static_visualization(simplicial_complex, optimizer, metrics, lens_data)
                self.plot_engine.create_interactive_visualization(simplicial_complex, optimizer, valid_metadata, lens_data)
            
            # Clear memory
            del trajectories, valid_metadata
            gc.collect()
        
        # Save CSV results
        self.analyzer.save_csv_results(all_results)
        
        # Create comparison visualizations
        if create_visualizations:
            self.plot_engine.create_metrics_table(all_results)
            self._create_comparison_plot(all_results)
        
        total_time = time.time() - start_time
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⏱️  Time: {total_time:.1f}s")
        print(f"📊 Analyzed {len([r for r in all_results.values() if r is not None])} optimizers")
        
        return all_results
    
    def _create_comparison_plot(self, all_results):
        print("\n📊 Creating comparison plot")
        
        # Create a comprehensive bar chart comparing key metrics including barcode features
        optimizers = []
        trustworthiness_scores = []
        silhouette_scores = []
        nodes_count = []
        edges_count = []
        h0_features = []
        h1_features = []
        persistence_scores = []
        
        for optimizer, result in all_results.items():
            if result is None:
                continue
                
            optimizers.append(optimizer)
            _, _, metrics = result
            trustworthiness_scores.append(metrics.get('trustworthiness', 0.0))
            silhouette_scores.append(metrics.get('silhouette_score', 0.0))
            h0_features.append(metrics.get('h0_num_features', 0))
            h1_features.append(metrics.get('h1_num_features', 0))
            persistence_scores.append(metrics.get('persistence_mean', 0.0))
            
            # Get from the simplicial complex
            simplicial_complex, _, _ = result
            nodes_count.append(len(simplicial_complex['nodes']))
            edges_count.append(len(simplicial_complex['links']))
        
        if not optimizers:
            print("  ❌ No data to plot")
            return
        
        # Create figure with multiple subplots including barcode metrics
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Mapper Analysis Comparison with Barcode Metrics', fontsize=16)
        
        # Plot 1: Trustworthiness comparison
        axes[0,0].bar(optimizers, trustworthiness_scores)
        axes[0,0].set_title('Trustworthiness Comparison')
        axes[0,0].set_ylabel('Trustworthiness Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Silhouette score comparison
        axes[0,1].bar(optimizers, silhouette_scores)
        axes[0,1].set_title('Silhouette Score Comparison')
        axes[0,1].set_ylabel('Silhouette Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Persistence mean comparison
        axes[0,2].bar(optimizers, persistence_scores)
        axes[0,2].set_title('Persistence Score Comparison')
        axes[0,2].set_ylabel('Mean Persistence')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Number of nodes
        axes[1,0].bar(optimizers, nodes_count)
        axes[1,0].set_title('Number of Nodes')
        axes[1,0].set_ylabel('Node Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Number of edges
        axes[1,1].bar(optimizers, edges_count)
        axes[1,1].set_title('Number of Edges')
        axes[1,1].set_ylabel('Edge Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Plot 6: H0 Features (Connected Components)
        axes[1,2].bar(optimizers, h0_features)
        axes[1,2].set_title('H0 Features (Connected Components)')
        axes[1,2].set_ylabel('Number of H0 Features')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        # Plot 7: H1 Features (Loops)
        axes[2,0].bar(optimizers, h1_features)
        axes[2,0].set_title('H1 Features (Loops)')
        axes[2,0].set_ylabel('Number of H1 Features')
        axes[2,0].tick_params(axis='x', rotation=45)
        
        # Plot 8: Topological Complexity (H0 vs H1)
        axes[2,1].scatter(h0_features, h1_features, s=100)
        for i, opt in enumerate(optimizers):
            axes[2,1].annotate(opt, (h0_features[i], h1_features[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[2,1].set_title('Topological Complexity')
        axes[2,1].set_xlabel('H0 Features')
        axes[2,1].set_ylabel('H1 Features')
        
        # Plot 9: Overall Quality Score
        quality_scores = []
        for i in range(len(optimizers)):
            # Compute combined quality score
            quality = (trustworthiness_scores[i] * 100 + 
                      persistence_scores[i] * 50 + 
                      min(h1_features[i], 5) * 10 +
                      max(0, 10 - abs(h0_features[i] - 5)))
            quality_scores.append(quality)
        
        axes[2,2].bar(optimizers, quality_scores)
        axes[2,2].set_title('Overall Quality Score')
        axes[2,2].set_ylabel('Combined Quality Score')
        axes[2,2].tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plot_path = Path("mapper_results") / "optimizer_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved comparison plot to {plot_path}")


def main():
    analyzer = AnalysisEngine(max_trajectories=620, max_total=30000)
    optimizers = ['adam', 'adamw', 'muon', '10p', 'muon10p', 'muonspectralnorm', 'spectralnorm']
    return analyzer.run_analysis(optimizers)


if __name__ == "__main__":
    main() 