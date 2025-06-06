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
from sklearn.manifold import TSNE, trustworthiness
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set global random seed for reproducibility
np.random.seed(42)

# Set environment variables for better performance
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
            
            # Memory-mapped loading for better IO
            with open(file_path, 'rb') as f:
                state_dict = torch.load(file_path, map_location='cpu', mmap=True)
            
            # Extract model parameters
            weight_vector = []
            for key, param in state_dict.items():
                if isinstance(param, torch.Tensor):
                    weight_vector.extend(param.flatten().detach().cpu().numpy())
            
            # Clear memory
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
            
            if (i + 1) % 10 == 0:
                print(f"  Loaded {len(trajectories)}/{i + 1} trajectories")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"  ✅ Loaded {len(trajectories)} trajectories")
        return np.array(trajectories), valid_metadata


class ProjectionEngine:
    def __init__(self, n_components=2, perplexity=30, n_iter=1000, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.variance_threshold = VarianceThreshold(threshold=1e-6)  # Drop near-zero weights
        self.pca = PCA(n_components=min(50, n_components), random_state=random_state)
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                        n_iter=n_iter, random_state=random_state)
    
    def preprocess_weights(self, weights):
        # Log-scale absolute values
        weights_log = np.sign(weights) * np.log1p(np.abs(weights))
        
        # Remove low-variance features
        weights_var = self.variance_threshold.fit_transform(weights_log)
        
        # Standardize
        weights_scaled = self.scaler.fit_transform(weights_var)
        
        return weights_scaled
    
    def create_projections(self, trajectories):
        print("🎯 Creating projections...")
        
        # Preprocess weights
        trajectories_processed = self.preprocess_weights(trajectories)
        
        # Apply PCA for dimensionality reduction
        pca_result = self.pca.fit_transform(trajectories_processed)
        
        # Apply t-SNE
        tsne_result = self.tsne.fit_transform(pca_result)
        
        return tsne_result, self.pca
    
    @staticmethod
    def create_lens_functions(projections, metadata):
        print("🔍 Creating lens functions...")
        lens_functions = {}
        
        if 'pca_2d' in projections:
            # Standardize PC1 and PC2 separately
            pc1 = projections['pca_2d'][:, 0]
            pc2 = projections['pca_2d'][:, 1]
            pc1_norm = (pc1 - np.mean(pc1)) / (np.std(pc1) + 1e-8)
            pc2_norm = (pc2 - np.mean(pc2)) / (np.std(pc2) + 1e-8)
            lens_functions['pc1_pc2'] = np.column_stack([pc1_norm, pc2_norm])
        
        if 'pca_2d' in projections and metadata:
            pc1 = projections['pca_2d'][:, 0]
            pc1_norm = (pc1 - np.mean(pc1)) / (np.std(pc1) + 1e-8)
            
            # PC1 + Validation Loss (if available)
            val_losses = [m.get('val_loss') for m in metadata]
            if any(v is not None for v in val_losses):
                val_losses = np.array([v if v is not None else np.mean([x for x in val_losses if x is not None]) for v in val_losses])
                val_loss_norm = (val_losses - np.mean(val_losses)) / (np.std(val_losses) + 1e-8)
                lens_functions['pc1_valloss'] = np.column_stack([pc1_norm, val_loss_norm])
            
            # PC1 + Epoch
            epochs = np.array([m['epoch'] for m in metadata])
            epoch_norm = (epochs - np.mean(epochs)) / (np.std(epochs) + 1e-8)
            lens_functions['pc1_epoch'] = np.column_stack([pc1_norm, epoch_norm])
        
        if 'pca_extended' in projections and projections['pca_extended'].shape[1] >= 3:
            pc1 = projections['pca_extended'][:, 0]
            pc3 = projections['pca_extended'][:, 2]
            pc1_norm = (pc1 - np.mean(pc1)) / (np.std(pc1) + 1e-8)
            pc3_norm = (pc3 - np.mean(pc3)) / (np.std(pc3) + 1e-8)
            lens_functions['pc1_pc3'] = np.column_stack([pc1_norm, pc3_norm])
        
        if 'tsne_2d' in projections:
            tsne = projections['tsne_2d']
            tsne_norm = (tsne - np.mean(tsne, axis=0)) / (np.std(tsne, axis=0) + 1e-8)
            lens_functions['tsne'] = tsne_norm
        
        return lens_functions


class MapperEngine:
    def __init__(self, n_cubes=10, perc_overlap=0.5, n_clusters=3, random_state=42):
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
        
        # Use density-aware cover
        try:
            cover = km.BallsCover(n_balls=self.n_cubes, perc_overlap=self.perc_overlap)
        except AttributeError:
            # Fallback to regular cover if BallsCover not available
            print(f"Balls Cover error {e}")
            cover = km.Cover(n_cubes=self.n_cubes, perc_overlap=self.perc_overlap)
        
        # Create the simplicial complex using normalized lens
        simplicial_complex = self.mapper.map(
            lens_normalized,
            X=projections,
            cover=cover,
            clusterer=DBSCAN(eps=0.5, min_samples=3)
        )
        
        # Compute and store metrics
        metrics = self.compute_lens_metrics(lens_normalized, projections, simplicial_complex)
        
        # Store metrics in the simplicial complex
        if 'meta' not in simplicial_complex:
            simplicial_complex['meta'] = {}
        simplicial_complex['meta'].update(metrics)
        
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
        
        # Compute persistence for DBSCAN clusters
        persistence_scores = []
        for node_id, members in simplicial_complex['nodes'].items():
            if len(members) > 1:
                # Compute persistence as min distance to non-members
                member_distances = cdist(lens[members], lens)
                non_member_mask = ~np.isin(np.arange(len(lens)), members)
                if np.any(non_member_mask):
                    min_distances = np.min(member_distances[:, non_member_mask], axis=1)
                    persistence_scores.extend(min_distances)
        
        metrics['persistence_mean'] = np.mean(persistence_scores) if persistence_scores else 0.0
        metrics['persistence_std'] = np.std(persistence_scores) if persistence_scores else 0.0
        
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
    
    def create_static_visualization(self, simplicial_complex, optimizer, metrics):
        print(f"🎨 Creating static visualization for {optimizer}...")
        
        # Create static visualization
        plt.figure(figsize=(12, 8))
        km.draw_matplotlib(simplicial_complex)
        plt.title(f"Mapper Visualization - {optimizer}")
        
        # Add metrics text box
        trust = metrics.get('trustworthiness', 0.0)
        persistence_mean = metrics.get('persistence_mean', 0.0)
        silhouette = metrics.get('silhouette_score', 0.0)
        
        metrics_text = (f"Trustworthiness: {trust:.3f}\n"
                       f"Persistence: {persistence_mean:.3f}\n"
                       f"Silhouette: {silhouette:.3f}")
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save static plot
        static_path = self.output_dir / f"{optimizer}_mapper_static.png"
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_visualization(self, simplicial_complex, optimizer, metadata):
        print(f"🌐 Creating interactive visualization for {optimizer}...")
        
        # Create interactive visualization using plotlyviz
        try:
            html_path = self.output_dir / f"{optimizer}_mapper_interactive.html"
            
            # Create custom tooltips
            custom_tooltips = [f"Epoch: {m['epoch']}, Loss: {m.get('val_loss', 'N/A')}" for m in metadata]
            
            # Use kmapper's plotlyviz for interactive visualization
            fig = km.plotlyviz(
                simplicial_complex, 
                title=f"Mapper Visualization - {optimizer}",
                custom_tooltips=custom_tooltips
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
        
        # Create mapper
        simplicial_complex = self.mapper_engine.create_mapper(projections)
        
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
            row = {
                'optimizer': optimizer,
                'n_nodes': len(simplicial_complex['nodes']),
                'n_edges': len(simplicial_complex['links']),
                'n_samples': len(projections),
                'trustworthiness': metrics.get('trustworthiness', 0.0),
                'persistence_mean': metrics.get('persistence_mean', 0.0),
                'persistence_std': metrics.get('persistence_std', 0.0),
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
                self.plot_engine.create_static_visualization(simplicial_complex, optimizer, metrics)
                self.plot_engine.create_interactive_visualization(simplicial_complex, optimizer, valid_metadata)
            
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
        
        # Create a simple bar chart comparing key metrics
        optimizers = []
        trustworthiness_scores = []
        silhouette_scores = []
        nodes_count = []
        edges_count = []
        
        for optimizer, result in all_results.items():
            if result is None:
                continue
                
            optimizers.append(optimizer)
            _, _, metrics = result
            trustworthiness_scores.append(metrics.get('trustworthiness', 0.0))
            silhouette_scores.append(metrics.get('silhouette_score', 0.0))
            
            # Get from the simplicial complex
            simplicial_complex, _, _ = result
            nodes_count.append(len(simplicial_complex['nodes']))
            edges_count.append(len(simplicial_complex['links']))
        
        if not optimizers:
            print("  ❌ No data to plot")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Mapper Analysis Comparison', fontsize=16)
        
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
        
        # Plot 3: Number of nodes
        axes[1,0].bar(optimizers, nodes_count)
        axes[1,0].set_title('Number of Nodes')
        axes[1,0].set_ylabel('Node Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Number of edges
        axes[1,1].bar(optimizers, edges_count)
        axes[1,1].set_title('Number of Edges')
        axes[1,1].set_ylabel('Edge Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
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