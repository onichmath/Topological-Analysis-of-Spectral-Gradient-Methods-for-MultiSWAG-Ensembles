import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import kmapper as km
import networkx as nx
import gc
import os
from collections import Counter

# Suppress warnings
os.environ.update({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, max_trajectories=60, max_total=300):
        self.max_trajectories = max_trajectories
        self.max_total = max_total
    
    def load_data(self, optimizers, epochs=range(10, 31), max_particles=20):
        print("🔄 Loading trajectories with metrics...")
        all_metadata = []
        
        for optimizer in optimizers:
            weights_dir = Path("results") / optimizer / "pretrain_weights"
            eval_dir = Path("results") / optimizer / "evaluation_results"
            
            if not weights_dir.exists():
                print(f"  ❌ No weights directory for {optimizer}")
                continue
            
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
                        # Get loss metric (prefer val_loss, fallback to train_loss)
                        loss = None
                        if epoch in eval_data:
                            for key in ['val_loss', 'validation_loss', 'train_loss']:
                                if key in eval_data[epoch]:
                                    loss = float(eval_data[epoch][key])
                                    break
                        
                        optimizer_files.append({
                            'file': weight_file, 'optimizer': optimizer, 'epoch': epoch,
                            'particle_id': particle_id, 'val_loss': loss
                        })
            
            # Subsample
            if len(optimizer_files) > self.max_trajectories:
                step = len(optimizer_files) // self.max_trajectories
                optimizer_files = optimizer_files[::step][:self.max_trajectories]
            
            all_metadata.extend(optimizer_files)
            print(f"  ✅ {optimizer}: {len(optimizer_files)} files")
        
        # Global subsample
        if len(all_metadata) > self.max_total:
            step = len(all_metadata) // self.max_total
            all_metadata = all_metadata[::step][:self.max_total]
        
        return all_metadata
    
    def load_trajectories(self, metadata_list):
        print("📦 Loading weight trajectories...")
        trajectories, valid_metadata = [], []
        
        for i, meta in enumerate(metadata_list):
            try:
                weights = torch.load(meta['file'], weights_only=False, map_location='cpu')
                weight_vector = []
                for param in weights.values():
                    weight_vector.extend(param.flatten().detach().cpu().numpy())
                
                trajectories.append(np.array(weight_vector, dtype=np.float32))
                valid_metadata.append(meta)
                del weights
                
                if (i + 1) % 50 == 0:
                    print(f"  Loaded {i + 1}/{len(metadata_list)} trajectories")
                    gc.collect()
            except Exception as e:
                print(f"  Error loading {meta['file']}: {e}")
                continue
        
        print(f"  ✅ Loaded {len(trajectories)} trajectories")
        return np.array(trajectories), valid_metadata


class ProjectionEngine:
    @staticmethod
    def create_projections(trajectories):
        print("🎯 Creating projections...")
        scaler = StandardScaler()
        trajectories_scaled = scaler.fit_transform(trajectories)
        
        pca = PCA(n_components=min(10, len(trajectories)-1), random_state=42)
        pca_result = pca.fit_transform(trajectories_scaled)
        
        projections = {'pca_2d': pca_result[:, :2], 'pca_extended': pca_result}
        
        if len(trajectories) <= 1000:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(trajectories)//4))
            projections['tsne_2d'] = tsne.fit_transform(trajectories_scaled[:, :50])
        
        return projections, pca
    
    @staticmethod
    def create_lens_functions(projections, metadata):
        print("🔍 Creating lens functions...")
        lens_functions = {}
        
        if 'pca_2d' in projections:
            lens_functions['pc1_pc2'] = projections['pca_2d']
        
        if 'pca_2d' in projections and metadata:
            pc1 = projections['pca_2d'][:, 0]
            
            # PC1 + Validation Loss (if available)
            val_losses = [m.get('val_loss') for m in metadata]
            if any(v is not None for v in val_losses):
                val_losses = np.array([v if v is not None else np.mean([x for x in val_losses if x is not None]) for v in val_losses])
                val_loss_norm = (val_losses - np.mean(val_losses)) / (np.std(val_losses) + 1e-8)
                lens_functions['pc1_valloss'] = np.column_stack([pc1, val_loss_norm])
            
            # PC1 + Epoch
            epochs = np.array([m['epoch'] for m in metadata])
            epoch_norm = (epochs - np.mean(epochs)) / (np.std(epochs) + 1e-8)
            lens_functions['pc1_epoch'] = np.column_stack([pc1, epoch_norm])
        
        if 'pca_extended' in projections and projections['pca_extended'].shape[1] >= 3:
            lens_functions['pc1_pc3'] = projections['pca_extended'][:, [0, 2]]
        
        if 'tsne_2d' in projections:
            lens_functions['tsne'] = projections['tsne_2d']
        
        return lens_functions


class MapperEngine:
    def __init__(self):
        self.mapper = km.KeplerMapper(verbose=0)
    
    def run_mapper(self, trajectories, lens, lens_name, metadata):
        print(f"🗺️  Running Mapper on {lens_name}")
        n_samples = len(trajectories)
        
        # Adaptive clustering
        if n_samples < 10:
            clusterers = {'kmeans': KMeans(n_clusters=min(3, n_samples), random_state=42, n_init=10)}
        else:
            eps = 0.5 if n_samples < 50 else 0.3
            min_samples = max(2, min(3, n_samples // 10))
            clusterers = {
                'dbscan': DBSCAN(eps=eps, min_samples=min_samples),
                'kmeans': KMeans(n_clusters=min(8, n_samples // 5), random_state=42, n_init=10),
                'hierarchical': AgglomerativeClustering(n_clusters=min(6, n_samples // 8), linkage='ward')
            }
        
        results = {}
        for cluster_name, clusterer in clusterers.items():
            try:
                n_cubes = min(8, max(3, n_samples // 10))
                overlap = 0.4 if n_samples < 50 else 0.3
                
                mapper_graph = self.mapper.map(
                    lens=lens, X=trajectories, clusterer=clusterer,
                    cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
                )
                
                if len(mapper_graph['nodes']) > 0:
                    analysis = self._analyze_graph(mapper_graph, metadata)
                    results[cluster_name] = {'mapper_graph': mapper_graph, 'analysis': analysis}
                    print(f"    ✅ {cluster_name}: {analysis['n_nodes']} nodes")
            except Exception as e:
                print(f"    ❌ {cluster_name} failed: {e}")
        
        return results
    
    def _analyze_graph(self, mapper_graph, metadata):
        G = nx.Graph()
        
        if len(mapper_graph['nodes']) == 0:
            return {'graph': G, 'n_nodes': 0, 'n_components': 0, 'avg_node_size': 0}
        
        # Add nodes with metadata
        for node_id, node_members in mapper_graph['nodes'].items():
            node_data = [metadata[i] for i in node_members]
            G.add_node(node_id, 
                      members=node_members, size=len(node_members),
                      optimizers=list(set([m['optimizer'] for m in node_data])),
                      avg_epoch=np.mean([m['epoch'] for m in node_data]),
                      avg_val_loss=np.mean([m['val_loss'] for m in node_data if m['val_loss'] is not None]),
                      epochs=[m['epoch'] for m in node_data],
                      val_losses=[m['val_loss'] for m in node_data])
        
        # Add edges
        for simplex in mapper_graph['simplices']:
            if len(simplex) == 2:
                G.add_edge(simplex[0], simplex[1])
        
        # Safe metrics computation
        def safe_compute(func, default=0):
            try:
                return func()
            except:
                return default
        
        return {
            'graph': G,
            'n_nodes': len(mapper_graph['nodes']),
            'n_components': safe_compute(lambda: nx.number_connected_components(G), 1 if G.number_of_nodes() > 0 else 0),
            'avg_node_size': np.mean([len(members) for members in mapper_graph['nodes'].values()]),
            'clustering_coefficient': safe_compute(lambda: nx.average_clustering(G) if G.number_of_edges() > 0 else 0),
            'density': safe_compute(lambda: nx.density(G))
        }


class VisualizationEngine:
    def __init__(self):
        self.mapper = km.KeplerMapper(verbose=0)
    
    def create_visualizations(self, results_dict, lens_name, metadata, optimizer):
        print(f"🎨 Creating visualizations for {optimizer} - {lens_name}")
        
        # Static visualization
        fig, axes = plt.subplots(1, len(results_dict), figsize=(6*len(results_dict), 6))
        if len(results_dict) == 1:
            axes = [axes]
        
        for i, (method, result) in enumerate(results_dict.items()):
            ax = axes[i] if len(results_dict) > 1 else axes[0]
            G = result['analysis']['graph']
            
            if G.number_of_nodes() > 0:
                pos = nx.spring_layout(G, k=1, iterations=50)
                node_epochs = [G.nodes[node]['avg_epoch'] for node in G.nodes()]
                node_sizes = [G.nodes[node]['size'] * 50 for node in G.nodes()]
                
                nx.draw(G, pos, ax=ax, node_color=node_epochs, node_size=node_sizes,
                       cmap='viridis', with_labels=True, font_size=8)
                ax.set_title(f'{optimizer.upper()} - {method.upper()}\n{result["analysis"]["n_nodes"]} nodes')
            else:
                ax.text(0.5, 0.5, f'{optimizer.upper()} - {method.upper()}\nNo nodes', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        filename = f"mapper_{optimizer}_{lens_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Interactive HTML
        self._create_html(results_dict, lens_name, metadata, optimizer)
        
        return filename
    
    def _create_html(self, results_dict, lens_name, metadata, optimizer):
        for method, result in results_dict.items():
            mapper_graph = result['mapper_graph']
            
            # Prepare node colors (by epoch)
            node_colors = []
            for node_id, node_members in mapper_graph['nodes'].items():
                node_meta = [metadata[i] for i in node_members]
                epochs = [m['epoch'] for m in node_meta]
                node_colors.append(np.mean(epochs))
            
            # Create HTML with fallback
            filename = f"mapper_{optimizer}_{lens_name}_{method}"
            for attempt in [
                lambda: self.mapper.visualize(mapper_graph, color_values=node_colors,
                                            path_html=f"{filename}.html",
                                            title=f"{optimizer.upper()} - {lens_name} - {method}"),
                lambda: self.mapper.visualize(mapper_graph, path_html=f"{filename}.html",
                                            title=f"{optimizer.upper()} - {lens_name} - {method}")
            ]:
                try:
                    attempt()
                    break
                except:
                    continue


class AnalysisEngine:
    def __init__(self, max_trajectories=60, max_total=300):
        self.data_loader = DataLoader(max_trajectories, max_total)
        self.projection_engine = ProjectionEngine()
        self.mapper_engine = MapperEngine()
        self.visualization_engine = VisualizationEngine()
    
    def run_analysis(self, optimizers):
        print("🚀 OPTIMIZER-SPECIFIC MAPPER ANALYSIS")
        print("=" * 80)
        start_time = time.time()
        
        # Load all data
        metadata = self.data_loader.load_data(optimizers)
        if not metadata:
            print("❌ No trajectories found.")
            return None
        
        trajectories, valid_metadata = self.data_loader.load_trajectories(metadata)
        
        # Group by optimizer
        optimizer_data = {}
        for i, meta in enumerate(valid_metadata):
            opt = meta['optimizer']
            if opt not in optimizer_data:
                optimizer_data[opt] = {'trajectories': [], 'metadata': []}
            optimizer_data[opt]['trajectories'].append(trajectories[i])
            optimizer_data[opt]['metadata'].append(meta)
        
        # Run analysis for each optimizer separately
        all_results = {}
        for optimizer, data in optimizer_data.items():
            print(f"\n🎯 ANALYZING {optimizer.upper()}")
            opt_trajectories = np.array(data['trajectories'])
            opt_metadata = data['metadata']
            
            # Create projections for this optimizer
            projections, pca = self.projection_engine.create_projections(opt_trajectories)
            lens_functions = self.projection_engine.create_lens_functions(projections, opt_metadata)
            
            optimizer_results = {}
            for lens_name, lens in lens_functions.items():
                lens_results = self.mapper_engine.run_mapper(opt_trajectories, lens, lens_name, opt_metadata)
                if lens_results:
                    self.visualization_engine.create_visualizations(lens_results, lens_name, opt_metadata, optimizer)
                    optimizer_results[lens_name] = lens_results
            
            all_results[optimizer] = optimizer_results
        
        # Create comparison plot
        self._create_comparison_plot(all_results)
        
        # Save results
        Path("results").mkdir(exist_ok=True)
        torch.save({'optimizer_results': all_results, 'metadata': valid_metadata}, 
                  "results/optimizer_mapper_results.pt")
        
        total_time = time.time() - start_time
        print(f"\n🎉 ANALYSIS COMPLETE! Time: {total_time:.1f}s")
        print(f"📊 Analyzed {len(all_results)} optimizers")
        print(f"📁 Plots: mapper_*.png")
        print(f"🌐 Interactive: mapper_*.html")
        
        return all_results
    
    def _create_comparison_plot(self, all_results):
        # Create heatmap comparison
        optimizers = list(all_results.keys())
        metrics = ['n_nodes', 'n_components', 'avg_node_size']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            data_matrix = []
            
            for opt in optimizers:
                row = []
                for lens_name in ['pc1_pc2', 'pc1_valloss', 'pc1_epoch']:
                    if lens_name in all_results[opt] and 'dbscan' in all_results[opt][lens_name]:
                        value = all_results[opt][lens_name]['dbscan']['analysis'][metric]
                        row.append(float(value) if value != 'disconnected' else 0)
                    else:
                        row.append(0)
                data_matrix.append(row)
            
            im = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
            ax.set_xticks(range(3))
            ax.set_xticklabels(['PC1+PC2', 'PC1+ValLoss', 'PC1+Epoch'], rotation=45)
            ax.set_yticks(range(len(optimizers)))
            ax.set_yticklabels(optimizers)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig("optimizer_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved comparison: optimizer_comparison.png")


def main():
    analyzer = AnalysisEngine(max_trajectories=60, max_total=300)
    optimizers = ['adam', 'adamw', 'muon', '10p', 'muon10p', 'muonspectralnorm', 'spectralnorm']
    return analyzer.run_analysis(optimizers)


if __name__ == "__main__":
    main() 