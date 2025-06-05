# Suppress all warnings including OpenBLAS - MUST BE FIRST
import os
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1', 
    'OMP_NUM_THREADS': '1', 
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'GOTOBLAS_NUM_THREADS': '1',
    'BLIS_NUM_THREADS': '1'
})

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import kmapper as km
import networkx as nx
import gudhi as gd
from itertools import combinations
import seaborn as sns
from collections import defaultdict

warnings.filterwarnings('ignore')


class MultiscaleMapperAnalyzer:
    """Persistent homology analysis of multiscale mapper for optimizer trajectories."""
    
    def __init__(self, max_trajectories=50):
        self.max_trajectories = max_trajectories
        self.mapper = km.KeplerMapper(verbose=0)
    
    def load_optimizer_data(self, optimizer, epochs=range(0, 31), max_particles=20):
        """Load trajectories and metrics for a single optimizer."""
        weights_dir = Path("results") / optimizer / "pretrain_weights"
        eval_dir = Path("results") / optimizer / "evaluation_results"
        
        if not weights_dir.exists():
            return [], []
        
        # Load evaluation data
        eval_data = {}
        if eval_dir.exists():
            for eval_file in eval_dir.glob("*_metrics.pt"):
                try:
                    data = torch.load(eval_file, map_location='cpu')
                    epoch = int(eval_file.stem.split('epoch')[1].split('_')[0])
                    eval_data[epoch] = data
                except: 
                    continue
        
        # Load trajectories
        trajectories, metadata = [], []
        for epoch in epochs:
            for particle_id in range(max_particles):
                weight_file = weights_dir / f"particle{particle_id}_epoch{epoch}_weights.pt"
                if weight_file.exists():
                    try:
                        weights = torch.load(weight_file, weights_only=False, map_location='cpu')
                        weight_vector = []
                        for param in weights.values():
                            weight_vector.extend(param.flatten().detach().cpu().numpy())
                        
                        loss = None
                        if epoch in eval_data:
                            for key in ['val_loss', 'validation_loss', 'train_loss']:
                                if key in eval_data[epoch]:
                                    loss = float(eval_data[epoch][key])
                                    break
                        
                        trajectories.append(np.array(weight_vector, dtype=np.float32))
                        metadata.append({
                            'optimizer': optimizer, 'epoch': epoch, 'particle_id': particle_id,
                            'val_loss': loss if loss is not None else 1.0
                        })
                        del weights
                    except Exception as e:
                        continue
        
        # Subsample if needed
        if len(trajectories) > self.max_trajectories:
            step = len(trajectories) // self.max_trajectories
            trajectories = trajectories[::step][:self.max_trajectories]
            metadata = metadata[::step][:self.max_trajectories]
        
        return np.array(trajectories) if trajectories else np.array([]), metadata
    
    def create_lens_function(self, trajectories, metadata, lens_type='pca'):
        """Create lens function for mapper."""
        if len(trajectories) == 0:
            return None
        
        scaler = StandardScaler()
        trajectories_scaled = scaler.fit_transform(trajectories)
        
        if lens_type == 'pca':
            pca = PCA(n_components=2, random_state=42)
            return pca.fit_transform(trajectories_scaled)
        elif lens_type == 'loss_epoch':
            val_losses = np.array([m['val_loss'] for m in metadata])
            epochs = np.array([m['epoch'] for m in metadata])
            return np.column_stack([val_losses, epochs])
        else:
            return trajectories_scaled[:, :2]
    
    def create_cover_tower(self, lens_data, n_scales=6):
        """Create a tower of covers at different resolutions."""
        covers = []
        
        # Determine range for covers
        min_vals = np.min(lens_data, axis=0)
        max_vals = np.max(lens_data, axis=0)
        ranges = max_vals - min_vals
        
        # Create covers from scale 5 to 10 (where interesting results appear)
        for scale_idx in range(5, 5 + n_scales):
            # Number of cubes decreases as we go coarser - use actual scale_idx
            n_cubes = max(3, 12 - scale_idx)
            # Overlap increases as we go coarser - use actual scale_idx
            overlap = 0.2 + (scale_idx * 0.1)
            
            cover_info = {
                'n_cubes': n_cubes,
                'overlap': overlap,
                'scale_idx': scale_idx,
                'resolution': 1.0 / (scale_idx + 1)
            }
            covers.append(cover_info)
        
        return covers
    
    def compute_mapper_at_scale(self, trajectories, lens_data, cover_info):
        """Compute Mapper at a specific scale."""
        try:
            mapper_graph = self.mapper.map(
                lens=lens_data,
                X=trajectories,
                clusterer=km.cluster.DBSCAN(eps=0.3, min_samples=2),
                cover=km.Cover(n_cubes=cover_info['n_cubes'], 
                             perc_overlap=cover_info['overlap'])
            )
            return mapper_graph
        except Exception as e:
            return {'nodes': {}, 'simplices': []}
    
    def mapper_to_simplicial_complex(self, mapper_graph):
        """Convert Mapper graph to simplicial complex for persistence computation."""
        if not mapper_graph['nodes']:
            return []
        
        # Get all simplices from the mapper graph
        simplices = []
        
        # Add vertices (nodes)
        for node_id in mapper_graph['nodes'].keys():
            simplices.append([node_id])
        
        # Add edges and higher-dimensional simplices
        for simplex in mapper_graph['simplices']:
            if len(simplex) > 1:
                # Add all faces of the simplex
                for k in range(2, len(simplex) + 1):
                    for face in combinations(simplex, k):
                        simplices.append(sorted(list(face)))
        
        return simplices
    
    def compute_multiscale_persistence(self, trajectories, lens_data, metadata):
        """Compute persistent homology of multiscale mapper."""
        if len(trajectories) == 0:
            return None
        
        print(f"    Computing multiscale persistence...")
        
        # Create tower of covers
        cover_tower = self.create_cover_tower(lens_data)
        
        # Store mapper results at each scale
        scale_results = []
        all_simplices = []
        
        for i, cover_info in enumerate(cover_tower):
            mapper_graph = self.compute_mapper_at_scale(trajectories, lens_data, cover_info)
            simplices = self.mapper_to_simplicial_complex(mapper_graph)
            
            scale_info = {
                'scale_idx': cover_info['scale_idx'],
                'resolution': cover_info['resolution'],
                'n_nodes': len(mapper_graph['nodes']),
                'n_simplices': len(simplices),
                'mapper_graph': mapper_graph,
                'simplices': simplices
            }
            scale_results.append(scale_info)
            
            print(f"      Scale {cover_info['scale_idx']}: {len(mapper_graph['nodes'])} nodes, {len(simplices)} simplices")
        
        # Compute persistent homology using filtration
        persistence_diagrams = self.compute_persistence_from_scales(scale_results)
        
        return {
            'scale_results': scale_results,
            'persistence_diagrams': persistence_diagrams,
            'cover_tower': cover_tower
        }
    
    def compute_persistence_from_scales(self, scale_results):
        """Compute persistence diagrams from the multiscale sequence."""
        # Create a filtration based on scale parameter
        all_simplices = set()
        filtration_values = {}
        
        # Create mapping from node IDs to integers
        all_node_ids = set()
        for scale_info in scale_results:
            for simplex in scale_info['simplices']:
                for node_id in simplex:
                    all_node_ids.add(node_id)
        
        # Map string node IDs to integers
        node_id_to_int = {node_id: i for i, node_id in enumerate(sorted(all_node_ids))}
        
        # Collect all simplices across scales with integer mapping
        for scale_info in scale_results:
            for simplex in scale_info['simplices']:
                # Convert string node IDs to integers
                int_simplex = tuple(sorted([node_id_to_int[node_id] for node_id in simplex]))
                all_simplices.add(int_simplex)
                
                # First appearance determines birth time
                if int_simplex not in filtration_values:
                    filtration_values[int_simplex] = scale_info['resolution']
        
        # Create filtration for gudhi
        simplex_tree = gd.SimplexTree()
        
        for simplex, birth_time in filtration_values.items():
            simplex_tree.insert(list(simplex), filtration=birth_time)
        
        # Compute persistence
        try:
            simplex_tree.persistence()
            persistence_pairs = simplex_tree.persistence_intervals_in_dimension(0)  # H0
            persistence_pairs_h1 = simplex_tree.persistence_intervals_in_dimension(1)  # H1
            
            return {
                'H0': persistence_pairs,
                'H1': persistence_pairs_h1,
                'simplex_tree': simplex_tree,
                'node_mapping': node_id_to_int
            }
        except Exception as e:
            print(f"      Persistence computation failed: {e}")
            return {'H0': [], 'H1': [], 'simplex_tree': None, 'node_mapping': {}}
    
    def visualize_multiscale_results(self, results, optimizer, lens_type):
        """Create visualizations for multiscale mapper results."""
        if not results:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Mapper graphs at different scales
        n_scales = len(results['scale_results'])
        n_cols = min(4, n_scales)
        n_rows = (n_scales + n_cols - 1) // n_cols
        
        for i, scale_info in enumerate(results['scale_results']):  # Show all computed scales
            if i >= 8: break  # Limit to 8 subplots max
            
            ax = plt.subplot(3, 4, i + 1)
            
            mapper_graph = scale_info['mapper_graph']
            if mapper_graph['nodes']:
                G = nx.Graph()
                
                # Add nodes
                for node_id in mapper_graph['nodes'].keys():
                    G.add_node(node_id)
                
                # Add edges
                for simplex in mapper_graph['simplices']:
                    if len(simplex) == 2:
                        G.add_edge(simplex[0], simplex[1])
                
                if G.number_of_nodes() > 0:
                    pos = nx.spring_layout(G, k=1, iterations=50)
                    nx.draw(G, pos, ax=ax, node_size=50, node_color='lightblue', 
                           edge_color='gray', with_labels=True, font_size=8)
            
            ax.set_title(f'Scale {scale_info["scale_idx"]}\nRes: {scale_info["resolution"]:.3f}')
            ax.set_aspect('equal')
        
        # Plot 2: Persistence diagrams
        if results['persistence_diagrams']['H0'] is not None and len(results['persistence_diagrams']['H0']) > 0:
            ax = plt.subplot(3, 2, 5)
            intervals_h0 = results['persistence_diagrams']['H0']
            
            # Find max finite death time or use fallback
            finite_deaths = [d for b, d in intervals_h0 if d != float('inf')]
            if finite_deaths:
                max_death = max(finite_deaths) * 1.1
            else:
                max_death = max([b for b, d in intervals_h0]) * 2  # Use max birth time * 2 as fallback
            
            for i, (birth, death) in enumerate(intervals_h0):
                if death == float('inf'):
                    death = max_death
                ax.plot([birth, death], [i, i], 'b-', linewidth=2)
                ax.scatter([birth], [i], color='green', s=30)
                ax.scatter([death], [i], color='red', s=30)
            
            ax.set_xlabel('Scale Parameter')
            ax.set_ylabel('H0 Features')
            ax.set_title('H0 Persistence Barcode')
            ax.grid(True, alpha=0.3)
        
        if results['persistence_diagrams']['H1'] is not None and len(results['persistence_diagrams']['H1']) > 0:
            ax = plt.subplot(3, 2, 6)
            intervals_h1 = results['persistence_diagrams']['H1']
            
            # Find max finite death time or use fallback
            finite_deaths = [d for b, d in intervals_h1 if d != float('inf')]
            if finite_deaths:
                max_death = max(finite_deaths) * 1.1
            else:
                max_death = max([b for b, d in intervals_h1]) * 2  # Use max birth time * 2 as fallback
            
            for i, (birth, death) in enumerate(intervals_h1):
                if death == float('inf'):
                    death = max_death
                ax.plot([birth, death], [i, i], 'r-', linewidth=2)
                ax.scatter([birth], [i], color='green', s=30)
                ax.scatter([death], [i], color='red', s=30)
            
            ax.set_xlabel('Scale Parameter')
            ax.set_ylabel('H1 Features')
            ax.set_title('H1 Persistence Barcode')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{optimizer}_multiscale_mapper_{lens_type}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Saved: {optimizer}_multiscale_mapper_{lens_type}.png")
    
    def compute_persistence_statistics(self, persistence_diagrams):
        """Compute summary statistics from persistence diagrams."""
        stats = {}
        
        for homology_dim, intervals in persistence_diagrams.items():
            if homology_dim in ['simplex_tree', 'node_mapping']:
                continue
                
            if intervals is not None and len(intervals) > 0:
                persistences = []
                for birth, death in intervals:
                    if death != float('inf'):
                        persistences.append(death - birth)
                
                if persistences:
                    stats[homology_dim] = {
                        'n_features': len(persistences),
                        'max_persistence': max(persistences),
                        'mean_persistence': np.mean(persistences),
                        'total_persistence': sum(persistences)
                    }
                else:
                    stats[homology_dim] = {
                        'n_features': 0,
                        'max_persistence': 0,
                        'mean_persistence': 0,
                        'total_persistence': 0
                    }
            else:
                stats[homology_dim] = {
                    'n_features': 0,
                    'max_persistence': 0,
                    'mean_persistence': 0,
                    'total_persistence': 0
                }
        
        return stats
    
    def run_analysis(self, optimizers, lens_types=['pca', 'loss_epoch']):
        """Run multiscale mapper persistence analysis for all optimizers."""
        print("🚀 MULTISCALE MAPPER PERSISTENCE ANALYSIS")
        print("=" * 80)
        
        start_time = time.time()
        all_results = {}
        
        for optimizer in optimizers:
            print(f"\n🎯 ANALYZING {optimizer.upper()}")
            print("-" * 40)
            
            # Load data
            trajectories, metadata = self.load_optimizer_data(optimizer)
            if len(trajectories) == 0:
                print(f"  ❌ No data found for {optimizer}")
                continue
            
            optimizer_results = {}
            
            for lens_type in lens_types:
                print(f"  🔍 Lens function: {lens_type}")
                
                # Create lens function
                lens_data = self.create_lens_function(trajectories, metadata, lens_type)
                if lens_data is None:
                    continue
                
                # Compute multiscale persistence
                results = self.compute_multiscale_persistence(trajectories, lens_data, metadata)
                if results:
                    # Compute statistics
                    stats = self.compute_persistence_statistics(results['persistence_diagrams'])
                    results['statistics'] = stats
                    
                    optimizer_results[lens_type] = results
                    
                    # Create visualizations
                    self.visualize_multiscale_results(results, optimizer, lens_type)
                    
                    # Print summary
                    print(f"    H0 features: {stats.get('H0', {}).get('n_features', 0)}")
                    print(f"    H1 features: {stats.get('H1', {}).get('n_features', 0)}")
            
            all_results[optimizer] = optimizer_results
        
        # Create comparison plot
        self.create_persistence_comparison(all_results)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        torch.save(all_results, results_dir / "multiscale_mapper_persistence.pt")
        
        total_time = time.time() - start_time
        print(f"\n🎉 MULTISCALE ANALYSIS COMPLETE!")
        print(f"⏱️  Time: {total_time:.1f}s")
        print(f"📁 Individual plots: *_multiscale_mapper_*.png")
        print(f"📊 Comparison: persistence_comparison.png")
        print(f"💾 Results: results/multiscale_mapper_persistence.pt")
        
        return all_results
    
    def create_persistence_comparison(self, all_results):
        """Create comparison plot of persistence statistics across optimizers."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Collect data for plotting
        data = defaultdict(lambda: defaultdict(list))
        optimizers = []
        
        for optimizer, lens_results in all_results.items():
            if not lens_results:
                continue
            optimizers.append(optimizer)
            
            for lens_type, results in lens_results.items():
                stats = results.get('statistics', {})
                
                # H0 statistics
                h0_stats = stats.get('H0', {})
                data['H0_features'][optimizer].append(h0_stats.get('n_features', 0))
                data['H0_max_pers'][optimizer].append(h0_stats.get('max_persistence', 0))
                data['H0_total_pers'][optimizer].append(h0_stats.get('total_persistence', 0))
                
                # H1 statistics
                h1_stats = stats.get('H1', {})
                data['H1_features'][optimizer].append(h1_stats.get('n_features', 0))
                data['H1_max_pers'][optimizer].append(h1_stats.get('max_persistence', 0))
                data['H1_total_pers'][optimizer].append(h1_stats.get('total_persistence', 0))
        
        if optimizers:
            # Average across lens types for each optimizer
            plot_data = {}
            for metric, opt_data in data.items():
                plot_data[metric] = [np.mean(opt_data[opt]) if opt in opt_data 
                                   else 0 for opt in optimizers]
            
            # Create plots
            metrics = ['H0_features', 'H0_max_pers', 'H0_total_pers', 
                      'H1_features', 'H1_max_pers', 'H1_total_pers']
            titles = ['H0 Features', 'H0 Max Persistence', 'H0 Total Persistence',
                     'H1 Features', 'H1 Max Persistence', 'H1 Total Persistence']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[i // 3, i % 3]
                ax.bar(optimizers, plot_data[metric])
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("persistence_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved comparison: persistence_comparison.png")


def main():
    """Run the multiscale mapper persistence analysis."""
    analyzer = MultiscaleMapperAnalyzer(max_trajectories=80)
    optimizers = ['adam', 'adamw', 'muon', '10p', 'muon10p', 'muonspectralnorm', 'spectralnorm']
    lens_types = ['pca']
    
    results = analyzer.run_analysis(optimizers, lens_types)
    return results


if __name__ == "__main__":
    main() 