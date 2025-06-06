import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
import gudhi
from utils.projections import ProjectionEngine

warnings.filterwarnings('ignore')


class ZigzagPersistenceAnalyzer:
    """Zigzag persistence analysis for optimization trajectories using PCA projections."""
    
    def __init__(self, max_dimension=2):
        self.max_dimension = max_dimension
        self.projection_engine = ProjectionEngine()
        
    def compute_adaptive_edge_length(self, trajectories, percentile=50):
        """Compute adaptive edge length based on pairwise distances."""
        # Sample a subset for efficiency
        n_sample = min(100, len(trajectories))
        sample_indices = np.random.choice(len(trajectories), n_sample, replace=False)
        sample_points = trajectories[sample_indices]
        
        # Compute pairwise distances
        distances = []
        for i in range(len(sample_points)):
            for j in range(i+1, len(sample_points)):
                dist = np.linalg.norm(sample_points[i] - sample_points[j])
                distances.append(dist)
        
        # Use percentile of distances as max_edge_length
        max_edge_length = np.percentile(distances, percentile)
        print(f"    Computed adaptive edge length: {max_edge_length:.4f}")
        return max_edge_length
        
    def create_zigzag_sequence(self, trajectories):
        """Create zigzag sequence using all trajectory points."""
        n_points = len(trajectories)
        print(f"    Using all {n_points} trajectory points")
        
        point_sets = []
        actions = []
        
        # Phase 1: Add points sequentially
        current_indices = []
        for i in range(n_points):
            current_indices.append(i)
            point_sets.append(current_indices.copy())
            actions.append('add')
            
        # Phase 2: Remove every other point
        if len(current_indices) > 6:
            removal_indices = list(range(1, len(current_indices), 2))
            for idx in sorted(removal_indices, reverse=True):
                current_indices.pop(idx)
                point_sets.append(current_indices.copy())
                actions.append('remove')
                
        # Phase 3: Add some points back
        available_indices = list(range(n_points))
        missing_indices = [i for i in available_indices if i not in current_indices]
        
        # Add back a reasonable number of points (up to 20% of removed)
        readd_count = min(len(missing_indices), max(10, len(removal_indices) // 5))
        for i in missing_indices[:readd_count]:
            current_indices.append(i)
            point_sets.append(current_indices.copy())
            actions.append('readd')
            
        return point_sets, actions
    
    def load_projection_data(self, optimizers):
        """Load PCA extended projections for zigzag analysis."""
        print("Loading projection data for zigzag analysis...")
        
        all_data = {}
        
        for optimizer in optimizers:
            projection_data = self.projection_engine.load_projections(optimizer)
            
            if projection_data and 'projections' in projection_data:
                pca_extended = projection_data['projections'].get('pca_extended')
                metadata = projection_data.get('metadata', [])
                
                if pca_extended is not None and len(pca_extended) > 10:
                    all_data[optimizer] = {
                        'trajectories': pca_extended,
                        'metadata': metadata,
                        'n_points': len(pca_extended)
                    }
                    print(f"  Loaded {len(pca_extended)} projections for {optimizer}")
                    
        return all_data
    
    def compute_persistence_for_points(self, points, max_edge_length):
        """Compute persistence for a set of points with adaptive edge length."""
        if len(points) < 3:
            return {0: [], 1: [], 2: []}
            
        try:
            rips_complex = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
            persistence = simplex_tree.persistence()
            
            persistence_by_dim = {0: [], 1: [], 2: []}
            
            for dimension, (birth, death) in persistence:
                if dimension <= self.max_dimension:
                    if death != float('inf'):
                        persistence_by_dim[dimension].append((birth, death))
                    else:
                        persistence_by_dim[dimension].append((birth, np.inf))
                        
            return persistence_by_dim
            
        except Exception:
            return {0: [], 1: [], 2: []}
    
    def track_features(self, point_sets, actions, trajectories):
        """Track topological features through zigzag sequence."""
        # Compute adaptive edge length based on full trajectory
        max_edge_length = self.compute_adaptive_edge_length(trajectories)
        
        feature_tracker = {}
        next_feature_id = 0
        zigzag_barcode = []
        
        for step, (indices, action) in enumerate(zip(point_sets, actions)):
            if len(indices) < 3:
                continue
                
            points = trajectories[indices]
            persistence_data = self.compute_persistence_for_points(points, max_edge_length)
            
            current_features = set()
            for dim in [0, 1, 2]:
                for birth, death in persistence_data[dim]:
                    # Create signature without step dependency
                    if death == np.inf:
                        # For infinite features, use birth and dimension only
                        feature_sig = f"dim{dim}_b{birth:.4f}_dinf"
                    else:
                        # For finite features, use birth, death, and dimension
                        # Round to avoid floating point precision issues
                        feature_sig = f"dim{dim}_b{birth:.4f}_d{death:.4f}"
                    current_features.add((feature_sig, dim, birth, death))
            
            # Track new features and update existing ones
            for feature_sig, dim, birth, death in current_features:
                if feature_sig not in feature_tracker:
                    # New feature
                    feature_tracker[feature_sig] = {
                        'birth_step': step,
                        'last_seen': step,
                        'dimension': dim,
                        'actions': [action],
                        'feature_id': next_feature_id,
                        'birth_value': birth,
                        'death_value': death
                    }
                    next_feature_id += 1
                else:
                    # Existing feature continues
                    feature_tracker[feature_sig]['last_seen'] = step
                    feature_tracker[feature_sig]['actions'].append(action)
        
        # Create barcode
        for feature_data in feature_tracker.values():
            lifetime = feature_data['last_seen'] - feature_data['birth_step']
            stability = len(feature_data['actions']) / (lifetime + 1)
            
            zigzag_barcode.append({
                'birth': feature_data['birth_step'],
                'death': feature_data['last_seen'],
                'dimension': feature_data['dimension'],
                'lifetime': lifetime,
                'stability': stability,
                'actions': feature_data['actions'],
                'birth_value': feature_data['birth_value'],
                'death_value': feature_data['death_value']
            })
            
        return zigzag_barcode
    
    def compute_zigzag_stability(self, point_sets, actions, trajectories):
        """
        Compute zigzag stability based on Oudot & Sheehy (2015).
        Measures how persistence diagrams change across zigzag sequence.
        """
        if len(point_sets) < 2:
            return 0.0
        
        # Compute adaptive edge length
        max_edge_length = self.compute_adaptive_edge_length(trajectories)
        persistence_diagrams = []
        
        # Compute persistence diagram for each step
        for indices in point_sets:
            if len(indices) < 3:
                persistence_diagrams.append([])
                continue
                
            points = trajectories[indices]
            persistence_data = self.compute_persistence_for_points(points, max_edge_length)
            
            # Flatten to list of (birth, death) pairs
            diagram = []
            for dim in [0, 1, 2]:
                for birth, death in persistence_data[dim]:
                    if death != np.inf:  # Only finite features
                        diagram.append((birth, death))
            
            persistence_diagrams.append(diagram)
        
        # Compute stability as inverse of diagram variance
        if len(persistence_diagrams) < 2:
            return 0.0
            
        # Simple stability measure: inverse coefficient of variation of feature counts
        feature_counts = [len(diagram) for diagram in persistence_diagrams]
        
        if len(feature_counts) == 0 or np.mean(feature_counts) == 0:
            return 0.0
            
        cv = np.std(feature_counts) / (np.mean(feature_counts) + 1e-8)
        stability = 1.0 / (1.0 + cv)  # Higher stability = lower variation
        
        return stability

    def analyze_optimizer(self, optimizer_data, optimizer_name):
        """Analyze single optimizer with zigzag persistence."""
        print(f"Analyzing {optimizer_name}...")
        
        trajectories = optimizer_data['trajectories']
        print(f"  Trajectory shape: {trajectories.shape}")
        
        # Create zigzag sequence
        point_sets, actions = self.create_zigzag_sequence(trajectories)
        print(f"  Created {len(point_sets)} zigzag steps")
        
        # Track features
        zigzag_barcode = self.track_features(point_sets, actions, trajectories)
        
        # Compute metrics
        if not zigzag_barcode:
            print(f"  WARNING: No features found for {optimizer_name}")
            return {
                'total_features': 0,
                'avg_lifetime': 0.0,
                'avg_stability': 0.0,
                'h0_features': 0,
                'h1_features': 0,
                'h2_features': 0,
                'persistent_entropy': 0.0,
                'zigzag_stability': 0.0,
                'zigzag_barcode': []
            }
        
        lifetimes = [f['lifetime'] for f in zigzag_barcode]
        stabilities = [f['stability'] for f in zigzag_barcode]
        
        h0_count = sum(1 for f in zigzag_barcode if f['dimension'] == 0)
        h1_count = sum(1 for f in zigzag_barcode if f['dimension'] == 1)
        h2_count = sum(1 for f in zigzag_barcode if f['dimension'] == 2)
        
        # Compute research-based metrics
        persistent_entropy = self.compute_persistent_entropy(zigzag_barcode)
        zigzag_stability = self.compute_zigzag_stability(point_sets, actions, trajectories)
        
        results = {
            'total_features': len(zigzag_barcode),
            'avg_lifetime': np.mean(lifetimes),
            'avg_stability': np.mean(stabilities),
            'h0_features': h0_count,
            'h1_features': h1_count,
            'h2_features': h2_count,
            'persistent_entropy': persistent_entropy,
            'zigzag_stability': zigzag_stability,
            'zigzag_barcode': zigzag_barcode
        }
        
        print(f"  Features: H0={h0_count}, H1={h1_count}, H2={h2_count} (total={results['total_features']})")
        print(f"  Avg lifetime: {results['avg_lifetime']:.2f}, "
              f"Entropy: {results['persistent_entropy']:.3f}, "
              f"Zigzag stability: {results['zigzag_stability']:.3f}")
        
        return results
    
    def create_visualizations(self, all_results):
        """Create zigzag persistence visualizations."""
        print("Creating visualizations...")
        
        optimizers = list(all_results.keys())
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Total features
        feature_counts = [all_results[opt]['total_features'] for opt in optimizers]
        axes[0].bar(optimizers, feature_counts, alpha=0.7)
        axes[0].set_title('Total Zigzag Features')
        axes[0].set_ylabel('Feature Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Average lifetimes
        lifetimes = [all_results[opt]['avg_lifetime'] for opt in optimizers]
        axes[1].bar(optimizers, lifetimes, alpha=0.7)
        axes[1].set_title('Average Feature Lifetime')
        axes[1].set_ylabel('Lifetime (steps)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. Stability scores
        stabilities = [all_results[opt]['avg_stability'] for opt in optimizers]
        axes[2].bar(optimizers, stabilities, alpha=0.7)
        axes[2].set_title('Zigzag Stability Score')
        axes[2].set_ylabel('Stability')
        axes[2].tick_params(axis='x', rotation=45)
        
        # 4. Dimension distribution (H0, H1, H2)
        h0_counts = [all_results[opt]['h0_features'] for opt in optimizers]
        h1_counts = [all_results[opt]['h1_features'] for opt in optimizers]
        h2_counts = [all_results[opt]['h2_features'] for opt in optimizers]
        
        x_pos = np.arange(len(optimizers))
        width = 0.25
        
        axes[3].bar(x_pos - width, h0_counts, width, label='H0', alpha=0.7)
        axes[3].bar(x_pos, h1_counts, width, label='H1', alpha=0.7)
        axes[3].bar(x_pos + width, h2_counts, width, label='H2', alpha=0.7)
        axes[3].set_title('Feature Dimension Distribution')
        axes[3].set_ylabel('Feature Count')
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(optimizers, rotation=45)
        axes[3].legend()
        
        # 5. Persistent Entropy (replaces Action Robustness)
        entropies = [all_results[opt]['persistent_entropy'] for opt in optimizers]
        axes[4].bar(optimizers, entropies, alpha=0.7)
        axes[4].set_title('Persistent Entropy')
        axes[4].set_ylabel('Entropy (bits)')
        axes[4].tick_params(axis='x', rotation=45)
        
        # 6. Zigzag Stability (replaces Complexity Ranking)
        zigzag_stabilities = [all_results[opt]['zigzag_stability'] for opt in optimizers]
        axes[5].bar(optimizers, zigzag_stabilities, alpha=0.7)
        axes[5].set_title('Zigzag Stability')
        axes[5].set_ylabel('Stability Score')
        axes[5].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('zigzag_persistence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to zigzag_persistence_analysis.png")
    
    def compute_persistent_entropy(self, zigzag_barcode):
        """
        Compute persistent entropy based on Atienza et al. (2020).
        Measures information content of persistence diagram.
        """
        if not zigzag_barcode:
            return 0.0
            
        lifetimes = [f['lifetime'] for f in zigzag_barcode if f['lifetime'] > 0]
        
        if not lifetimes:
            return 0.0
            
        total_persistence = sum(lifetimes)
        if total_persistence == 0:
            return 0.0
            
        # Normalize lifetimes to probabilities
        probabilities = [l / total_persistence for l in lifetimes]
        
        # Compute Shannon entropy: H = -sum(p_i * log(p_i))
        entropy = -sum(p * np.log(p + 1e-12) for p in probabilities if p > 1e-12)
        
        return entropy

    def plot_zigzag_barcode(self, zigzag_barcode, optimizer_name):
        """Plot zigzag barcode for a single optimizer."""
        if not zigzag_barcode:
            print(f"  No barcode data to plot for {optimizer_name}")
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group features by dimension
        dim_colors = {0: 'red', 1: 'blue', 2: 'green'}
        dim_labels = {0: 'H0 (Components)', 1: 'H1 (Loops)', 2: 'H2 (Voids)'}
        
        y_pos = 0
        dim_positions = {0: [], 1: [], 2: []}
        
        # Plot each feature as a horizontal bar
        for feature in sorted(zigzag_barcode, key=lambda x: (x['dimension'], x['birth'])):
            dim = feature['dimension']
            birth = feature['birth']
            death = feature['death']
            
            # Plot the bar
            ax.barh(y_pos, death - birth, left=birth, height=0.8, 
                   color=dim_colors[dim], alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            
            dim_positions[dim].append(y_pos)
            y_pos += 1
        
        # Add dimension labels
        ax.set_xlabel('Zigzag Step')
        ax.set_ylabel('Persistent Features')
        ax.set_title(f'Zigzag Persistence Barcode - {optimizer_name}')
        
        # Create legend
        legend_elements = []
        for dim in [0, 1, 2]:
            if dim_positions[dim]:
                legend_elements.append(plt.Rectangle((0,0),1,1, 
                                     facecolor=dim_colors[dim], alpha=0.7,
                                     label=f'{dim_labels[dim]} ({len(dim_positions[dim])} features)'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        # Format plot
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(f['death'] for f in zigzag_barcode) + 1)
        
        # Save individual barcode plot
        filename = f'zigzag_barcode_{optimizer_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved barcode plot: {filename}")
        
    def save_barcode_data(self, all_results):
        """Save detailed barcode data for each optimizer."""
        barcode_data = {}
        
        for optimizer, results in all_results.items():
            barcode_data[optimizer] = {
                'zigzag_barcode': results['zigzag_barcode'],
                'summary': {
                    'total_features': results['total_features'],
                    'h0_features': results['h0_features'],
                    'h1_features': results['h1_features'],
                    'h2_features': results['h2_features'],
                    'avg_lifetime': results['avg_lifetime'],
                    'persistent_entropy': results['persistent_entropy'],
                    'zigzag_stability': results['zigzag_stability']
                }
            }
        
        # Save to file
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        barcode_path = results_dir / "zigzag_barcodes_detailed.pt"
        
        try:
            torch.save(barcode_data, barcode_path)
            print(f"Detailed barcode data saved to {barcode_path}")
            
            # Validate save
            file_size = barcode_path.stat().st_size
            print(f"Barcode file size: {file_size / 1024:.1f} KB")
            
        except Exception as e:
            print(f"Error saving barcode data: {e}")


def run_zigzag_persistence_analysis():
    """Run zigzag persistence analysis using PCA projections."""
    print("ZIGZAG PERSISTENCE ANALYSIS")
    print("=" * 50)
    print("Analyzing optimization with alternating add/remove operations")
    print()
    
    analyzer = ZigzagPersistenceAnalyzer()
    optimizers = ['adam', 'adamw', 'muon', '10p', 'muon10p', 'muonspectralnorm', 'spectralnorm']
    
    start_time = time.time()
    
    # Load projection data
    all_data = analyzer.load_projection_data(optimizers)
    
    if not all_data:
        print("No projection data found")
        return None
    
    # Analyze each optimizer
    all_results = {}
    
    for optimizer in optimizers:
        if optimizer in all_data:
            results = analyzer.analyze_optimizer(all_data[optimizer], optimizer)
            if results['total_features'] > 0:
                all_results[optimizer] = results
                # Plot individual barcode
                analyzer.plot_zigzag_barcode(results['zigzag_barcode'], optimizer)
    
    if not all_results:
        print("No analysis results generated")
        return None
    
    # Create visualizations
    analyzer.create_visualizations(all_results)
    
    total_time = time.time() - start_time
    print(f"\nZIGZAG ANALYSIS COMPLETE: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save results with validation
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "zigzag_persistence_analysis.pt"
    
    try:
        torch.save(all_results, save_path)
        print(f"Results saved to {save_path}")
        
        # Validate save was successful
        file_size = save_path.stat().st_size
        print(f"Saved file size: {file_size / 1024:.1f} KB")
        
        # Test loading to ensure data integrity
        test_load = torch.load(save_path, map_location='cpu')
        print(f"Save validation: Successfully saved {len(test_load)} optimizer results")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return None
    
    # Print rankings
    print_results_summary(all_results)
    
    # Save barcode data
    analyzer.save_barcode_data(all_results)
    
    return all_results


def print_results_summary(all_results):
    """Print summary of zigzag analysis results."""
    print(f"\n{'='*50}")
    print("ZIGZAG PERSISTENCE RESULTS")
    print(f"{'='*50}")
    
    # Feature stability ranking (individual feature stability)
    print("\nFEATURE STABILITY RANKING:")
    stability_scores = {opt: data['avg_stability'] for opt, data in all_results.items()}
    sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (optimizer, score) in enumerate(sorted_stability, 1):
        print(f"  {i}. {optimizer}: {score:.4f}")
    
    # Total features ranking
    print("\nTOTAL FEATURES:")
    feature_counts = {opt: data['total_features'] for opt, data in all_results.items()}
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (optimizer, count) in enumerate(sorted_features, 1):
        h0 = all_results[optimizer]['h0_features']
        h1 = all_results[optimizer]['h1_features'] 
        h2 = all_results[optimizer]['h2_features']
        print(f"  {i}. {optimizer}: {count} features (H0:{h0}, H1:{h1}, H2:{h2})")
    
    # Persistent Entropy ranking (replaces action robustness)
    print("\nPERSISTENT ENTROPY RANKING:")
    entropy_scores = {opt: data['persistent_entropy'] for opt, data in all_results.items()}
    sorted_entropy = sorted(entropy_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (optimizer, score) in enumerate(sorted_entropy, 1):
        print(f"  {i}. {optimizer}: {score:.3f} bits")
    
    # Zigzag Stability ranking (replaces complexity ranking)
    print("\nZIGZAG STABILITY RANKING:")
    zigzag_stability_scores = {opt: data['zigzag_stability'] for opt, data in all_results.items()}
    sorted_zigzag_stability = sorted(zigzag_stability_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (optimizer, score) in enumerate(sorted_zigzag_stability, 1):
        print(f"  {i}. {optimizer}: {score:.3f}")
    
    # Homology dimension analysis
    print(f"\nHOMOLOGY DIMENSION ANALYSIS:")
    print(f"  H0 (connected components) ranking:")
    h0_scores = {opt: data['h0_features'] for opt, data in all_results.items()}
    sorted_h0 = sorted(h0_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (optimizer, count) in enumerate(sorted_h0, 1):
        print(f"    {i}. {optimizer}: {count} components")
    
    print(f"  H1 (1-dimensional holes) ranking:")
    h1_scores = {opt: data['h1_features'] for opt, data in all_results.items()}
    sorted_h1 = sorted(h1_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (optimizer, count) in enumerate(sorted_h1, 1):
        print(f"    {i}. {optimizer}: {count} loops")
        
    print(f"  H2 (2-dimensional voids) ranking:")
    h2_scores = {opt: data['h2_features'] for opt, data in all_results.items()}
    sorted_h2 = sorted(h2_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (optimizer, count) in enumerate(sorted_h2, 1):
        print(f"    {i}. {optimizer}: {count} voids")
    
    print(f"\nKEY FINDINGS:")
    print(f"  - Persistent entropy measures information content of persistence diagrams")
    print(f"  - Zigzag stability measures diagram consistency across perturbations")
    print(f"  - Higher entropy indicates more diverse topological features")
    print(f"  - Higher zigzag stability indicates robust optimization patterns")
    print(f"  - H0 = connected components, H1 = loops, H2 = voids in optimization trajectory")


if __name__ == "__main__":
    run_zigzag_persistence_analysis() 