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
    
    def __init__(self, max_dimension=1, max_edge_length=2.0):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.projection_engine = ProjectionEngine()
        
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
    
    def create_zigzag_sequence(self, trajectories, max_points=30):
        """Create zigzag sequence: add points, remove some, add back."""
        n_points = min(len(trajectories), max_points)
        
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
        
        for i in missing_indices[:5]:
            current_indices.append(i)
            point_sets.append(current_indices.copy())
            actions.append('readd')
            
        return point_sets, actions
    
    def compute_persistence_for_points(self, points):
        """Compute persistence for a set of points."""
        if len(points) < 3:
            return {0: [], 1: []}
            
        try:
            rips_complex = gudhi.RipsComplex(points=points, max_edge_length=self.max_edge_length)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
            persistence = simplex_tree.persistence()
            
            persistence_by_dim = {0: [], 1: []}
            
            for dimension, (birth, death) in persistence:
                if dimension <= self.max_dimension:
                    if death != float('inf'):
                        persistence_by_dim[dimension].append((birth, death))
                    else:
                        persistence_by_dim[dimension].append((birth, np.inf))
                        
            return persistence_by_dim
            
        except Exception:
            return {0: [], 1: []}
    
    def track_features(self, point_sets, actions, trajectories):
        """Track topological features through zigzag sequence."""
        feature_tracker = {}
        next_feature_id = 0
        zigzag_barcode = []
        
        for step, (indices, action) in enumerate(zip(point_sets, actions)):
            if len(indices) < 3:
                continue
                
            points = trajectories[indices]
            persistence_data = self.compute_persistence_for_points(points)
            
            current_features = set()
            for dim in [0, 1]:
                for birth, death in persistence_data[dim]:
                    feature_sig = f"dim{dim}_b{birth:.3f}_d{death:.3f}"
                    current_features.add((feature_sig, dim, birth, death))
            
            # Track new features
            for feature_sig, dim, birth, death in current_features:
                if feature_sig not in feature_tracker:
                    feature_tracker[feature_sig] = {
                        'birth_step': step,
                        'last_seen': step,
                        'dimension': dim,
                        'actions': [action],
                        'feature_id': next_feature_id
                    }
                    next_feature_id += 1
                else:
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
                'actions': feature_data['actions']
            })
            
        return zigzag_barcode
    
    def analyze_optimizer(self, optimizer_data, optimizer_name):
        """Analyze single optimizer with zigzag persistence."""
        print(f"Analyzing {optimizer_name}...")
        
        trajectories = optimizer_data['trajectories']
        
        # Create zigzag sequence
        point_sets, actions = self.create_zigzag_sequence(trajectories)
        
        # Track features
        zigzag_barcode = self.track_features(point_sets, actions, trajectories)
        
        # Compute metrics
        if not zigzag_barcode:
            return {
                'total_features': 0,
                'avg_lifetime': 0.0,
                'avg_stability': 0.0,
                'h0_features': 0,
                'h1_features': 0,
                'persistent_entropy': 0.0,
                'zigzag_stability': 0.0,
                'zigzag_barcode': []
            }
        
        lifetimes = [f['lifetime'] for f in zigzag_barcode]
        stabilities = [f['stability'] for f in zigzag_barcode]
        
        h0_count = sum(1 for f in zigzag_barcode if f['dimension'] == 0)
        h1_count = sum(1 for f in zigzag_barcode if f['dimension'] == 1)
        
        # Compute research-based metrics
        persistent_entropy = self.compute_persistent_entropy(zigzag_barcode)
        zigzag_stability = self.compute_zigzag_stability(point_sets, actions, trajectories)
        
        results = {
            'total_features': len(zigzag_barcode),
            'avg_lifetime': np.mean(lifetimes),
            'avg_stability': np.mean(stabilities),
            'h0_features': h0_count,
            'h1_features': h1_count,
            'persistent_entropy': persistent_entropy,
            'zigzag_stability': zigzag_stability,
            'zigzag_barcode': zigzag_barcode
        }
        
        print(f"  Features: {results['total_features']}, "
              f"Avg lifetime: {results['avg_lifetime']:.2f}, "
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
        
        # 4. Dimension distribution
        h0_counts = [all_results[opt]['h0_features'] for opt in optimizers]
        h1_counts = [all_results[opt]['h1_features'] for opt in optimizers]
        
        x_pos = np.arange(len(optimizers))
        width = 0.35
        
        axes[3].bar(x_pos - width/2, h0_counts, width, label='H0', alpha=0.7)
        axes[3].bar(x_pos + width/2, h1_counts, width, label='H1', alpha=0.7)
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
    
    def compute_zigzag_stability(self, point_sets, actions, trajectories):
        """
        Compute zigzag stability based on Oudot & Sheehy (2015).
        Measures how persistence diagrams change across zigzag sequence.
        """
        if len(point_sets) < 2:
            return 0.0
            
        persistence_diagrams = []
        
        # Compute persistence diagram for each step
        for indices in point_sets:
            if len(indices) < 3:
                persistence_diagrams.append([])
                continue
                
            points = trajectories[indices]
            persistence_data = self.compute_persistence_for_points(points)
            
            # Flatten to list of (birth, death) pairs
            diagram = []
            for dim in [0, 1]:
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
        print(f"  {i}. {optimizer}: {count} features")
    
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
    
    print(f"\nKEY FINDINGS:")
    print(f"  - Persistent entropy measures information content of persistence diagrams")
    print(f"  - Zigzag stability measures diagram consistency across perturbations")
    print(f"  - Higher entropy indicates more diverse topological features")
    print(f"  - Higher zigzag stability indicates robust optimization patterns")


if __name__ == "__main__":
    run_zigzag_persistence_analysis() 