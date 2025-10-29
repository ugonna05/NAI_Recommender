# Blockchain-Enhanced Hybrid Recommender System

A hybrid recommender system that combines collaborative filtering, content-based filtering, and graph-based features with blockchain-inspired trust mechanisms for enhanced recommendation quality and transparency.

## Features

- **Hybrid Recommendation Approach**: Combines multiple recommendation techniques
- **Graph Neural Networks**: Leverages user-user and asset-asset relationships
- **Blockchain-inspired Trust**: Implements trust scoring and transparency features
- **Cold Start Handling**: Specialized mechanisms for new users and items
- **Flexible Architecture**: Modular design for easy extension

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/blockchain-hybrid-recommender.git
cd blockchain-hybrid-recommender

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.7+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- NetworkX

## Usage

### Basic Usage

```python
from blockchain_recommender import BlockchainHybridRecommender
import pandas as pd

# Load your datasets
interactions_df = pd.read_csv('interactions.csv')
users_df = pd.read_csv('users.csv')
assets_df = pd.read_csv('assets.csv')
asset_graph_df = pd.read_csv('asset_graph.csv')
user_graph_df = pd.read_csv('user_graph.csv')

# Initialize recommender system
recommender = BlockchainHybridRecommender(
    interactions_df, users_df, assets_df, asset_graph_df, user_graph_df
)

# Train the model
recommender.train(epochs=10, batch_size=32)

# Get recommendations for existing user
recommendations = recommender.recommend_top_n("U1", n=5)
print(f"Top recommendations: {recommendations}")

# Handle cold start users
new_user_features = np.random.randn(recommender.user_features_dim)
cold_start_recs = recommender.handle_cold_start_user(new_user_features, n=5)
```

### Data Requirements

The system expects the following CSV files:

- `interactions.csv`: User-item interactions with columns `user_id`, `asset_id`, `rating`
- `users.csv`: User features and metadata
- `assets.csv`: Asset/content features and metadata  
- `asset_graph.csv`: Asset-asset relationships for graph features
- `user_graph.csv`: User-user relationships for social/trust features

## Model Architecture

The hybrid recommender combines:

1. **Neural Collaborative Filtering** for user-item interactions
2. **Content-based Features** from user and asset metadata
3. **Graph Neural Networks** for relationship-based recommendations
4. **Trust Mechanisms** inspired by blockchain consensus

## Configuration

Key parameters can be adjusted in the model initialization:

- Embedding dimensions
- Neural network architecture
- Trust scoring weights
- Graph propagation layers
- Training hyperparameters

## Examples

See the `main()` function in the source code for a complete working example including:

- Data loading and validation
- Model initialization and training
- Recommendation generation
- Cold start scenario handling
- Prediction for specific user-item pairs

## Output

The system provides:
- Top-N recommendations with confidence scores
- Cold start recommendations for new users
- Individual user-item predictions
- Trust scores and explanation features

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{blockchain_recommender_2024,
  title = {Blockchain-Enhanced Hybrid Recommender System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/blockchain-hybrid-recommender}
}
