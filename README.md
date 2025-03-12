# Exoplanet Atmosphere Classification with Neural Networks

This project explores how I can create and use artificial (synthetic) data to train AI models for astronomy. I focus on a specific case: teaching a neural network to identify different types of exoplanet atmospheres from their spectral signatures.

## 🌍 Project Overview

I started with a simple question: can I create realistic artificial data to train AI models when real astronomical data is limited? <br> <br> My first attempt revealed an interesting problem: the neural network was too good, achieving 100% accuracy. <br> This meant **my synthetic data was too perfect** and didn't capture the complexity of real astronomical observations.

To make the data more realistic, I added several real-world complications:

- **Mixed atmospheres** (30% chance of planets having mixed atmospheric features)
- **Various noise types** that real telescopes encounter
- **Earth's atmosphere interference** (like when observing through our atmosphere)
- **Temperature changes** (as planets orbit their stars)
- **Instrument effects** (how telescopes and sensors affect measurements)

## 📊 Dataset Details

The current dataset includes:

- **Size**: 100,000 synthetic spectra
- **Resolution**: 200 wavelength points (0.3-2.5 μm)
- **Classes**: 5 exoplanet types with realistic distribution:
  - **Earth-like** (30%): O₂ and H₂O absorption bands
  - **Venus-like** (25%): Dense CO₂ and sulfuric acid features
  - **Hot Jupiter** (20%): Na and K absorption features
  - **Mini-Neptune** (15%): H₂O and H₂ dominated
  - **Exotic** (10%): CH₄ and NH₃ signatures

## 🧠 Neural Network Architecture

The model uses a feedforward neural network with:

- **Input layer**: 200 neurons (spectral features)
- **Hidden layers**: 128 → 64 → 32 neurons with ReLU
- **Regularization**: Dropout (0.3) and BatchNorm
- **Output**: 5 classes with softmax
- **Training**: AdamW optimizer with learning rate scheduling

## 📈 Results and Challenges

- Initial model achieved 100% accuracy (problematic)
- After complexity improvements: ~98-99% accuracy
- Current challenges:
  - Balancing realism with learning difficulty
  - Simulating complex atmospheric interactions
  - Reproducing realistic noise patterns

## 🔮 Future Directions

1. **Data Enhancement**:
   - Further increase atmospheric mixing complexity
   - Add more realistic cloud coverage effects
   - Implement more sophisticated instrumental artifacts

2. **Real Data Integration**:
   - Use real astronomical datasets as templates
   - Create hybrid synthetic-real data approaches
   - Develop methods to enhance limited real datasets

3. **Model Improvements**:
   - Explore alternative architectures (CNNs, Transformers)
   - Implement uncertainty quantification
   - Add physical constraints to predictions

## 🎯 Project Goals

This project serves as a proof-of-concept for synthetic data generation in astronomical applications. While starting from scratch provided valuable insights, my future work will focus on:

1. Using real astronomical datasets as templates
2. Developing methods to augment limited observational data
3. Creating more physically accurate synthetic spectra

## ⚙️ Usage Instructions

### Install Dependencies
```bash
pip install torch numpy matplotlib scipy
```

### Generate Dataset
```python
python dataset_creation.py
```

### Train Model
```python
python exoplanet_nn.py
```

## 📚 References

- Exoplanet spectroscopy techniques
- Atmospheric physics models
- Machine learning in astronomy
- sunset: A database of synthetic atmospheric-escape transmission spectra https://arxiv.org/abs/2410.03228
- Parameterizing pressure-temperature profiles of exoplanet atmospheres with neural networks https://arxiv.org/abs/2309.03075

## 📜 License

This project is licensed under the MIT License.

