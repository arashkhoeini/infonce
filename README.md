# InfoNCE Loss - PyTorch Implementation

The **InfoNCE loss** (Information Noise-Contrastive Estimation) is commonly used in contrastive learning to maximize the similarity between positive pairs while minimizing it between negative pairs. This repository provides a PyTorch implementation supporting both unsupervised and supervised modes.

## Classes

- **InfoNCE**: Positive pairs are generated by using two independent augmentations of the same batch.
- **SupervisedInfoNCE**: Positive pairs are defined using class labels, ensuring that negative pairs are not from the same class.

## Installation

To install the package , run:

```bash
pip install infonce
```

## Usage

### Unsupervised InfoNCE

```python
from infonce import InfoNCELoss

# Example usage
loss_fn = InfoNCELoss(temperature=0.07)
# augmentation1 and augmentation2 are two different augmentations of the same batch
features1 = model(augmentations1) 
features2 = model(augmentations2) 
features = torch.stack([features1, features2], dim=0)
loss = loss_fn(features)
```

### Supervised InfoNCE

```python
from infonce import SupervisedInfoNCELoss

# Example usage
loss_fn = SupervisedInfoNCELoss(temperature=0.07)
loss = loss_fn(features, labels)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

This implementation is inspired by the original InfoNCE loss paper and various open-source implementations.
