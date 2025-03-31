# HumMorph: Generalized Dynamic Human Neural Fields from Few Views
We propose **HumMorph** -- a novel generalized approach to free-viewpoint rendering of dynamic human bodies with explicit pose control. HumMorph renders a human actor in any specified pose given a few observed views (starting from just one) in arbitrary poses.
The observations can, for example, be extracted from a monocular video where each view captures the actor in a different pose.
Our method enables fast inference due to relying only on feed-forward passes through the model.
We first construct a coarse representation of the actor in the canonical T-pose, which combines visual features from individual partial observations and fills missing information using learned prior knowledge.
The coarse representation is complemented by fine-grained pixel-aligned features extracted directly from the observed views, which provide high-resolution appearance information.

We demonstrate that HumMorph is competitive with the state-of-the-art when only a single input view is available, however, we achieve results with significantly better visual quality given just 2 monocular observations. 
Moreover, previous generalized approaches to this problem were evaluated assuming access to accurate body shape and pose parameters estimated using synchronized multi-camera setups.
In contrast, we consider a more practical scenario where these body parameters are noisily estimated directly from the observed views. Our experimental results demonstrate that our architecture is more robust to errors in the noisy parameters and clearly outperforms the state-of-the-art in this setting.
