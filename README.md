EEGVision: Reconstructing vision from human brain signals
===
![images](https://github.com/AvancierGuo/EEGVision/blob/main/image/Figure1.png)
Abstract: The intricate mechanisms elucidating the interplay between human visual
perceptions and cognitive processes still remain elusive. Delving into and
reconstructing visual stimuli from cerebral signals could enhance our comprehension
of the processes by which the human brain generates visual imagery. However, the
inherent complexity and significant noise in brain signals limit current efforts to
reconstruct visual stimuli, resulting in low-granularity images that miss details. To
address these challenges, this paper proposes EEGVision, a comprehensive framework
for generating high-quality images directly from brain signals. Leveraging the recent
strides in multi-modal models within the realm of deep learning, it is now feasible to
bridge the gap between EEG data and visual representation. EEGVision initiates this
process with a time-frequency fusion encoder, efficiently extracting cross-domain and
robust features from EEG signals. Subsequently, two parallel pipelines are designed to
align EEG embeddings with image features on both perceptual and semantic levels.
Utilizing a pre-trained image-to-image pipeline from Stable Diffusion, the process
integrates coarse and fine-grained information to recover high-quality images from
EEG data. Both quantitative and qualitative assessments affirm that EEGVision
surpasses contemporary benchmarks. This network architecture holds promise for
further applications in the domain of neuroscience, aiming to unravel the genesis of
human visual perception mechanisms. All code is accessible via
https://github.com/AvancierGuo/EEGVision.
