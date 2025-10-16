

# Smatable

![Gesture Recognition](https://img.shields.io/badge/Swipe--Gesture%20Recognition-Tap--Direction-critical) ![FPGA](https://img.shields.io/badge/FPGA-AMD%20Spartan--7-blue) ![Quantization](https://img.shields.io/badge/Quantization-Integer--Only-green) ![Model](https://img.shields.io/badge/Model-1D--CNN%20%7C%201D--SepCNN-success)


The original **Smatable system** ([Yoshida et al., IEEE Access 2023](https://ieeexplore.ieee.org/document/10360828)) enables swipe gesture recognition on furniture surfaces using piezoelectric vibration sensors and STFT-based 2D-CNNs. While accurate, their implementation depends on a high-end CPU and large memory footprint, limiting real-world applicability on IoT-grade platforms.

This repository presents a **deployment-oriented enhancement** of Smatable that enables **real-time**, **energy-efficient inference** on **edge device (especially AMD Spartan-7 FPGAs**) through model redesign, quantization, and efficient hardware implementation.

---

## 🔍 Project Overview

Our enhancements include:

- 🧠 Using compact **1D-CNNs** and **1D-SepCNNs** operating on raw waveforms
- 🛠️ Implementing **integer-only quantized** and **RTL-synthesizable accelerators** for FPGA deployment.
- 🎯 Enabling **hardware-aware model configuration selection** using Optuna-guided search with application-driven constraints.
- ⚡ Achieving <10 ms inference latency and <1.2 mJ energy per inference on **AMD Spartan-7 (XC7S25)**.
  
---

### Corresponding Paper
**Enabling Vibration-Based Gesture Recognition on Everyday Furniture via Energy-Efficient FPGA Implementation of 1D Convolutional Networks**，which was accepted at IEEE Annual Congress on Artificial Intelligence of Things, Osaka, Japan, Dec 3–5, 2025. 

> **Abstract** The growing demand for smart home interfaces has increased interest in non-intrusive sensing methods like vibration-based gesture recognition. While prior studies demonstrated feasibility, they often rely on complex preprocessing and large Neural Networks (NNs) requiring costly high-performance hardware, resulting in high energy usage and limited real-world deployability.
> This study proposes an energy-efficient solution deploying compact NNs on low-power Field-Programmable Gate Arrays (FPGAs) to enable real-time gesture recognition with competitive accuracy. We adopt a series of optimizations:
(1) We replace complex spectral preprocessing with raw waveform input, eliminating complex on-board preprocessing while reducing input size by 21x without sacrificing accuracy.
(2) We design two lightweight architectures (1D-CNN and 1D-SepCNN) tailored for embedded FPGAs, reducing parameters from 369 million to as few as 216 while maintaining comparable accuracy.
(3) With integer-only quantization and automated RTL generation, we achieve seamless FPGA deployment. A ping-pong buffering mechanism in 1D-SepCNN further improves deployability under tight memory constraints.
(4) We extend a hardware-aware search framework to support constraint-driven model configuration selection, considering accuracy, deployability, latency, and energy consumption.
Evaluated on two swipe-direction datasets with multiple persons and ordinary tables, our approach achieves low-latency, energy-efficient inference on the AMD Spartan-7 XC7S25 FPGA. Under the PS data splitting setting, the selected 6-bit 1D-CNN reaches 0.970 average accuracy across persons with 9.22 ms latency. The chosen 8-bit 1D-SepCNN further reduces latency to 6.83 ms (over 53x CPU speedup) with slightly lower accuracy (0.949). Both consume under 1.2 mJ per inference, demonstrating suitability for long-term edge operation.

If you use the released code, please consider citing our work:

```bibtex
@inproceedings{ling2025smatable,
  title     = {Enabling Vibration-Based Gesture Recognition on Everyday Furniture via Energy-Efficient FPGA Implementation of 1D Convolutional Networks},
  author    = {Shibata, Koki and Ling, Tianheng and Qian, Chao and Matsui,Tomokazu and Suwa, Hirohiko and Yasumoto, Keiichi and Schiele, Gregor},
  booktitle = {Proceedings of the IEEE Annual Congress on Artificial Intelligence of Things (AIoT)},
  year      = {2025},
  location  = {Osaka, Japan},
  note      = {To appear},
  url       = {https://arxiv.org/abs/YYYY}
}
```

---
#### Smatable Dataset

The **Smatable** dataset enables swipe gesture recognition on ordinary furniture surfaces using vibration signals from piezoelectric sensors. Originally collected by [Yoshida et al.](https://ieeexplore.ieee.org/document/10360828), it includes:
- DataByPerson: Different people, same table
- DataByTable: Same person, different tables
- Each session: 4 swipe directions × 10 trials = 40 recordings
- We additionally augmented them to 400 samples/session using sliding-window downsampling

The dataset can be downloaded from Zenodo into the `data/wav/` directory:
👉 [Download link will be provided here]

If you use this dataset, please cite:
```bibtex
@article{yoshida2023smatable,
  title={Smatable: A vibration-based sensing method for making ordinary tables touch-interfaces},
  author={Yoshida, Makoto and Matsui, Tomokazu and Ishiyama, Tokimune and Fujimoto, Manato and Suwa, Hirohiko and Yasumoto, Keiichi},
  journal={IEEE Access},
  volume={11},
  pages={142611--142627},
  year={2023},
  publisher={IEEE}
}
```
---

#### Getting Started
```
# Clone and enter repo
git clone https://github.com/tianheng-ling/smatable
cd smatable

# Set up virtual environment (Python 3.11)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install requirements
pip install -r requirements.txt
```
> ⚠️ This repository works in tandem with our [ElasticAI.Creator](https://github.com/es-ude/elastic-ai.creator/tree/add-linear-quantization) library for VHDL code generation and quantization support. Please install it as part of the setup process.

---

#### Usage
All runnable scripts are organized in the **`scripts/`** folder for convenience: You can **run scripts directly** from their folders.  
For example:

```bash
# ▶️ Experiment 1: Floating-point model comparison
# Train the original 2D-CNN (Yoshida et al.) using STFT features
bash scripts/exp1/train_baseline.sh
# Train our proposed 1D-CNN/1D-SepCNN on raw waveform input
bash scripts/exp1/train.sh

# ▶️ Experiment 2: Deployment-aware optimization on a single participant
# Perform hardware-aware quantization search (4/6/8-bit) using Optuna
bash scripts/exp2/quant.sh

# ▶️ Experiment 3: Cross-participant generalization
# Evaluate best config from Exp2 across all participants
bash scripts/exp3/1DCNN_AOS.sh 

```
---

#### Contributors 

This work is a collaboration between two institutions:

**🇯🇵 NARA Institute of Science and Technology(NAIST), Nara, Japan**
Prof.Dr.Keiichi Yasumoto, Prof.Dr.Tomokazu Matsui, Prof.Dr.Hirohiko Suwa, B.Sc.Koki Shibata, 
- Original Smatable system design
- Sensor data collection and dataset preparation

**🇩🇪 University of Duisburg-Essen(UDE), Duisburg, Germany**
Prof.Dr.Gregor Schiele, M.Sc.Tianheng Ling, M.Sc.Chao Qian, 
- Model redesign, compression and quantization
- FPGA implementation and deployment optimization


---

#### Contact
We welcome feedback and collaboration inquiries. For questions regarding:
- Data collection & application →  📧 [Koki Shibata](koki.shibata@ubi-lab.com)
- Model redesign, compression and quantization  →  📧 [Tianheng Ling](tianheng.ling@uni-due.de)
- Hardware & deployment →  📧 [Chao Qian](chao.qian@uni-due.de)

---
#### Acknowledgements

This work is supported by the German Federal Ministry for Economic Affairs and Climate Action under the RIWWER project (01MD22007C). 

---

#### Related Repositories
Explore other FPGA-deployable time-series models from our UDE intelligent embedded system chair:

- **OnDevice-MLP** → [GitHub Repository](https://github.com/tianheng-ling/OnDeviceSoftSensorMLP)  
- **OnDevice-LSTM** → [GitHub Repository](https://github.com/tianheng-ling/EdgeOverflowForecast)
- **OnDevice-Transformer** → [GitHub Repository](https://github.com/tianheng-ling/TinyTransformer4TS)
- **OnDevice-1D(Sep)CNN** → [GitHub Repository](https://github.com/tianheng-ling/Smatable)
- **OnDevice Running Gait Recognition** → [GitHub Repository](https://github.com/tianheng-ling/StrikeWatch)

