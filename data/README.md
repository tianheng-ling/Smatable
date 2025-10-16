# Smatable Dataset

The **Smatable dataset** is designed for swipe gesture recognition on ordinary surfaces using vibration signals. It was originally collected and published by Yoshida et al. (2023) as part of their study on enabling touch interfaces on wooden tables using piezoelectric sensors.

This dataset includes two subsets:

- **DataByPerson**: Collected from different participants performing swipe gestures on the same table.  
- **DataByTable**: Collected on different tables, with swipe gestures performed by the same participant.  

Each session contains 40 multichannel vibration waveform recordings (4 directions × 10 repetitions).  
To enrich the dataset, we apply **sliding-window augmentation with fractional offsets during downsampling**, which expands each session to **400 samples**.

---

####  Data Splits

We adopt the three evaluation protocols from the original paper:

1. **Per-Subject (PS)**  
   Each target (person/table) is split independently: 6 sessions for training, 3 for testing.

2. **Leave-One-Subject-Out (LOSO)**  
   Two targets are used for training, and the third is held out for testing, simulating cross-user or cross-surface generalization.

3. **Add-One-Session (AOS)**  
   One session from the held-out target is included in training, reflecting few-shot personalization.

---

####  📦 Download

The dataset can be downloaded from **Zenodo** into the `data/wav/` directory:  
👉 _[Download link will be provided here]_  

---

####  📄 Citation

If you use this dataset, please cite the following paper:

```bibtex
@article{yoshida2023smatable,
  title     = {Smatable: A vibration-based sensing method for making ordinary tables touch-interfaces},
  author    = {Yoshida, Makoto and Matsui, Tomokazu and Ishiyama, Tokimune and Fujimoto, Manato and Suwa, Hirohiko and Yasumoto, Keiichi},
  journal   = {IEEE Access},
  volume    = {11},
  pages     = {142611--142627},
  year      = {2023},
  publisher = {IEEE}
}
```