# AndroGen – Synthetic Sperm Image Generator
 
✅ The codebase is complete and fully functional.  
📄 Deployment documentation is currently under preparation and will be available in the coming days.
 
Thank you for your interest in **AndroGen**. Stay tuned for detailed setup instructions and usage examples.

## ✨ Key Features

- 📷 Synthetic generation of microscopic sperm images for multiple animal species.
- 🔧 Intuitive GUI with preloaded dataset configurations (SVIA, VISEM, BOSS).
- ⚙️ Full customization: morphology, concentration, movement, background, and debris.
- 🖥️ Compatible with both local and cloud environments via Gradio.
- 📂 Exports datasets with images, segmentation masks, and annotations (PNG, TXT).
- 🧪 Quantitatively and qualitatively validated using FID/KID metrics.
- 🧬 Modular and extensible: easily add new species, environments, motion patterns, and artifacts.

## 📦 Installation

```bash
git clone https://github.com/dhernandezgit/AndroGen.git
cd AndroGen
./launch.sh
```

> Requires Python 3.10.

## 🚀 Usage

Run the GUI locally:

```bash
python run.py
```

From the GUI, you can:

1. Load predefined configurations based on SVIA, VISEM, or BOSS datasets.
2. Adjust visual parameters: background, brightness, contrast, blur, color, depth distribution.
3. Define morphology: species, sperm classes, dimensions, mobility, and more.
4. Generate customized images or sequences and save them to your target directory.

## 📈 Validation

AndroGen has been validated by replicating SVIA, VISEM, and BOSS datasets. Synthetic images were evaluated with:

- **FID (Frechet Inception Distance)**
- **KID (Kernel Inception Distance)**

Results show strong similarity to real datasets at low computational cost and without real data.


| SVIA Real | SVIA Synthetic | VISEM Real | VISEM Synthetic | BOSS Real | BOSS Synthetic |
|-----------|----------------|------------|------------------|-----------|----------------|
| ![svia-real](examples/svia_real.png) | ![svia-syn](examples/svia_syn.png) | ![visem-real](examples/visem_real.png) | ![visem-syn](examples/visem_syn.png) | ![boss-real](examples/boss_real.png) | ![boss-syn](examples/boss_syn.png) |


# License
**AGPL-3.0 License**: This open-source license is ideal for students and researchers, promoting open collaboration and knowledge sharing. See the LICENSE file for more details.

**Enterprise License**: Designed for commercial use, this license permits using this software for commercial solutions, bypassing the open-source requirements of AGPL-3.0. Please contact us for more details.

## 👥 Authors

- Daniel Hernández-Ferrándiz  - `daniel.hernandezf@urjc.es`
- Juan J. Pantrigo -`juanjose.pantrigo@urjc.es`
- Soto Montalvo  - `soto.montalvo@urjc.es`
- Raúl Cabido  - `raul.cabido@urjc.es`

Universidad Rey Juan Carlos, Móstoles, Spain
