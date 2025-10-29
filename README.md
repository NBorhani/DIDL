# DIDL: A deep learning approach to predict inter-omics interactions in multi-layer networks


This repository contains the Python code and dataset for the paper titled **“A deep learning approach to predict inter-omics interactions in multi-layer networks”** by Niloofar Borhani, Jafar Ghaisari, Maryam Abedi, Marzieh Kamali and Yousof Gheisari.
Data Integration with Deep Learning (DIDL) is a nonlinear deep learning framework designed to predict inter-omics interactions. It combines automatic feature extraction and interaction prediction, achieving state-of-the-art performance across diverse biological networks such as drug–target, TF–DNA, and miRNA–mRNA interactions.
The paper is available at [https://doi.org/10.1186/s12859-022-04569-2](https://doi.org/10.1186/s12859-022-04569-2).

## Training and Evaluation with k-Fold Cross Validation
To train the DIDL model and validate it using k-fold cross-validation, run the following command:

```bash
python main.py \
  --data_name dataset/miRNAmRNA.csv \
  --negative_sampling 1.0 \
  --mir_layer [64,32,20] \
  --prot_layer [64,32,20] \
  --dropout 0.5 \
  --reg_L2 0.08 \
  --batch_size 32 \
  --n_fold 10 \
  --learning_rate 1e-5 \
  --epochs 20 \
  --n_epochs_stop 5
```

## Contact Information
For further inquiries, please contact.

**Niloofar Borhani**  
Ph.D. Student, Control Engineering  
Isfahan University of Technology  
Email: [n.borhani@ec.iut.ac.ir](mailto:n.borhani@ec.iut.ac.ir)  
CV: [Google Scholar](https://scholar.google.com/citations?user=SSD_k8MAAAAJ&hl=en)
