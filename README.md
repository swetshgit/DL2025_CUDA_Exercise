# Deep_Learning_Project
CS7643 Deep Learning Project Collaboration

Please refer to the jupyter notebooks:
1. GRU (at root)
2. PCA (at root)
3. LSTM folder under models

## Private notebook cells

The CNN-GRU and LSTM notebooks use a required Git clean filter. The working
copies retain all cells, while Git stages only the training/testing sections
and their results. Configure the filter once after cloning:

```powershell
git config --local filter.hide-model-code.clean "python sanitize_notebook.py"
git config --local filter.hide-model-code.required true
```
