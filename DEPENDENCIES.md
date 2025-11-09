# Dependency Compatibility Notes

## PyTorch and Transformers Version Compatibility

### Issue
As of November 2025, there is a compatibility issue between PyTorch and Transformers versions:

- **Transformers >= 4.56.0** requires **PyTorch >= 2.6.0** due to a security vulnerability fix (CVE-2025-32434)
- **PyTorch 2.6.0+** is not yet available for CUDA 12.1 via pip

### Current Solution
The project uses the following compatible versions:
- **PyTorch 2.5.1** (latest available for CUDA 12.1)
- **Transformers 4.45.0** (compatible with PyTorch 2.5.1)

### requirements.txt Configuration
```
transformers>=4.45.0,<4.56.0  # Prevents automatic upgrade to incompatible versions
torch>=2.5.0,<2.6.0           # Ensures PyTorch 2.5.x is used
```

### Future Updates
When PyTorch 2.6.0+ becomes available for CUDA:
1. Update PyTorch: `pip install --upgrade torch torchvision torchaudio`
2. Update Transformers: `pip install --upgrade transformers`
3. Update requirements.txt to remove version constraints

### Installation
To install dependencies with correct versions:
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Troubleshooting
If you encounter the error:
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, 
we now require users to upgrade torch to at least v2.6 in order to use the function.
```

**Solution:**
```bash
pip install "transformers>=4.45.0,<4.56.0" --force-reinstall
```

### Related Issues
- CVE-2025-32434: PyTorch `torch.load` vulnerability
- GitHub Issue: transformers requiring PyTorch 2.6.0+

### Last Updated
November 3, 2025
