# 🚀 ASQA RAG - Step-by-Step Execution Guide

**Current Status**: ✅ Step 1 Complete - Data prepared!

---

## 📍 **Where Are All The Files?**

All notebooks are in the `notebooks/` folder:

```
notebooks/
├── asqa_data_preparation.ipynb       ← ✅ YOU JUST RAN THIS
├── rag-asqa-baseline.ipynb           ← 🎯 RUN THIS NEXT
├── synthetic_data_generation.ipynb   ← Run after rag-asqa-baseline
└── evaluation_analysis.ipynb         ← Run last
```

---

## 🎯 **Step 2: Train RAG Models (NEXT STEP)**

### **Open the notebook:**

```bash
cd "c:\Users\Admin\OneDrive - VNU-HCMUS\Documents\GitHub\ragas-evaluation"
jupyter notebook notebooks/rag-asqa-baseline.ipynb
```

### **What this notebook does:**

| Section | What It Does | Time | Can Run Overnight? |
|---------|--------------|------|-------------------|
| 1-4 | Setup, load data, configure | 5 min | No |
| 5 | **Train Retriever** | 4-6 hours | ✅ Yes |
| 6 | Evaluate Retriever | 30 min | No |
| 7 | Build FAISS Index | 30 min | No |
| 8 | **Train Generator** | 6-8 hours | ✅ Yes |
| 9 | **Normal RAG Inference** | 2-3 hours | ✅ Yes |
| 10-11 | **RAG + Filter Inference** | 3-4 hours | ✅ Yes |

### **How to run:**

1. **Run Cells 1-4** (setup) - Takes 5 minutes
2. **Run Cell 5** (retriever training) - **START THIS AND GO TO SLEEP** 💤
3. Next morning: Run Cells 6-7 (evaluate + index)
4. **Run Cell 8** (generator training) - **START THIS AND GO TO SLEEP** 💤
5. Next morning: Run Cells 9-11 (inference) - **START THIS AND GO TO WORK** 💼

**Total: 3 days** (mostly waiting for training)

---

## 🎯 **Step 3: Generate Synthetic Data**

### **Open the notebook:**

```bash
jupyter notebook notebooks/synthetic_data_generation.ipynb
```

### **What this notebook does:**

| Section | What It Does | Time |
|---------|--------------|------|
| 1-4 | Setup, load predictions, define judge | 10 min |
| 5 | **Generate labels with LLM** | 3-4 hours |
| 6-7 | Filter and balance data | 30 min |
| 8-10 | Save and visualize | 10 min |

### **How to run:**

1. Run Cells 1-4 (setup)
2. **Run Cell 5** (labeling) - **START AND GO TO LUNCH** 🍕
3. Run Cells 6-10 (finish up)

**Total: 4-5 hours**

---

## 🎯 **Step 4: Comprehensive Evaluation**

### **Open the notebook:**

```bash
jupyter notebook notebooks/evaluation_analysis.ipynb
```

### **What this notebook does:**

| Section | What It Does | Time |
|---------|--------------|------|
| 1-2 | Setup, load results | 5 min |
| 3 | Filter effectiveness metrics | 10 min |
| 4 | **RAGAS evaluation** | 2-3 hours |
| 5 | ROUGE-L and BERTScore | 1 hour |
| 6-9 | Analysis, visualizations, reports | 30 min |

### **How to run:**

1. Run Cells 1-3 (quick analysis)
2. **Run Cell 4** (RAGAS) - **START AND TAKE A BREAK** ☕
3. Run Cells 5-9 (finish up)

**Total: 4-5 hours**

---

## 📅 **Recommended Schedule**

### **Day 1 (Monday Morning)**
- ✅ Run `asqa_data_preparation.ipynb` - DONE!
- 🎯 Open `rag-asqa-baseline.ipynb`
- Run Cells 1-4 (setup)
- **Start Cell 5 (retriever training) at 5 PM**
- Go home, let it run overnight

### **Day 2 (Tuesday Morning)**
- Check retriever training results
- Run Cells 6-7 (evaluate + index)
- **Start Cell 8 (generator training) at 5 PM**
- Go home, let it run overnight

### **Day 3 (Wednesday Morning)**
- Check generator training results
- **Start Cells 9-11 (inference) at 9 AM**
- Go to lunch/meetings while it runs (5-7 hours)
- By afternoon: inference complete!

### **Day 4 (Thursday)**
- Open `synthetic_data_generation.ipynb`
- **Start Cell 5 (labeling) at 9 AM**
- Go to lunch (3-4 hours)
- Finish Cells 6-10

### **Day 5 (Friday)**
- Open `evaluation_analysis.ipynb`
- **Start Cell 4 (RAGAS) at 9 AM**
- Go to lunch (2-3 hours)
- Finish Cells 5-9
- **DONE!** 🎉

---

## 🔧 **Quick Troubleshooting**

### **If you get import errors:**

```bash
# Reinstall dependencies
pip install --upgrade transformers torch
pip install -r requirements.txt
```

### **If notebooks don't show up:**

```bash
# List notebooks
dir notebooks\*.ipynb

# Should show:
# asqa_data_preparation.ipynb
# rag-asqa-baseline.ipynb
# synthetic_data_generation.ipynb
# evaluation_analysis.ipynb
```

### **If you need to restart:**

All notebooks save checkpoints! You can resume from where you left off.

---

## ✅ **Current Status**

- ✅ **Step 1 Complete**: Data prepared in `data/asqa/`
- 🎯 **Next**: Open `notebooks/rag-asqa-baseline.ipynb` and start training!

---

## 📞 **Need Help?**

Check these files:
- `ASQA_IMPLEMENTATION_GUIDE.md` - Detailed technical guide
- `IMPLEMENTATION_SUMMARY.md` - Overview and results
- This file - Quick execution guide

**Ready to continue?** Open `rag-asqa-baseline.ipynb` now! 🚀
