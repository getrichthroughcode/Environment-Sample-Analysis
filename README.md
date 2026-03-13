# Environmental Sample Analysis

Computer vision and generative AI pipeline for environmental sample classification, built on the [WHOI Plankton dataset](https://www.whoi.edu/).

## Product vision

```
[ Upload images ]  ──►  CV pipeline (ViT)  ──►  Results displayed
                                            ──►  Report saved to database

[ Chat zone ]      ──►  Technician asks a question
                   ──►  RAG searches stored reports
                   ──►  Synthetic answer with cited sources
```

A technician uploads sample images → the model classifies species → a report is generated and indexed. They can then query the analysis history in natural language.
