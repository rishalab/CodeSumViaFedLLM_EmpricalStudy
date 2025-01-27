
**Collaborative Code Summarization Via FedLLM -An Empirical Study**

We empirically extend our previous study "Code Summarization without Direct Access to Code - Towards Exploring Federated LLMs for Software Engineering" published in the 28th International Conference on Evaluation and Assessment in Software Engineering (EASE 2024).

We perform below set of experiments for two federated aggregation techniques: existing FedIT and our own implementation of FedAvg, **FedDecomp**. Our previous conference paper had 1st experiment with FedIT, which has been extended for other set of experiments in this study:

| # | Language | Model     | PEFT  |
| - | -------- | --------- | ----- |
| 1 | Python   | LLaMA2    | LoRA  |
| 2 | Python   | LLaMA2    | QLoRA |
| 3 | Python   | CodeLLaMA | LoRA  |
| 4 | Python   | CodeLLaMA | QLoRA |
| 5 | Java     | LLaMA2    | LoRA  |
| 6 | Java     | LLaMA2    | QLoRA |
| 7 | Java     | CodeLLaMA | LoRA  |
| 8 | Java     | CodeLLaMA | QLoRA |

**[Datasets]()** Processed Python and Java datasets used for the study are made available. Please cite the original dataset source paper during replication.

**[Code]()** Java code for fine-tuning CodeLLaMA with QLoRA technique is made available.

* Central, Fed0, FedITRounds, and FedDecompRounds represent the python scripts (training and evaluation) for centralized training, vanilla (LoRA initialisation + metric evaluation), FL with FedIT, FL with FedDecomp.
* Indi, anecdotal scripts perform indivdiual client model's evaluation and anecdotal generation for the PreTrained, Non-Fed, and FedBEST.

By refering to our previous (LoRA + LLaMA2 + FedIT) and current study codes, all the combination of experiements' code can be inferred easily.

**[Results]()** has Metric results and anecdotal csvs:

* Metrics: For all combination of experiments, it has the one respective csv, which has vanilla, Non-Fed (Pre-Trained), Fed 1-20 rounds metric results. Additionally, each experiment's individual-client model results are also available in it's pairing csv file.
* Anecdotals: <input, generated output, ground truth> triplet for all the 3 models: PreTrained, Non-Fed, and FedBEST (BEST round selected from all 20 rounds). Each experiments csvs are in the respective folder#.

**[Models]()** has the Java best model's Non-Fed, vanilla round's, Fed rounds adapters for usage. Our previous study has the Python model.

**[For replication]()**, please place the dataset in the current directory, and run the python scripts for respective model training or evaluation.
