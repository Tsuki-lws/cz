# Track A Data Sources

Initial candidate mix:

- 2WikiMultihopQA train, HotpotQA, MuSiQue for multihop text reasoning.
- OK-VQA, A-OKVQA, InfoSeek for visual knowledge QA.
- Natural Questions, TriviaQA, BeerQA for open-domain retrieval QA.
- WebQA and a small amount of FRAMES for tool/evidence aggregation.

Benchmark data must never be used for parameter updates.

Teacher / training models used for distillation must be no larger than 32B.
