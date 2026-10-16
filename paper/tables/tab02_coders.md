**Table 2. Coders, confidence signals and realized API cost**

| signal | model | kind | settings | sample | USD / 1k segments | with batch API |
|---|---|---|---|---|---|---|
| Jev | TypeSafe jev-1.13.0 | decision model | Choice (exclusive) / Noul per code (multi) | full | 0.047 | – |
| OpenAI Decisions | OpenAI Decisions API, gpt-6-luna | decision model | Jev's requests mapped to choice / predicate | full | 0.102 | – |
| Luna verbal | OpenAI gpt-6-luna | LLM | reasoning effort none; verbal 0–100 | full | 0.107 | 0.053 |
| Luna logprob | OpenAI gpt-6-luna | LLM | answer-token logprobs, same call | full | 0.107 | 0.053 |
| Luna self-consistency | OpenAI gpt-6-luna | LLM | 5 samples at T = 1, vote share | 350/task | 0.283 | 0.142 |
| Haiku verbal | Anthropic claude-haiku-4-5 | LLM | T = 0; verbal | full | 1.906 | 0.953 |
| gpt-oss verbal | openai/gpt-oss-120b (Together) | LLM | reasoning low; verbal | full | 0.232 | – |
| gpt-oss logprob | openai/gpt-oss-120b (Together) | LLM | answer-token logprobs | full | 0.232 | – |
| Qwen3.5-9B verbal | Qwen/Qwen3.5-9B (Together) | LLM | thinking off; verbal | full | 0.176 | – |
| Qwen3.5-9B logprob | Qwen/Qwen3.5-9B (Together) | LLM | thinking off; logprobs | full | 0.176 | – |
| 3-LLM agreement | Luna + Haiku + gpt-oss | ensemble | vote share of three verbal runs | full | 2.245 | – |
| Sonnet verbal | Anthropic claude-sonnet-5-5 | LLM (frontier) | thinking off, effort low; verbal | full | 1.964 | 0.982 |
| Qwen3.8-2.4T verbal | Qwen/Qwen3.8-2.4T-A95B (Together) | LLM (frontier) | thinking off; verbal | 270/task | 2.481 | – |
| Qwen3.8-2.4T logprob | Qwen/Qwen3.8-2.4T-A95B (Together) | LLM (frontier) | thinking off; logprobs | 270/task | 2.481 | – |
| Clef (27B) | Cloudflare @cf/cloudflare/clef (27B) | decision model | Jev's exact requests | full | 0.213 | – |
| Clef-flash (9B) | Cloudflare @cf/cloudflare/clef-flash (9B) | decision model | Jev's exact requests | full | 0.080 | – |
| pplx-decider | Perplexity pplx-decider-v1.1-27b | decision model | Jev's exact requests | full | 0.019 | – |
| Kev-9B | jaredpalmer/kev-9b (Modal, H100, bf16) | decision model (open) | Jev's exact requests | full | 0.073 | – |
| Kev-27B | jaredpalmer/kev-27b (Modal, B200, bf16) | decision model (open) | Jev's exact requests | full | 0.315 | – |

_Cost: realized ledger cost of the canonical calls (list prices incl. prompt-cache discounts), mean over the five confirmatory tasks. Batch column assumes the 50% batch discount (OpenAI Batch, Anthropic Message Batches); '–' = no batch tier used or available._
