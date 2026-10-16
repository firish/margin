**Table 18. Model snapshot: what was called, which version answered, and when**

| coder | provider | model requested | version returned | calls (all arms) | first call (UTC) | last call (UTC) |
|---|---|---|---|---|---|---|
| Jev | typesafe | jev-1.13.0 | jev-1.13.0 | 65995 | 2026-10-10 | 2026-10-11 |
| OpenAI Decisions | openai decisions | gpt-6-luna | gpt-6-luna | 65993 | 2026-10-11 | 2026-10-11 |
| Clef | cloudflare | @cf/cloudflare/clef | clef | 14617 | 2026-10-11 | 2026-10-11 |
| Clef-flash | cloudflare | @cf/cloudflare/clef-flash | clef-flash | 65993 | 2026-10-11 | 2026-10-11 |
| pplx-decider | perplexity | pplx-decider-v1.1-27b | pplx-decider-v1.1-27b | 65993 | 2026-10-11 | 2026-10-11 |
| Kev-9B | kev | jaredpalmer/kev-9b | kev-latest | 51368 | 2026-10-11 | 2026-10-11 |
| Kev-27B | kev | jaredpalmer/kev-27b | kev-latest | 14117 | 2026-10-11 | 2026-10-11 |
| Luna | openai | gpt-6-luna | gpt-6-luna | 13905 | 2026-10-10 | 2026-10-11 |
| Haiku | anthropic | claude-haiku-4-5 | claude-haiku-4-5-20251001 | 11280 | 2026-10-10 | 2026-10-12 |
| Sonnet | anthropic | claude-sonnet-5-5 | claude-sonnet-5-5 | 12437 | 2026-10-10 | 2026-10-12 |
| gpt-oss-120b | together | openai/gpt-oss-120b | openai/gpt-oss-120b | 10030 | 2026-10-10 | 2026-10-11 |
| Qwen3.5-9B | together | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B | 9930 | 2026-10-11 | 2026-10-11 |
| Qwen3.8-2.4T | together | Qwen/Qwen3.8-2.4T-A95B | Qwen/Qwen3.8-2.4T-A95B | 3281 | 2026-10-11 | 2026-10-11 |

_Several APIs were in public beta (OpenAI Decisions from 2026-10-06; decision models launched September–October 2026). Kev weights: Hugging Face revisions db029f0 (kev-9b) and af0e6d5 (kev-27b), the latest on the call dates; kev package commit fc4a17e. Results describe these versions on these dates; the released harness allows re-running against later versions._
