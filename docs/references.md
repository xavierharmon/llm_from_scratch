# References and Further Reading

## Foundational Papers

### Must-read (in this order)

1. **Attention Is All You Need** (Vaswani et al., 2017)
   The original transformer paper. Read after completing Phase 4.
   https://arxiv.org/abs/1706.03762

2. **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
   GPT-2 paper. Introduces the decoder-only architecture we implement.
   https://openai.com/research/gpt-2-1-5b-released

3. **Training Compute-Optimal LLMs** (Hoffmann et al., 2022)
   Chinchilla paper. The definitive guide to scaling laws.
   https://arxiv.org/abs/2203.15556

4. **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
   The fine-tuning method implemented in Phase 9.
   https://arxiv.org/abs/2106.09685

### Deeper reading

5. **An Image is Worth 16x16 Words** (Dosovitskiy et al., 2020)
   Transformers applied to vision — shows the architecture's generality.
   https://arxiv.org/abs/2010.11929

6. **Transformer Feed-Forward Layers Are Key-Value Memories** (Geva et al., 2021)
   Explains what FFN layers are actually doing.
   https://arxiv.org/abs/2012.14913

7. **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019)
   RMSNorm — simpler than LayerNorm, used in LLaMA.
   https://arxiv.org/abs/1910.07467

8. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
   Modern best practices: RoPE, SwiGLU, RMSNorm, GQA.
   https://arxiv.org/abs/2302.13971

## Best Video Resources

- **Andrej Karpathy — "Let's build GPT"** (YouTube, 2023)
  The single best complement to this project. Watch alongside Phase 5.
  https://www.youtube.com/watch?v=kCc8FmEb1nY

- **Andrej Karpathy — "makemore" series** (YouTube, 2022–2023)
  Builds up to transformers from bigrams. Watch before this project.
  https://www.youtube.com/watch?v=PaCmpygFfXo

- **3Blue1Brown — "Attention in transformers"** (YouTube, 2024)
  Best visual intuition for attention weights.
  https://www.youtube.com/watch?v=eMlx5fFNoYc

## Best Written Resources

- **The Illustrated Transformer** (Jay Alammar, 2018)
  Canonical visual explanation of the transformer.
  https://jalammar.github.io/illustrated-transformer/

- **The Illustrated GPT-2** (Jay Alammar, 2019)
  Extends the above to the specific architecture we build.
  https://jalammar.github.io/illustrated-gpt2/

- **Lillian Weng — "Attention? Attention!"** (2018)
  Mathematical treatment of various attention mechanisms.
  https://lilianweng.github.io/posts/2018-06-24-attention/

- **Sebastian Raschka — "Build LLM from scratch"** (book, 2024)
  A full book covering exactly this material.
  https://github.com/rasbt/LLMs-from-scratch

## Code References

- **NanoGPT** (Karpathy) — the cleanest GPT-2 implementation in existence
  https://github.com/karpathy/nanoGPT

- **MinGPT** (Karpathy) — educational predecessor to NanoGPT
  https://github.com/karpathy/minGPT

- **GPT-2 original** (OpenAI)
  https://github.com/openai/gpt-2

## Running Data Sources

- **Kaggle running datasets**: https://www.kaggle.com/search?q=running+activities
- **Strava Metro data**: https://metro.strava.com/
- **USATF race results**: https://www.usatf.org/
- **r/running**: Public posts are fair game for a personal corpus
