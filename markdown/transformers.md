# How transformers work

Basically you have a couple important steps in a decoder-only transformer. I'm going to explain this in the context of generative language models first. Fundamentally, though, any sequential prediction task can use a transformer -- from [decision transformers](https://arxiv.org/pdf/2106.01345) to [image transformers](https://huggingface.co/docs/transformers/model_doc/vit), and probably to some crazy obscure things I've never heard of as well.

(Encoder-decoder transformers, like the ones in [the orignal paper](https://arxiv.org/pdf/1706.03762), are pretty cool, and I have weird half-baked theories about how they relate to language production, but it's easier to figure things out initially just with decoders. If you don't know the difference, don't worry about it right now. You can learn the difference later. For now, just think, "LLMs like ChatGPT are decoder-only, text classification models like BERT are encoder-only, and machine translation models are encoder-decoder.")

1. Embedding (semantic + positional) 
2. Flows through a bunch of transformer blocks
	1. "residual stream" is just the input -- the initial embedding if it's the first layer of the transformer, and in later layers it's just the output of the previous layer
	2. You take the input and run it through multi-head attention, combining together all the attention heads' outputs and adding it to the residual stream
	3. Then you take the newly-updated residual stream and put it through an MLP network, and then output it to the next transformer block or just send it to unembedding
3. After you've gone through all the transformer blocks/layers, you use softmax to turn everything into probabilities, and then you get your output -- a probability distribution over the next token. 

 I'll try and explain the intuitions for each thing, and how it actually works inside -- probably stay at least for the intuitions since these seem to be skipped by many resources, and then refer to other better-written resources for technical details on internal implementations. I'll link some, probably, once I write the sections.
