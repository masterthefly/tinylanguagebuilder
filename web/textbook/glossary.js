// The textbook's built-in glossary, used by the "Explain" popover (explain.js).
// Entries are written by hand, not generated: each says what the idea is, then what it is in THIS project's model
// (vocabulary 512, 2 layers, 2 attention heads, 32 numbers per token, a 64-token window).
//
//   id       unique key (other entries refer to it in `related`)
//   term     the heading shown to the reader
//   aliases  words and phrases that should match it (case-insensitive, plural endings handled)
//   chapter  [number, title] of the chapter that teaches it, or null
//   short    the one-or-two sentence answer
//   more     optional extra detail
//   related  ids of neighbouring ideas

export const CHAPTERS = {
  1: "chapter1.html",
  2: "chapter2.html",
  3: "chapter3.html",
  4: "chapter4.html",
  5: "chapter5.html",
};

export const GLOSSARY = [
  // ---- chapter 1: tokens ----
  {
    id: "token", term: "Token", aliases: ["token", "tokens", "tokenise", "tokenize", "tokenised", "tokenized", "tokenisation", "tokenization"],
    chapter: [1, "Tokens"],
    short: "A piece of text the model treats as one unit: a whole common word, part of a rarer word, or a single character. The model never sees letters, only a list of token numbers.",
    more: "In this project a token is a short run of bytes, such as \" the\" (with its leading space) or \"ing\". Common words are usually one token; unusual words split into several.",
    related: ["tokenizer", "vocabulary", "bpe"],
  },
  {
    id: "tokenizer", term: "Tokenizer", aliases: ["tokenizer", "tokenizers"],
    chapter: [1, "Tokens"],
    short: "The step that cuts text into tokens and turns them into numbers (and back again). It is built once from sample text and then never changes.",
    related: ["token", "bpe", "vocabulary"],
  },
  {
    id: "bpe", term: "Byte-pair encoding (BPE)", aliases: ["byte-pair encoding", "byte pair encoding", "bpe", "byte-level"],
    chapter: [1, "Tokens"],
    short: "A way to build a vocabulary. Start with single bytes, then repeatedly find the pair of neighbouring tokens that occurs most often in the training text and glue it into one new token. Repeat until the vocabulary is the size you want.",
    more: "Because the rules are learned in order, using only the first k merges gives exactly the tokenizer you would get by asking for a smaller vocabulary. That is what the merge slider in chapter 1 does.",
    related: ["merge", "vocabulary", "byte"],
  },
  {
    id: "merge", term: "Merge", aliases: ["merge", "merges", "merging"],
    chapter: [1, "Tokens"],
    short: "One learned rule of byte-pair encoding: \"whenever token A is followed by token B, treat them as the single new token AB\". Each merge adds one token to the vocabulary.",
    related: ["bpe", "vocabulary"],
  },
  {
    id: "vocabulary", term: "Vocabulary", aliases: ["vocabulary", "vocab", "vocab size", "vocabulary size"],
    chapter: [1, "Tokens"],
    short: "The complete list of tokens a model knows, each with a fixed number (its id). A bigger vocabulary means fewer tokens per sentence but more rows for the model to learn.",
    more: "The textbook's model has a vocabulary of 512: 256 single bytes plus 256 learned merges.",
    related: ["token", "bpe", "embedding"],
  },
  {
    id: "byte", term: "Byte", aliases: ["byte", "bytes", "utf-8", "utf8"],
    chapter: [1, "Tokens"],
    short: "The smallest piece text is stored as: a number from 0 to 255. Plain English letters take one byte each; many other characters and emoji take two to four. Starting from bytes means the tokenizer can represent any text at all.",
    related: ["bpe", "token"],
  },

  // ---- chapter 2: embeddings ----
  {
    id: "embedding", term: "Embedding", aliases: ["embedding", "embeddings", "embed", "embedding table"],
    chapter: [2, "Embeddings"],
    short: "A short list of numbers (a vector) that stands for one token. The model keeps one such list per token in a table and looks it up when it sees the token. Training adjusts the lists so tokens used in similar ways end up with similar numbers.",
    more: "Here each token has 32 numbers. Before training they are random noise; afterwards neighbours in the table tend to be related words.",
    related: ["vector", "dimension", "cosine", "pca"],
  },
  {
    id: "vector", term: "Vector", aliases: ["vector", "vectors"],
    chapter: [2, "Embeddings"],
    short: "A list of numbers. You can think of it as an arrow or a point in space with one coordinate per number.",
    related: ["embedding", "dimension", "dot-product"],
  },
  {
    id: "dimension", term: "Dimension", aliases: ["dimension", "dimensions", "d_model", "d model"],
    chapter: [2, "Embeddings"],
    short: "One of the numbers in a vector. A 32-dimensional embedding is a list of 32 numbers. More dimensions let a token carry more information but cost more to train.",
    related: ["vector", "embedding"],
  },
  {
    id: "cosine", term: "Cosine similarity", aliases: ["cosine similarity", "cosine", "similarity", "nearest neighbour", "nearest neighbor", "neighbour", "neighbours", "neighbor", "neighbors"],
    chapter: [2, "Embeddings"],
    short: "A score for how nearly two vectors point the same way: 1 means identical direction, 0 unrelated, negative opposite. It ignores how long the arrows are.",
    more: "The \"nearest neighbours\" list in chapter 2 ranks every other token by this score against the one you picked.",
    related: ["embedding", "vector", "dot-product"],
  },
  {
    id: "pca", term: "2-D map (PCA)", aliases: ["pca", "principal component", "principal components", "2-d map", "map"],
    chapter: [2, "Embeddings"],
    short: "A way to squash 32 numbers per token down to 2 so they can be drawn on a flat map: it picks the two directions along which the tokens vary most. The picture is a shadow of the real table, so some distances are distorted.",
    related: ["embedding", "dimension"],
  },

  // ---- chapter 3: attention ----
  {
    id: "attention", term: "Attention", aliases: ["attention", "attend", "attends", "attention weights", "attention weight"],
    chapter: [3, "Attention"],
    short: "The step where each token looks back at earlier tokens and takes a weighted blend of what it finds. It is how a word like \"She\" can use information from \"Lily\" many tokens earlier.",
    more: "The weights are positive and add up to 1 for each token, and a high weight shows where information can flow, not why the model gave an answer.",
    related: ["qkv", "softmax", "causal-mask", "head"],
  },
  {
    id: "qkv", term: "Query, key, value", aliases: ["query", "queries", "key", "keys", "value", "values", "qkv"],
    chapter: [3, "Attention"],
    short: "Three short vectors made from each token. The query says what this token is looking for, the key says what it offers to others, and the value is what it hands over if chosen. Match a query against every earlier key to get scores, then blend the values by those scores.",
    related: ["attention", "dot-product", "softmax"],
  },
  {
    id: "dot-product", term: "Dot product", aliases: ["dot product", "dot-product", "dot products"],
    chapter: [3, "Attention"],
    short: "Multiply two vectors number by number and add up the results. It is large when the two vectors point the same way, so attention uses it as a match score between a query and a key.",
    related: ["qkv", "cosine"],
  },
  {
    id: "softmax", term: "Softmax", aliases: ["softmax"],
    chapter: [3, "Attention"],
    short: "Turns any list of scores into probabilities: raise e to each score (so everything is positive and bigger scores pull ahead), then divide each by the total so they add to exactly 1.",
    more: "Subtracting the largest score first changes nothing about the result but stops the numbers overflowing, which is why chapter 3's table has an \"e^(score minus top)\" column. Softmax is used twice: for attention weights, and to turn the final scores into next-token chances.",
    related: ["attention", "logit", "temperature"],
  },
  {
    id: "causal-mask", term: "Causal mask", aliases: ["causal mask", "mask", "masked", "causal", "the future"],
    chapter: [3, "Attention"],
    short: "The rule that a token may only look at itself and earlier tokens, never later ones. Without it the model could cheat by copying the answer during training. It shows up as the empty triangle in the attention picture.",
    related: ["attention"],
  },
  {
    id: "head", term: "Attention head", aliases: ["head", "heads", "attention head", "attention heads"],
    chapter: [3, "Attention"],
    short: "One independent copy of the attention mechanism, with its own query, key and value tables. Running several side by side lets different heads pick up different habits, such as \"look one token back\" or \"spread out\".",
    more: "The textbook's model has 2 heads in each of its 2 layers.",
    related: ["attention", "layer"],
  },
  {
    id: "layer", term: "Layer", aliases: ["layer", "layers"],
    chapter: [3, "Attention"],
    short: "One round of attention plus a small calculation, stacked on top of the previous round. Later layers work on tokens the earlier layers have already mixed, so they can combine information in ways one layer cannot.",
    related: ["head", "transformer"],
  },
  {
    id: "transformer", term: "Transformer", aliases: ["transformer", "transformers", "gpt"],
    chapter: [3, "Attention"],
    short: "The kind of model used here and in most modern language models: token embeddings, then a stack of layers each built around attention, then a final step that scores every token in the vocabulary as the possible next one.",
    related: ["attention", "layer", "embedding"],
  },
  {
    id: "context-window", term: "Context window", aliases: ["context window", "window", "context length", "context", "ctx"],
    chapter: [3, "Attention"],
    short: "The most tokens the model can look at at once. Anything older is invisible to it. This model's window is 64 tokens, so long text is cut to fit.",
    related: ["attention", "token"],
  },

  // ---- chapter 4: learning ----
  {
    id: "loss", term: "Loss", aliases: ["loss", "losses", "cross-entropy", "surprise"],
    chapter: [4, "Learning"],
    short: "A score for how badly the model did: the negative natural log of the probability it gave to the token that really came next. Near 0 means it was sure and right; large means it was surprised. Training tries to make the average loss small.",
    more: "A model that knows nothing gives every one of the 512 tokens an equal chance and scores ln 512, about 6.24.",
    related: ["probability", "gradient-descent", "ln", "held-out"],
  },
  {
    id: "ln", term: "ln (natural logarithm)", aliases: ["ln", "natural log", "natural logarithm", "log"],
    chapter: [4, "Learning"],
    short: "The inverse of e raised to a power: ln(e^x) = x. It turns small probabilities into large numbers (ln 1 = 0, ln 0.1 is about -2.3), which is why \"minus ln(p)\" works as a penalty that grows as the model's guess gets worse.",
    related: ["loss"],
  },
  {
    id: "probability", term: "Probability", aliases: ["probability", "probabilities", "chance", "chances", "likelihood"],
    chapter: [4, "Learning"],
    short: "A number from 0 to 1 saying how likely something is; the model gives one to every token as the possible next one, and they all add up to 1.",
    related: ["softmax", "logit", "loss"],
  },
  {
    id: "gradient", term: "Gradient", aliases: ["gradient", "gradients", "slope"],
    chapter: [4, "Learning"],
    short: "The slope of the loss: for every number in the model, which way and how steeply the loss changes if you nudge that number. It tells you which direction is downhill.",
    related: ["gradient-descent", "backprop", "loss"],
  },
  {
    id: "gradient-descent", term: "Gradient descent", aliases: ["gradient descent", "descent", "optimiser", "optimizer", "adamw", "adam"],
    chapter: [4, "Learning"],
    short: "The learning rule: new value = old value minus learning rate times slope. Take a small step downhill for every number, measure again, and repeat thousands of times.",
    more: "This project's trainer uses AdamW, a refinement that adapts the step size for each number, but the idea is the same.",
    related: ["gradient", "learning-rate", "loss"],
  },
  {
    id: "learning-rate", term: "Learning rate", aliases: ["learning rate", "step size", "lr"],
    chapter: [4, "Learning"],
    short: "How big each downhill step is. Too small and training crawls; too big and the steps overshoot, zigzag, and can make the loss grow instead of shrink (diverge).",
    related: ["gradient-descent", "diverge"],
  },
  {
    id: "diverge", term: "Divergence", aliases: ["diverge", "diverges", "diverged", "divergence", "overshoot", "overshoots"],
    chapter: [4, "Learning"],
    short: "When training gets worse instead of better because the steps are too large: each step lands further from the best point than the last. In the toy this happens above a learning rate of 1.0.",
    related: ["learning-rate", "gradient-descent"],
  },
  {
    id: "backprop", term: "Backpropagation", aliases: ["backpropagation", "backprop", "back-propagation"],
    chapter: [4, "Learning"],
    short: "An efficient way to compute the gradient for all of the model's numbers at once, by working backwards from the loss through each calculation the model made. It is what the project's Rust code does by hand.",
    related: ["gradient", "gradient-descent"],
  },
  {
    id: "parameter", term: "Parameter", aliases: ["parameter", "parameters", "weights", "model size"],
    chapter: [4, "Learning"],
    short: "One of the numbers inside the model that training adjusts: embedding values, attention tables and so on. The textbook's model has about 60,000 of them; large commercial models have billions or more.",
    related: ["gradient-descent", "embedding"],
  },
  {
    id: "batch", term: "Batch", aliases: ["batch", "batches", "mini-batch", "minibatch"],
    chapter: [4, "Learning"],
    short: "The small handful of text snippets the model is scored on in one training step. Using a different batch each time makes the training loss jittery, but over many steps it averages out.",
    related: ["training-step", "loss"],
  },
  {
    id: "training-step", term: "Training step", aliases: ["training step", "training steps", "step", "steps", "iteration", "iterations"],
    chapter: [4, "Learning"],
    short: "One round of: take a batch, measure the loss, compute the gradient, nudge every parameter a little. Chapter 4's model trains for a few hundred to a few thousand of them.",
    related: ["batch", "gradient-descent"],
  },
  {
    id: "held-out", term: "Held-out (validation) loss", aliases: ["held-out", "held out", "validation", "validation loss", "held-out loss", "training loss", "unseen text"],
    chapter: [4, "Learning"],
    short: "The loss measured on text the model never trained on. It is the honest score: if training loss keeps dropping while held-out loss stops, the model is memorising rather than learning.",
    related: ["loss", "overfitting"],
  },
  {
    id: "overfitting", term: "Overfitting (memorising)", aliases: ["overfitting", "overfit", "memorising", "memorizing", "memorisation", "memorization"],
    chapter: [4, "Learning"],
    short: "When a model gets good at its training text specifically, by remembering it, without getting better on new text. The gap between training loss and held-out loss is the warning sign.",
    related: ["held-out"],
  },
  {
    id: "random-init", term: "Untrained (random) model", aliases: ["untrained", "random", "random numbers", "randomly", "initialisation", "initialization", "step 0"],
    chapter: [4, "Learning"],
    short: "A model before any training: every parameter is a small random number, so its guesses are nearly even across the vocabulary and its loss is close to ln 512 (about 6.24).",
    related: ["loss", "parameter"],
  },

  // ---- chapter 5: sampling ----
  {
    id: "logit", term: "Logit (raw score)", aliases: ["logit", "logits", "raw score", "raw scores", "score", "scores"],
    chapter: [5, "Sampling"],
    short: "The model's raw score for each token as the possible next one, before it is turned into a probability. Bigger means more likely. Temperature works by dividing these scores before the softmax.",
    related: ["softmax", "temperature", "probability"],
  },
  {
    id: "temperature", term: "Temperature", aliases: ["temperature", "temp"],
    chapter: [5, "Sampling"],
    short: "A dial applied when picking the next token: divide the raw scores by T before softmax. Low T exaggerates the favourite (safe, repetitive); high T flattens the chances (varied, but more mistakes); T = 1 leaves the model's own probabilities untouched.",
    more: "Temperature only changes how the model's output is used; nothing inside the trained model changes.",
    related: ["logit", "softmax", "sampling", "greedy"],
  },
  {
    id: "top-k", term: "Top-k", aliases: ["top-k", "top k", "topk"],
    chapter: [5, "Sampling"],
    short: "Keep only the k most likely tokens, throw the rest away, rescale the remaining chances to add up to 1, and pick from those. It cuts off the long tail of silly options. Top-k of 1 is the same as always taking the favourite.",
    related: ["temperature", "sampling", "greedy"],
  },
  {
    id: "greedy", term: "Greedy choice", aliases: ["greedy", "argmax", "always take the favourite"],
    chapter: [5, "Sampling"],
    short: "Always picking the single most likely token. Predictable, but it tends to get stuck repeating itself.",
    related: ["sampling", "temperature", "top-k"],
  },
  {
    id: "sampling", term: "Sampling", aliases: ["sampling", "sample", "samples", "sampled", "weighted die", "die", "dice", "generation", "generate", "generating", "generated"],
    chapter: [5, "Sampling"],
    short: "Choosing the next token at random, weighted by the model's chances: a token with 30% probability is picked about 30% of the time. The chosen token is added to the text and the model is asked again, one token at a time.",
    related: ["temperature", "top-k", "greedy", "seed"],
  },
  {
    id: "seed", term: "Random seed", aliases: ["seed", "seeds", "random seed"],
    chapter: [5, "Sampling"],
    short: "The starting number of a random-number generator. The same seed gives the same \"random\" choices every time, which makes an experiment repeatable. A new seed gives different text.",
    related: ["sampling"],
  },
  {
    id: "entropy", term: "Uncertainty (entropy)", aliases: ["entropy", "uncertainty", "bits", "bit"],
    chapter: [5, "Sampling"],
    short: "How spread out a set of chances is, measured in bits. 0 bits means one token has all the probability; a flat spread over 512 tokens is 9 bits. Raising 2 to the power of the entropy gives the number of equally likely tokens it is \"as if\" choosing among.",
    related: ["probability", "temperature"],
  },

  // ---- beyond the five chapters ----
  {
    id: "jsonl", term: "JSONL (examples file)", aliases: ["jsonl", "json lines", "examples file", "prompt and response", "held-out examples"],
    chapter: null,
    short: "A text file with one JSON object per line. For fine-tuning here, each line is one example: {\"prompt\": \"...\", \"response\": \"...\"}. The model learns to produce the response when it sees the prompt.",
    more: "About a tenth of your examples are held back and never trained on, so the page can score the model on questions it has not seen.",
    related: ["fine-tuning", "held-out", "overfitting"],
  },
  {
    id: "pretraining", term: "Pre-training", aliases: ["pre-training", "pretraining", "pre-train", "pretrain", "pre-trained", "pretrained"],
    chapter: null,
    short: "The first and longest stage: the model reads lots of ordinary text and learns to predict the next token. This is the training you run in chapters 2 to 5.",
    related: ["fine-tuning", "loss"],
  },
  {
    id: "fine-tuning", term: "Fine-tuning (SFT)", aliases: ["fine-tuning", "fine tuning", "finetuning", "fine-tune", "sft", "supervised fine-tuning"],
    chapter: null,
    short: "Further training of an already pre-trained model on examples of the behaviour you want, such as an instruction followed by a good answer. The main app's second tab does this.",
    related: ["pretraining", "dpo"],
  },
  {
    id: "dpo", term: "Preference tuning (DPO)", aliases: ["dpo", "preference tuning", "preference", "preferences", "direct preference optimization", "direct preference optimisation"],
    chapter: null,
    short: "Training the model to prefer better answers over worse ones, using pairs of the form \"this answer is better than that one\". The main app's third tab does this with a checkable rule.",
    related: ["fine-tuning", "loss"],
  },
];
