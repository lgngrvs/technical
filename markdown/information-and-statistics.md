# Information theory and mathematical statistics

### What actually is information lol 
The idea of information in the specific context of information theory is pretty interesting. I think the best way to get a sense for it is talking about surprisal with a specific thing.

You come to a fork in the road and you have no ideas about which way might be the correct way to go. We can represent them as paths $A$ and $B$. A natural way to represent this is with binary labels, hence it would be instead path $0$ for $A$ and path $1$ for $B$. Intuitively, if someone perfectly trustworthy tells you which way to go, they would be be giving you 1 bit (binary digit) of information: they're giving you a 0 or 1 to distinguish the options in the binary choice. 

What if you do have some information though? What if you have 70% confidence that path $A$ is correct, since someone earlier on the road who seemed somewhat trustworthy told you it was? They wouldn't be giving you still 1 bit of information, because you'd be updating less — you're less surprised. You already thought this was the case, they're just upgrading your confidence. No longer can we work with simple integers of bits.

Intuitively, you might think that they were giving you 0.3 bits of information making up for the difference in the outcomes. This doesn't work out mathematically, though, because that would mean that distinguishing between two perfectly uncertain options — events that have priors of 0.5 each — would only give you 0.5 bits of information, not two. 

Here's another way that this doesn't work out nicely: if you have two fair coins and you flip them both separately, each coin flip would theoretically give you 0.5 bits of information. However, if you looked at it from the perspective of their joint probability distribution — $P(\text{2 heads}) = 0.25$ for example — we would get 0.75 bits of information instead. This just straightforwardly doesn't work out — the same independent events, randomly combined, do not give different amounts information just based on whether you look at their probabilities together or separately. 

We want something that scales differently to calculate information — we want information to increase additively, not multiplicatively. The log function works nicely for that. We'll use the $-\log_2$ scale, since we're working in binary already. (The negative sign makes the log values positive for decimals.)

This gives us nice properties: 

- We get 1 bit of information from perfect uncertainty between 2 outcomes: $-\log_2(0.5) = 1$
- When we are looking at the coins, the information is the same whether or not we look at the joint probability distribution. $-\log_2(0.25) = 2 (-\log(0.5))= 2$. In other words, **while probability multiplies, information adds.** That's why we use the log scale.

![Information Graph](information-graph.png)
*Fig 1: The graph of $y = \log_2(x)$, where $y$ is the "information content" or the surprisal produced by the occurrence of an event whose probability in your mind is $x$.*

So there's the idea about why the information content of an event uses a log scale. To sum it up, $\text{Information} = -\log_2P(x)$ where $x$ is an outcome and $P(x)$ is the probability of that outcome. (Sorry to mix notations with $x$ being probability and $x$ being an outcome — hopefully that's not too annoying.)
### Entropy
We might want to look at information on the level of a probability distribution. We've talked about the information content of specific events, but how do we talk about our beliefs about an event — our internal distributions?

One thing we can talk about is *entropy.* Entropy in information theory describes something like, "How predictable is this distribution? How *surprised* will I be on average by an outcome?" In general, entropy measures the predictability of a distribution. (This is not a notion limited to information theory. For example, in statistical mechanics, the entropy of a particular macrostate/thermodynamic state is the number of possible microstates that could produce that macrostate — in other words, the unpredictability of the precise arrangements of atoms for a given macrostate; the unpredictability of the particular positions of atoms, summed over the distribution of all of them.)

Intuitively, to talk about how much we expect to be surprised given a distribution of the probabilities of possible outcomes of an event, we can do a simple expected value calculation: take the sum of the information content of each possible outcome, weighted by the probability of the outcome.

Hence, we can describe Shannon entropy (information-theoretic entropy) using the following formula:

$$ H(X) =-\sum_{x \in X} P(x) \space \log_2P(x) $$
where $X$ is a distribution of individual events $x$ and $P(x)$ is the probability you assign to the event. Hence, this sum represents going across each possible event in the distribution and computing the value $P(x) * \log_2P(x)$ — corresponding to the surprisal of an outcome, weighted by how likely it is — then summing that up for each distribution. 


![Entropy Distribution](entropy-distribution.png)
*Fig 2: The graph of $-x*\log_2(x)$, which you can call the "expected surprisal" — the probability of the outcome occurring times the surprise that you would receive if it did occur. Notice how the graph is skewed right, but low-probability events, even though they have high surprisal, will have low expected surprisal because they're so unlikely.*

Higher entropy means less predictability. For example, for the distribution $X: \{x_1 = 0.9, x_2 = 0.05, x_3 = 0.05\}$ the entropy is $-[(0.9* \log_2(0.9)) + 2*(0.05*\log_2(0.05)] \approx 0.473$, where for the less-predictable distribution $X: \{x_1 = 0.33, x_2 = 0.33, x_3 = 0.34\}$ the entropy is $-(2*(0.33*\log_2(0.33)) + (0.34*\log_2(0.34)) \approx 1.585$.
