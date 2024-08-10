# Formal models of decision-making

*"Wouldn't it be cool to have systematic, predictable, accurate decision-making under uncertainty?"* said every systematizer ever. We have good and bad news. The good news is that lots of people want this, and they started making theories about it. The bad news is that this is really hard.

---

The **Nash Equilibrium** is the strategy for a game such that, holding the other players' strategies constant, there would be no incentive for the player to change strategy. *Hmm I actually don't properly get the details here*

- For example, you and 20 people are playing a game where you pick a number between 0 and 100, such that your guess will be the closest guess to 2/3 times the mean of everyone's guesses. Normally this is kind of difficult -- you have to simulate everyone else in your head, then simulate them simulating you, then simulate them simulating you simulating them, et cetera, then guess what level the others are simulating you at, and then use that to pick a number $n$ for $(2/3)^n * 50$ -- but there's a strategy where everyone can win instead: *just pick 0*. 

- If everyone picks 0 in the first round of this game, and then you decide to play a second one, no one has a reason to unilaterally change their guess -- if they do, they'll be ~guaranteed not to win. Hence, once everyone picks 0, you're at a stable strategic state: the Nash Equilibrium. 


You could argue that the Nash Equilibrium is generally the "rational choice" for an agent, since it's the only place where you don't have an incentive to change strategy, but for the classical single-round prisoner's dilemma, the Nash Equilibrium is to defect. If either player cooperates the other has an incentive to change strategy and defect.

Can't we do better than this?

Game theory is what we call modeling games like these as a whole: you're modeling multiple agents, looking at their interactions. Decision theories, on the other hand, are about modeling different formal algorithms for *individual* decisions.

This is a bunch of notes on this and related topics.

## Decision theories

*This section is in progress.*

If you want to formalize reasoning and decision-making, make your own decision theory. It's fun! Try it at home!

Sources: 
- [Introduction to Logical Decision Theory](https://arbital.com/p/logical_dt/?l=5gc)
- [Evidential Decision Theory - Wikipedia](https://en.wikipedia.org/wiki/Evidential_decision_theory)
- [An explanation of decision theories](https://www.lesswrong.com/posts/RiYhceiQy4w8JQAsn/an-explanation-of-decision-theories)

The typical conventional reasoning we might use about the world is what we'd describe as **Causal Decision Theory**, or CDT. CDT evaluates decisions for an agent that is distinct from its environment at a specific point in time (i.e. the theory doesn't consider itself "part of the world," which is important if you start doing weird things like blackmail and acausal trades), making decisions based on its predictions about how well things will turn out. 

In other words, you hold everything else constant, and look at the agent's decision in isolation; you look at the world in which the agent makes decision A as opposed to decision B, you compare the outcomes, and decide based on which one looks better -- which one has higher *expected utility*.

It's really not that deep. Literally, "look at the causal consequences of your actions and decide which consequences you like better."

But there are outcomes where this formal theory doesn't match our intuitions.[^1] The typical example is that an economist can argue that, based on CDT, that it's pointless for any given individual to vote; it's impossibly unlikely that a democratic election with tens of thousands of voters will be swung by *your vote* in particular, nevermind one with millions of votes. It's so unlikely that it's not worth it to even vote in the first place -- the wasted time is a net negative in expectation.

I don't really like this. I like voting in elections. I'd like a formalization of rationality that says you should vote in elections, since my imagination of a "perfectly rational society" filled with rational agents where everyone follows perfect systematic rationality (which is what we're trying to create here, presumably) is *not* one in which no one votes in elections.

The rest of decision theory is weirder and different in order to try and account for this. We can group all these decision theories under the name **Logical decision theories**. I'll start with the original competitor (I guess), which is **Evidential decision theory.**
### Evidential Decision Theory
To this point, I've been basically going for introduction and haven't done much formalization — which is funny considering the whole point of decision theories is to allow formalization of decision making.

The reason that I did that is because I wanted to get you started with intuitions that CDT is "the default." If I showed you the formalization first, it would distract from that point.

But now we need the formalizations in order to proceed with intuition-building — at least, I do, lol. Why? EDT is a very natural way of thinking about decision-making, but that's only visible in its formalization. Attempts at intuition-building definitions of EDT kind of suck, just due to its nature. For example: 

> For a while, there were two dominant decision theories, causal decision theory (CDT), and evidential decision theory (EDT):
> 
> - Causal decision theory: Take actions that physically cause you to get higher expected utility.
> - Evidential decision theory: Take actions that would be evidence of you getting higher expected utility.

(from [LessWrong](https://www.lesswrong.com/posts/RiYhceiQy4w8JQAsn/an-explanation-of-decision-theories), also linked above in sources)

Causal decision theory makes sense to me — you just do things that cause outcomes you like better — but it doesn't really click for me why doing things that "would be evidence of you getting higher expected utility" makes sense or is different.

Here is [Wikipedia's version](https://en.wikipedia.org/wiki/Evidential_decision_theory): 

> when a rational agent is confronted with a set of possible actions, one should select the action with the highest _news value_, that is, the action which would be indicative of the best outcome in expectation if one received the "news" that it had been taken. In other words, it recommends to "do what you most want to learn that you will do."

This didn't help much for me; seeing the formalization further down in the article did get rid of the confusion for me, and hopefully it will for you. 

Here's how you formalize CDT. You calculate the *counterfactual* expected utility:

$$\mathbb{E}_{CDT}(a) = \sum_{o \in \mathcal{O}} \mathbb{U}(o_{i})*P(a \space\Box \rightarrow o_{i})$$
Where

- $\mathbb{E}_{CDT}(a)$ is the expected utility of an action $a$ under CDT
- $\mathbb{U}(o_{i})$ is the utility (i.e. desirability) of the outcome ${o_i}$
- $P(a \space \Box → o_{i})$ is the probability that action $a$ *causes* outcome $o_{i}$ in some counterfactual world where $a$ is taken

For EDT, you just use a simple expected utility function:

$$\mathbb{E}_{EDT}(a) = \sum_{o \in \mathcal{O}} \mathbb{U}(o_{i}) * P(o_{i}|a)$$

Where

- $\mathbb{E}_{EDT}(a_{x})$ is the expected utility of an action ${a_x}$
- $\mathbb{U}(o_{i})$ is the utility of the outcome ${o_i}$
- $P(o_{i}|a)$ is your probability of the outcome $o_i$ if the action $a$ is taken.

With CDT, your actions must *cause* $o_i$. With EDT, your action acts as Bayesian evidence that outcome $o_i$ is more likely to have occurred. The latter is focused on, simply, *how your beliefs update* in response to an action — it doesn't matter the causal relationship of your action to the outcome, only how it causes you to update your beliefs.

If you're not familiar with the ideas of predictive processing, or just Bayesian statistics in general,  this might be a weird-feeling frame to you and it might not click. If that's the case, those are the two things you should try to go understand I guess — Bayes' theorem, and an intuition for how formal probabilistic models of the human mind treat belief updating and such. (maybe this is a topic for some future notes or something.) 

Anyway, some thought experiments will likely help the intuition. The classic one is Newcomb's paradox. 

. . .

If we want to do better than Defect-Defect in the classic prisoner's dilemma -- where you're in two different rooms, you'll need *acausal trades*. Maybe. Maybe there are other ways to do this or something that aren't absurd, but this one's kind of interesting and fun.

## Game theory with computers

*This section is in progress.*

from [Parametric Bounded Lob's Theorem and Robust Cooperation of Bounded Agents](http://arxiv.org/abs/1602.04184)

- We can have a version of the prisoner's dilemma, but instead of two people playing it, it's two computers playing it, and they each get access to each other's source code -- i.e. they can perfectly simulate one another. (I think, maybe there's some weird edge case here)
- Fun note: if you have a formally-defined program, it can be rewritten in math or something. Since the computer program is deterministic, we can use proofs to determine the outcome of our bot. Specifically: 


		def FairBot_k(Opponent):
			search for a proof of length k characters that 
			Opponent(FairBot_k) = C  

			if found,  
				return C  
			else  
				return D


(We bound it, I'm assuming, so that the program is guaranteed to terminate.)

This is great. What if we put FairBot against itself? It looks for a proof of length k that the opponent will cooperate, and in order to figure it out must find a proof that its opponent will find a proof that itself finds a proof that its opponent...

Seemingly this goes infinitely, and since we have bounded things at length `k` the program will terminate without finding a proof and fail. 

Actually this is not the case. There's some interesting math around provability that shows us why, called **Lob's theorem**. Using Lob's theorem, we can create "robust cooperative program equilibria for computationally bounded agents" -- this means unexploitable agents in this computer-based prisoner's dilemma game.

The statement "$\Box_{k} p$" means, "$p$ has a proof of $k$ or fewer characters."


[^1]: As is always the case.
