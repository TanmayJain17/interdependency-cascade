# Week 32 — spoken update (about 9 minutes read aloud)

---

Two weeks ago, Dr. Lin asked a question about our fast model. Until now the fast model worked like this: it reads the flood map, works out for each site how likely the water is to knock it out, then rolls the dice to pick one set of knocked-out sites, runs the network once, and repeats that a hundred times and averages. The question was: can we skip the dice? Can the network read the flood map once and give every site's failure probability directly?

This week I built that and tested it. I'll tell you what I did, what came out, and what it means.

---

**What I did.**

The good news was that we did not need any new simulation. For each of our 70 storms we already have one thousand simulator runs. If you ask, for one site, "in how many of those thousand runs had this site failed by hour 24", that share is exactly the probability we want the new model to produce. So the training targets already existed.

The new model gets, for every site, three things: how deep the water is there, how likely the water is to knock that site out directly — that is the same fragility curve the simulator uses — and a summary of how big the storm is overall. It then only has to learn one thing: how likely the cascade is to reach each site. The final answer is put together as: the chance the water takes the site out, plus the chance it survives the water but the cascade gets it. We give the network the part we already know, and make it learn the part we don't.

For the test, I split the storms by storm, never by run: 50 to train on, 7 to decide when to stop training, and the same 13 unseen storms we used last week, so every comparison is on the same ground. I trained it in four settings — with the flood arriving all at once, or arriving over hours; and with or without telling the model when the water reaches each site — and at three training sizes, 17, 34 and 50 storms, to see whether more storms keep helping. Two random starts each. Twenty-four models in total.

Before training a single one, I wrote down seven predictions and committed them to the repository on the 27th. Three passed, four missed. I'll explain the misses, because two of them are the model doing better than I expected.

---

**Result one. One pass does what a hundred did.**

The way we score is simple: for each site and each time horizon, the model gives a probability, the simulator gives the share of its thousand runs where that site failed, and we charge the model for how far off it is. Lower is better. And there is a floor: because the simulator's own answer comes from a thousand dice rolls, even a perfect model cannot score zero.

In all four settings, the new one-pass model scores the same as the old hundred-pass model, to two decimals. Static: 0.069 against 0.071. Arrival: 0.116 against 0.114. With timing inputs: 0.104 against 0.105. Storm by storm, the old model wins only two to five of the thirteen storms, and never by much. And the new model takes about eleven milliseconds per storm instead of about a second.

Under static labels, the new model is within seven percent of that floor — within seven percent of perfect. Under arrival labels, both models sit 25 to 40 percent above the floor. That is where the remaining work is.

My prediction here was that the old model would clearly win. It did not. That is the first miss, in the good direction.

---

**Result two. It gets the size of every storm right.**

If you add up the model's probabilities over all sites, you get the number of sites it expects to fail. For every one of the thirteen storms, that number is within three percent of the simulator's — on the smallest storm, 264 against 270; on the tipping storm t580, 1,159 against 1,167; on the biggest, 2,586 against 2,659. Thirteen out of thirteen. Last week, before we added the storm-size input, the model was three to five times too high on small storms. That prediction passed.

---

**Result three. Against a lookup table.**

The simplest competitor is a table: for a new storm, find the training storm of closest size and copy its answers. Because the GISSR maps are nested — a bigger storm is the smaller storm with more water — that table is very hard to beat. Under static labels it still wins on nine of twelve GISSR storms, by a hair. Under arrival labels, where timing matters, it picks storms with the wrong timing and the model pulls ahead. And on GeoClaw, the one test storm with a different shape, the table falls apart: its error is 0.31 against the model's 0.077. A table works as long as the new storm looks like an old one. The model keeps working when it doesn't. My prediction was that the table would win everywhere on GISSR; it only did under static labels. Second miss, in the good direction.

---

**Result four — and this is the one I want you to take away. More of the same storms stopped helping.**

Going from 17 to 34 training storms made the model 13 percent better. Going from 34 to 50 did nothing.

And when I looked at where that first gain came from, it was entirely on the storms above 3.6 metres. The old 17 storms had no examples that high. The new storms filled that gap, and on the biggest test storms the model got up to 35 percent better. Inside the range the old storms already covered, the gain was two percent — basically nothing. So the model did not improve because it saw more maps. It improved because it saw water levels it had never seen. Once those were covered, sixteen more GISSR maps added nothing.

I had written down in advance: if the second step gives less than a fifth of the first, we read it as "the limit is the variety of maps, not the number". That is what happened. The GISSR maps are the same shape at different heights. Generating more of them will give us heights we already have.

---

**Where it falls short.** Three things.

First, the arrival-time case — the realistic one — is not solved. Both models are 25 to 40 percent above the floor there, mostly on the biggest storms and on GeoClaw. Telling the model the arrival times helps, but doesn't close it.

Second, GeoClaw. It is the only test storm shaped differently from the GISSR family. Under arrival labels it is our worst case, and it actually gets worse as we add GISSR storms to training — from 0.16 with 17 storms to 0.18 with 50. The old model gets 0.13. The new model is being pulled toward the GISSR shape. It is one storm, so it is a warning, not a result — but it is the clearest sign we have that what the model is missing is storms of different shapes.

Third, on the smallest storm under arrival labels, the failure count is 17 percent low. Under static labels it is two percent.

---

**What it means.**

For Dr. Lin's question: yes, the one-pass model works. On this family of storms it is as accurate as the hundred-pass version, a hundred times faster, and better on storm size. Giving it the fragility probability and letting it learn only the cascade is what made fifty storms enough.

For "it will need a few hundred flood conditions": it needs more data, but not more GISSR storms — 34 was as good as 50. What would move it is maps with different shapes: different tracks, different surge patterns. The first option we control is more GeoClaw runs with different tracks. There are outside sources too — NOAA's SLOSH scenarios for the New York basin, and the FEMA synthetic-storm library — which I still have to check for access and coverage.

For now I would keep both models: the new one for speed and simplicity, the old one where we need to see run-to-run spread on tipping storms, and for GeoClaw under arrival labels.

Next week: understand why the big storms and GeoClaw miss the floor under arrival labels, try GeoClaw with different tracks as the first source of shape diversity, and add the direct model to the report.

One last thing on process. After the runs I made two corrections, both to the tolerance of a sanity check, neither to a prediction or a result — and the raw outputs are kept alongside. Everything is committed on a new branch; the two-stage work is closed and tagged.

That's the update. Happy to take questions.
