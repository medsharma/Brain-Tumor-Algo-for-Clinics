# How to talk to me

- Be direct. No fluff, no hedging, no over-explaining.
- No jargon. If a technical term is unavoidable, explain it in plain words.
- Do not use em dashes.
- Use short sentences. Bullet points are good. Write like a clear LinkedIn post, not an essay.
- Give me the honest answer even if it is not what I want to hear. Especially then.
- If something is a bad idea, tell me straight and tell me why.
- Do not flatter me or pad answers with praise.
- When there is a risk to patients, say so plainly. Do not soften it.

# Repositories

There are two. Do not mix them up.

- **Clinical work goes here: https://github.com/medsharma/Brain-Tumor-Algo-for-Clinics**
  This is `origin`. Every change from 2026-07-30 onward goes to this repo.
- **The paper lives here: https://github.com/medsharma/Brain-Tumor-ML-Publication**
  Frozen at commit `b53c0b3`. Do not push to it. Do not add it as a remote.
  It is not configured in this checkout, and that is deliberate.

If a future task genuinely needs to update the paper repo, ask first. Do not
assume it.

# What I am working on

- A brain tumor MRI classifier I want to get to clinical grade for rural clinics.
- 4 classes: glioma, meningioma, pituitary, no tumor. PyTorch. Two backbones: ViT-B/16 and ResNet-50.
- Uses MC-Dropout (T=20 passes) for uncertainty and entropy-based defer-to-human.
- Right now I am running an outside-data test on the BRISC 2025 dataset, which the model never trained on.

# Rules for this work

- Never retrain or fine-tune on BRISC. Run my existing trained model on it untouched. That is the test.
- The number that matters most is how often it misses a real tumor and calls it no-tumor. Watch that above all.
- Report: four-way accuracy, tumor-vs-no-tumor accuracy, tumor miss rate, and whether the defer-to-human confidence still holds.

# Mission

Build a brain MRI tool that helps rural clinics catch tumors faster, in places
where there is no radiologist and scans wait days or weeks to be read.

The tool does not replace a doctor. It is a second set of eyes. It flags scans
that look abnormal, says how confident it is, shows where it looked, and pushes
the uncertain ones to a human. In a clinic with no specialist, that means the
right patients get prioritized and sent on quickly, instead of slipping through.

# Who it is for

Rural and under-resourced clinics, first. Places where the alternative to this
tool is nothing, or a long wait. We start where the need is highest and the bar
to add value is lowest.

# What it does, in order of importance

1. Tumor vs no tumor. The call a clinic can act on: does this person need urgent
   referral. This is the core.
2. Confidence and defer-to-human. The tool knows when it is unsure and says so.
   This is what makes it safe and what makes it trusted.
3. Tumor type (glioma, meningioma, pituitary). Useful context, but secondary.
   A clinic refers either way.
4. Show its work. A heatmap over the scan showing where the model looked to make
   its call. Lets a doctor see, in one glance, whether the model focused on the
   actual mass or on something irrelevant. A trust and sanity-check layer, not
   proof on its own.

# What it is not

- Not a diagnosis. A doctor or specialist makes the call.
- Not a full tumor detector. It knows three tumor families plus no-tumor. It does
  not yet cover metastases or rarer types. We are honest about this everywhere.
- Not autonomous. A human is always in the loop for now.

# How we get there, step by step

1. Prove it works on outside data. Run the trained model, untouched, on datasets
   it never saw (BRISC 2025 first). If accuracy and confidence hold up, we have
   something real. If not, we found out cheaply.
2. Teach it to spot scans it should not judge. Blurry images, non-brain scans,
   things outside its training. It must say "I don't know this" instead of
   guessing. This is what makes triage safe.
3. Test it against real doctors. Have radiologists use it on real cases and tell
   us if it actually helps.
4. Sort out the legal and regulatory path for the specific country we launch in.
5. Package it to run on a laptop, offline, and get it into pilot clinics.

# The rules we hold to

- Patient safety over speed, always. The worst mistake is calling a real tumor
  "no tumor" and sending someone home. We measure that specific error above all.
- Be honest about what the tool can and cannot do. Overclaiming gets people hurt
  and kills trust.
- Prove it on outside data before it touches a real patient anywhere.
- Human in the loop until the evidence says otherwise.
- Always let a human see why. The tool shows where it looked, not just what it
  decided, so a doctor can catch a wrong call fast.

# How we know we succeeded

A clinic with no radiologist can scan a patient, get a fast, trustworthy signal
on whether this needs urgent attention, know when to trust it and when to send it
to a human, and act on that. Fewer missed tumors. Faster referrals. Offline, on
a laptop, where it was never possible before.
