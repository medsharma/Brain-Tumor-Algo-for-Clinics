# If you are thinking about using this in a clinic

Written for a clinician, a clinic manager, or anyone deciding whether this tool
should go anywhere near a patient. No maths. No jargon without a plain-words
explanation.

**The short answer is no, not yet, and this page explains why in enough detail
that you can judge for yourself.**

Last updated: 2026-07-31.

---

## What the tool does

You give it one brain MRI image, saved as a JPEG or PNG. It tells you:

- whether it thinks there is a tumour,
- which of three tumour families it looks most like,
- how sure it is,
- and it colours in the part of the image that drove its answer.

It runs on a laptop with no internet. Nothing is uploaded anywhere.

That is genuinely useful in a place where a scan otherwise waits days or weeks
for a radiologist. It is not a diagnosis and it is not a substitute for one.

---

## The three things you most need to know

### 1. It misses about 1 tumour in every 100

On the only data it has been tested on, roughly one in a hundred real tumours
comes back as "no tumour".

That is on **easy** data: images from the same collection the tool learned from.
On real scans from your clinic, from your scanner, on your patients, the number
is **unknown**. It has never been tried. There is no reason to assume it would be
better and good reason to expect it to be worse.

Gliomas, which are the most aggressive of the three types it knows, are missed
about twice as often as the others.

### 2. "No tumour" does not mean the scan is normal

This is the most important sentence on this page.

The tool knows exactly four things: glioma, meningioma, pituitary tumour, and
"none of those three".

That last one gets displayed as "no tumour". But it really means **"not one of my
three tumour types"**, which is a much smaller claim. A scan can come back "no
tumour" and contain:

- a stroke
- a bleed
- an infection or abscess
- multiple sclerosis
- hydrocephalus
- a metastasis, which is cancer that spread from elsewhere in the body and is one
  of the **most common** brain tumours
- any tumour type outside the three it knows: lymphoma, acoustic neuroma,
  craniopharyngioma, and others

For every one of those, the tool will say "no tumour", and it will be technically
correct and clinically useless. A patient with a brain bleed can get a confident
"no tumour" from this tool.

**If a patient has symptoms, refer them. This tool does not overrule what is in
front of you, and it was never built to.**

### 3. When it is wrong, it sometimes sounds certain

The tool tells you how confident it is, and that confidence number is real and
useful most of the time. Scans it is unsure about really are the ones it gets
wrong most often.

But not always. Of the tumours it missed, about 1 in 10 were missed while the
tool was more than 90% confident. So a small number of missed tumours look, on
screen, exactly like a clean, confident answer.

There is no setting that fixes this. It is a property of the tool.

---

## What has never been done

Not "not done well". Not done at all.

- **No radiologist has ever used this on a real case.** Nobody has measured
  whether it helps you, slows you down, or misleads you.
- **It has never been used in a clinic.** Not once. No pilot.
- **No patient outcome has ever been measured.** There is no evidence that
  anything this tool does changes what happens to a person.
- **It has no regulatory approval anywhere.** Not FDA, not CE, not any national
  regulator. Nothing has been submitted.
- **It has never been tested on data from a different hospital.** We tried. The
  dataset we thought was independent turned out to be mostly the same image files
  the tool had already learned from.
- **It has never been tested on children.**
- **It has never been tested on any named scanner.** We do not know what scanners
  produced the images it learned from. That information does not exist.
- **We cannot tell you whether it works equally well for men and women, or for
  people of different ages or ethnicities.** The images it learned from have no
  such information attached. None. So no check was possible.

That last one deserves a moment. This tool is aimed at under-served communities.
We cannot show it works for them, because the data does not allow the check.
Nobody is hiding a bad result. There is no result.

---

## The false alarm problem, which nobody mentions enough

The tool also gets it wrong the other way. It flags scans as suspicious when
there is no tumour, about 2 times in every 100 clean scans.

Two in a hundred sounds small. Here is why it is not.

Tumours are rare in an unselected clinic population. If 2 out of every 100 people
you scan actually have one of these tumours, then out of every 100 flagged
scans, only about **54 will really have a tumour**. Nearly half your referrals
would be for nothing.

| how common tumours are in the people you scan | of the scans it flags, how many really have a tumour |
|---|---|
| 15 in 100 | about 91 |
| 5 in 100 | about 75 |
| 2 in 100 | about 54 |
| 1 in 100 | about 37 |

In the setting this tool is aimed at, a referral is not a small thing. It is a
long journey, transport money, days of lost work, possibly for a family member
too, and it uses a specialist appointment somebody else needed.

**Nobody has measured how common tumours are in a real target clinic**, so we
cannot tell you which row of that table applies to you. That is a question worth
answering before anything else happens.

---

## What would have to change before this is safe to use

1. **Radiologists test it on real cases** and tell us whether it helps.
2. **It is tested on scans from a genuinely different source**, so we know it
   works on more than the images it grew up with.
3. **It is trialled in a clinic like yours**, prospectively, so we learn the real
   false-alarm burden on the real referral pathway.
4. **Somebody answers who is responsible** if it says "no tumour" and it is
   wrong. Currently nobody has.
5. **A regulator in your country signs off.** No country has been chosen and
   nothing has been submitted.

None of these are in progress.

---

## If you are being offered this tool today

Reasonable questions to ask whoever is offering it:

- Has a radiologist ever tested it? *(Today the answer is no.)*
- Has it been tested on scans from a hospital, not just a research dataset?
  *(Today the answer is no.)*
- What is its miss rate, and on what data? *(About 1 in 100, on research images
  from the same collection it learned from. Unknown on real scans.)*
- Who is responsible if it misses something? *(Currently unanswered.)*
- Has any regulator approved it? *(No.)*

If any of those answers has changed since 2026-07-31, ask to see what changed and
where it is written down. Everything on this page is checkable in the repository:
[MODEL_CARD.md](../MODEL_CARD.md) has the numbers,
[LIMITATIONS.md](../LIMITATIONS.md) has the full list of gaps.

---

## Why this page is so negative

Because the alternative is worse.

A tool like this can genuinely help in a clinic with no radiologist. The way it
stops helping is if somebody trusts it more than the evidence supports, sends a
patient home, and the patient turns out to have had a tumour it saw and
dismissed. That would harm a patient and it would also destroy the trust that any
future version of this needs.

Being clear about what is not known is how this eventually becomes usable. It is
not an argument against the tool. It is the work.
