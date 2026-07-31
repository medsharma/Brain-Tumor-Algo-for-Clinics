# Serving this over a network

The clinic build listens on `127.0.0.1` and nothing else. This file is about the
other mode: running it as a website, where a browser somewhere else uploads a
scan and this machine reads it.

It works. Read the next section before you use it.

---

## What changes the moment you do this

**The scan leaves the machine it was taken on.** That is the whole difference,
and every consequence follows from it.

- **You become the custodian of patient images.** They travel over a network to
  a machine you control. In most countries that is a legal position with
  obligations attached, not a technical detail. If real patient scans will ever
  touch it, find out what applies where you are *before* it is running, not
  after.
- **A public URL means anyone can upload anything.** There is no login. The
  "NOT FOR CLINICAL USE" banner is the only thing standing between a stranger
  and a tumour verdict on a scan of their own head.
- **The offline guarantee stops being true**, and the app stops claiming it. The
  upload page rewrites its "Nothing leaves this laptop" line into a warning that
  the image is being sent to a server. Do not put that line back.
- **HTTP is not encrypted.** On a LAN, anyone who can see the traffic can see
  the scans. Put it behind HTTPS before it crosses anything you do not control.

None of this is an argument against doing it. A pilot with de-identified scans,
or a demo for people who will never install a 5 GB folder, are both good reasons.
It is an argument for knowing which one you are doing.

---

## Turning it on

Two things are needed, and both are deliberate. Neither happens by accident.

```
set MRI_TRIAGE_ALLOW_PUBLIC_BIND=1
python -m app --host 0.0.0.0 --port 8765
```

On macOS or Linux:

```
MRI_TRIAGE_ALLOW_PUBLIC_BIND=1 python -m app --host 0.0.0.0 --port 8765
```

Without the environment variable, `--host 0.0.0.0` is refused and the app tells
you why. That is the intended behaviour and it is covered by tests.

The variable unlocks two guards at once, on purpose:

| guard | what it does |
|---|---|
| bind check | refuses a network-visible listen address |
| per-request check | refuses requests whose client is not loopback |

They are both released by the same switch. Releasing only the first produced a
server that listened on the network and then answered `403` to everyone on it,
which looks exactly like a firewall problem and is not one.

---

## Reaching it

**From another machine on the same network**, use this machine's LAN address:

```
http://<this-machine-lan-ip>:8765/
```

Find it with `ipconfig` on Windows or `ip addr` on Linux. It usually starts
`192.168.` or `10.`.

**Windows Firewall will probably block the first connection.** Allow it for
private networks only, never public.

**From the internet** you need one of:

- a tunnel, for a quick demo: `cloudflared tunnel --url http://localhost:8765`
  or `ngrok http 8765`. Fastest route to a shareable link. Anyone with the URL
  gets in.
- a real host: a VM with the repo, the model files, and a reverse proxy
  terminating HTTPS. This is what a pilot should use.

---

## What it needs to run

CPU only, no GPU required.

| | |
|---|---|
| memory | 8 GB, and the five ensemble models want most of 4 GB resident |
| disk | 1.7 GB of model files, plus the Python environment |
| speed | roughly 0.8 to 2 seconds per scan on a laptop CPU |

**It handles one scan at a time.** There is no queue and no worker pool. Two
people uploading at once will both wait. For anything past a handful of
simultaneous users you need a real server setup, which does not exist yet.

---

## What is written to disk

**Image pixels are not saved.** They are held in memory for the request and
dropped.

**One audit line per scan is saved**, as JSONL under the app's data directory.
It holds the result, the timings, the model configuration and a SHA-256 of the
image. Not the image, and not the file name unless someone explicitly ticks the
box on the result page.

A SHA-256 is not a picture, but it is a stable unique identifier: two uploads of
the same file produce the same hash, so the log can show that a scan was seen
twice. Treat the audit log as sensitive.

---

## Before you point real patients at it

- [ ] HTTPS, not plain HTTP.
- [ ] Decide who is allowed to reach it. There is no login in this app.
- [ ] Decide how long audit logs are kept and who can read them.
- [ ] Answer the legal question for your country about holding patient images.
- [ ] Keep the "NOT FOR CLINICAL USE" banner. It is accurate: this tool has
      never been validated on data it did not train on, and no radiologist has
      reviewed a single output. See `MODEL_CARD.md`.

The last one is not paperwork. On the uncontaminated test data, three tumours in
1,476 were sent home, and **all three were sent home confidently**. No
uncertainty threshold catches those. The only control that does is the
instruction printed on every result: if the patient has symptoms, refer them
anyway.
