# Deployment Guide

This explains how to run the system as a real, always-on deployment
with **zero terminal commands needed for day-to-day use** (setup is a
one-time terminal step; everything after that -- monitoring,
registering employees, exporting records, changing sensitivity -- is
done from the browser).

It also explains, honestly, the one thing that can't be avoided: the
**camera has to be physically reachable from whatever machine does the
video processing.** That's not a limitation of this codebase -- it's
true of any camera-based system. Where the "administrator/employer
dashboard" runs is a separate question from where the video is
processed, and the two are more flexible than they might seem. Read
"Architecture reality" below before assuming you need one giant cloud
server.

---

## 1. One-time setup (terminal, done once)

```bash
# On the machine that will run this permanently (see "Where does this
# run" below):
git clone <your repo> attendance_system   # or copy the folder over
cd attendance_system
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Verify the install actually worked before going further** -- two
real dependency traps were found and fixed this session (see
`CONTEXT_HANDOFF.md`, Session 14 and 16): mediapipe silently losing its
hand-detection API on a newer version, and insightface pulling in a
protobuf version mediapipe can't use. Confirm both:

```bash
python3 -c "import mediapipe as mp; mp.solutions.hands.Hands(); print('mediapipe OK')"
python3 -c "from insightface.app import FaceAnalysis; print('insightface OK')"
```

If either fails, re-read `requirements.txt`'s comments next to
`mediapipe==0.10.14` and `onnx==1.16.0` -- those exact pins are what
make both of these coexist in the same environment.

**Calibrate the entrance line once** (this needs a person to actually
walk through frame, so it's a physical, one-time task, not something
sensibly done as a "browser feature"):

```bash
python main.py --sensitivity balanced
# press 'C', follow the on-screen calibration instructions, then quit
```

This saves a calibration file the dashboard reads automatically from
then on.

That's it for terminal use. Registering employees, watching the live
feed, exporting records, and adjusting sensitivity/day-boundary are all
done from the dashboard from here on.

---

## 2. Running it as an always-on service (not a terminal window)

Flask's own dev server (what `app.run(...)` uses) is not meant to be
left running unattended -- it says so itself on startup. `waitress`
(already in `requirements.txt`) is a production-grade WSGI server that
is. This is the actual command to run permanently, in place of `python
web_dashboard.py`:

```bash
waitress-serve --host=0.0.0.0 --port=5000 web_dashboard:app
```

That alone is enough to test with. To make it survive a reboot and
restart automatically if it ever crashes, without a terminal window
open:

**Linux (systemd)** -- create `/etc/systemd/system/attendance.service`:
```ini
[Unit]
Description=Attendance Dashboard
After=network.target

[Service]
WorkingDirectory=/opt/attendance_system
ExecStart=/opt/attendance_system/venv/bin/waitress-serve --host=0.0.0.0 --port=5000 web_dashboard:app
Restart=always
User=attendance

[Install]
WantedBy=multi-user.target
```
Then: `sudo systemctl enable --now attendance`.

**Windows** -- use Task Scheduler with "Run whether user is logged on
or not", action = the same `waitress-serve` command, trigger = "At
startup". (NSSM is a common, simpler alternative if you want it to
show up as a proper Windows Service.)

Either way: from this point on, nobody needs to open a terminal to use
the system. The camera keeps running, the dashboard keeps serving,
across reboots.

---

## 3. Making it reachable as "a link"

**On the office network (the realistic default for a single site):**
whatever machine is running the service, other devices on the same
network open `http://<that machine's LAN IP>:5000`. No extra setup.
This alone satisfies "no terminal, just open a link" for anyone in the
building -- HR checking the dashboard from their own desk, a manager
checking on their phone over WiFi.

**Remote access (checking it from home, or a manager at another
site):** do **not** just port-forward 5000 to the public internet --
the dashboard now has an admin login, but a raw port with plain http
still sends that password unencrypted. Two safer options,
in order of how much setup they need:
- **Tailscale** (or another WireGuard-based mesh VPN): install it on
  the server and on whatever device needs access; it gives you a
  private, stable address for the dashboard that only your devices can
  reach. Free for small teams, ~10 minutes to set up, no port-forwarding.
- **A reverse proxy with a real domain + HTTPS + login** (Caddy or
  Nginx in front of waitress, Caddy auto-provisions a Let's Encrypt
  certificate for you): the right answer for "give it a proper URL".
  This is now set up for you -- see AWS_DEPLOY_GUIDE.md, Dockerfile,
  docker-compose.yml and deploy/Caddyfile.

---

## 4. Architecture reality: what "deploy to the cloud" actually means here

**The camera's video has to be processed by something with a network
path to that camera.** There are exactly two honest ways to satisfy
that:

**(a) Edge deployment (what to actually do for one site) -- recommended
now:** a small PC (an Intel NUC-class machine, ~$300-500, or even a
capable mini-PC) sits on-site, plugged into the network the door
camera is on (or directly to a USB webcam). It runs everything --
`web_dashboard.py`, the database, the dashboard -- exactly as described
above. This is what every real commercial face-attendance box (Hikvision,
ZKTeco, etc.) actually is under the hood: an edge device near the door,
not a cloud service watching the video. Video inference over the
public internet would also mean constant bandwidth and multi-second
lag between someone waving and the system reacting -- edge processing
is not just the safe choice, it's the only responsive one.

**(b) Multi-site, centrally managed (the real "sell this to large
companies" architecture) -- a real next project, not a flip of a
switch:** each site still runs its own edge box as in (a), but instead
of (or in addition to) each site's own local dashboard, `record_event()`
(in `face_id/identified_attendance.py`) also pushes each event to a
central cloud API over HTTPS, which writes to one shared database (e.g.
Postgres, not SQLite -- SQLite is fine for one site, not for many
sites writing concurrently). A separate central dashboard, reachable at
one real company-wide URL, reads from that shared database for
cross-site reporting. This is a genuinely separate, sizable piece of
work from what exists today (see the roadmap below) -- it is the right
target architecture, not something to fake with a single shared SQLite
file over a network mount.

For a single office right now: do (a). It's real, it works today, and
it's what's actually described in sections 1-3 above.

---

## 5. What's still needed before selling this to companies

Being direct about this, since it's the honest answer to "make this a
premium product": everything above makes this a genuinely working,
deployable single-site system. Selling it to companies -- especially
large ones -- needs several things that are real projects on top of
today's work, not settings to flip:

- **Roles.** An admin login now protects the dashboard (one admin
  account). A real product would add an admin-vs-viewer role split.
- **Multi-tenancy / multi-site**, if selling to companies with more
  than one office -- see 4(b) above.
- **A production database.** SQLite (what this uses today) is genuinely
  fine for one site's traffic, but a company-wide product should move
  to Postgres for concurrent writes, backups, and multi-site (b) above.
- **Data privacy/compliance review for biometric data.** Face
  embeddings are biometric data; depending on where you sell this,
  that brings real legal obligations (e.g. BIPA in Illinois, GDPR in
  the EU/UK, and similar laws elsewhere) around consent, storage, and
  deletion rights that a company buying this will ask about.
- **Backups and monitoring** of the database and the service itself
  (is the camera still connected? is the process still running?) --
  today, if the edge box's process crashes, systemd/Task Scheduler
  above will restart it, but nothing alerts anyone that it happened.
- **Payroll/HR system integration** (most companies won't want to
  manually export CSVs forever) -- an API or scheduled export into
  whatever payroll system a customer already uses.

None of this changes what's true today: sections 1-3 describe a real,
working, single-site deployment you can stand up now. This section is
the honest roadmap for turning that into a company-wide sellable
product, not a list of missing basics.
