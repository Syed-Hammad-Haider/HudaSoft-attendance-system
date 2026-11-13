# Beginner's Setup & Testing Guide
### Attendance Detection System — Phase 1

This guide assumes you have **never run a Python project before**. Follow
it top to bottom, in order. Don't skip steps even if they look obvious.

Every step is marked:
- 🔴 **MANDATORY** — the project will not work without this
- 🟡 **OPTIONAL** — only needed in specific situations (explained inline)

---

## 1. Software You Need to Install

### 🔴 Python 3.10, 3.11, or 3.12
This project is written in Python. Do **not** use Python 3.13+ yet —
some of the AI libraries we use (Ultralytics, MediaPipe) don't
reliably support the newest Python versions.

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 (any 3.11.x version).
3. Run the installer. **Important:** on the first screen, check the box
   that says **"Add Python to PATH"** before clicking Install. This is
   the single most common thing beginners forget.
4. Click "Install Now."

**Mac:**
1. Go to https://www.python.org/downloads/
2. Download the macOS installer for Python 3.11.
3. Run it, click through the installer normally.

**Verify it worked** — open a terminal (Windows: search "Command
Prompt" or "PowerShell"; Mac: search "Terminal") and type:

```bash
python --version
```

If that shows an error on Mac, try:
```bash
python3 --version
```

You should see something like `Python 3.11.9`. If you see `Python
2.7.x` or a command-not-found error, Python isn't installed correctly
— go back and reinstall, making sure to check "Add to PATH" on
Windows.

---

### 🔴 A code editor (IDE)
You technically don't *need* one — you could edit files in Notepad —
but you'll want one. Recommended: **Visual Studio Code (VS Code)**,
free and beginner-friendly.

1. Download from https://code.visualstudio.com/
2. Install it normally (click through defaults).
3. Open it, go to the Extensions icon on the left sidebar (looks like
   4 squares), search for **"Python"** (by Microsoft), and click
   Install. This gives you syntax highlighting and error checking.

---

### 🔴 pip (comes with Python automatically)
`pip` is Python's package installer — it's what downloads all the AI
libraries this project needs. You don't install it separately; it
comes bundled with Python. Verify it's there:

```bash
pip --version
```

If that fails on Mac/Linux, try `pip3 --version`.

---

### 🟡 Git (optional, only if you received the project as a GitHub link)
If I gave you the project as a `.zip` file, skip this entirely — just
unzip it. If you're pulling it from a GitHub repository instead,
install Git from https://git-scm.com/downloads.

---

### Things you do **NOT** need to install yet
Skip these for now — they're covered later in this guide, only if
relevant to your situation:
- CUDA (only if you have an NVIDIA GPU and want faster processing)
- Visual C++ Redistributables (only if Windows gives a specific DLL error)
- FFmpeg (only needed later for CCTV/RTSP streams)
- Camera drivers (only if Windows doesn't already recognize your webcam)

---

## 2. Get the Project Onto Your Computer

1. Unzip `attendance_system.zip` somewhere easy to find, e.g.
   `Desktop/attendance_system`.
2. Open VS Code.
3. Go to **File → Open Folder**, and select the `attendance_system`
   folder you just unzipped.
4. Open a terminal **inside VS Code**: menu bar → **Terminal → New
   Terminal**. This opens a terminal already pointed at your project
   folder, which saves you typing `cd` commands.

---

## 3. Create a Virtual Environment 🔴

A "virtual environment" is an isolated box for this project's
Python packages, so they don't clash with anything else on your
computer. Think of it as a clean, dedicated toolbox just for this
project.

In the VS Code terminal, type:

```bash
python -m venv venv
```

(Mac/Linux: use `python3` if `python` doesn't work.)

This creates a folder called `venv` inside your project. Now
**activate** it:

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):** if you get a "running scripts is disabled"
error, run this once first:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
then try activating again.

**Mac/Linux:**
```bash
source venv/bin/activate
```

**How to know it worked:** your terminal line should now start with
`(venv)` before your usual prompt. If you don't see that, the
environment isn't active — repeat the activation command.

> ⚠️ You need to activate this **every time** you open a new terminal
> to work on this project. If commands later say "package not found"
> even though you installed it, this is almost always why — you
> forgot to activate `venv` first.

---

## 4. Install All Dependencies 🔴

With `(venv)` active, run:

```bash
pip install -r requirements.txt
```

This will download and install everything the project needs:
OpenCV (camera + drawing), Ultralytics (YOLOv8, the person-detector),
`supervision` (ByteTrack, the tracker), MediaPipe (hand/wave
detection), and NumPy.

This will take a few minutes and print a lot of text — that's normal.
Wait for it to finish and return you to a normal prompt with no red
error text at the end.

**If you see red error text:** jump to the [Troubleshooting](#7-troubleshooting-common-errors)
section below — search for the specific error message.

---

## 5. Launch the Application 🔴

Still with `(venv)` active, run:

```bash
python main.py
```

**What should happen:**
- The first time you run this, it will download the YOLOv8n AI model
  (~6 MB) automatically — you need internet for this one-time step.
- A window should pop up titled **"Attendance Detection - Phase 1"**
  showing your live webcam feed.
- If Windows/Mac asks "Allow this app to access your camera?" — click
  **Allow**. If you miss this prompt or click deny, the app will fail
  to open the camera (see Troubleshooting).

**To close it:** click on the video window so it's focused, then press
the **`q`** key on your keyboard. Don't just close the window with the
X button — pressing `q` lets the program shut down cleanly.

---

## 6. Test It Using Your Laptop Webcam

Now let's actually verify each feature works, one at a time.

### ✅ Test 1: Basic detection
Sit or stand in front of your webcam so your whole upper body is
visible. You should see a **green box** appear around you, with a
label like `ID 1 | UNKNOWN` above it.

**Pass condition:** the box tightly follows you as you move, and the
ID number stays the same (doesn't jump to ID 2, ID 3...) as long as
you stay in frame.

### ✅ Test 2: Multiple people
Get a second person (or prop a photo/video of a person on your phone
in front of the camera) into frame at the same time.

**Pass condition:** each person gets their own box and their own
unique ID number (ID 1, ID 2, etc.) at the same time.

### ✅ Test 3: Direction estimation
Walk sideways across the frame — from one edge toward the other —
for about 2 seconds.

**Pass condition:** the label above your box changes from `UNKNOWN` to
either `ENTERING` or `LEAVING` partway through your walk. Walk the
other direction and confirm the label flips.

> Note: which direction shows as "ENTERING" depends on a setting
> (`entering_is_left_to_right` in `config/settings.py`) — see Section
> 9 below. It's fine if it seems "backwards" right now; that's
> expected until you set it for your real camera position.

### ✅ Test 4: Genuine wave detection
Face the camera and wave your hand back and forth, like waving hello,
for about 1-2 seconds (at least 3-4 side-to-side swings).

**Pass condition:** within a second or two, a dark banner appears
below your box saying something like:
```
✅ Suggested Action: CHECK IN  (Dir: ENTERING)
```
It should disappear on its own after about 2.5 seconds.

### ✅ Test 5: Wave rejection (no false triggers)
Just hold your hand up and hold it still (don't wave it) for 3
seconds. Then try scratching your head or adjusting your hair.

**Pass condition:** **no** Suggested Action banner appears for either
of these. This confirms the system isn't falsely triggering on
non-wave hand movement.

### ✅ Test 6: Cooldown
Wave genuinely (like Test 4), then immediately wave again right away.

**Pass condition:** you get exactly one banner, not two back-to-back.
Wait 5+ seconds and wave again — you should get a second banner. This
confirms the cooldown is working.

### ✅ Test 7: Two people waving independently
With two people in frame, have both wave at the same time.

**Pass condition:** each person gets their **own** banner under their
**own** box, roughly simultaneously, and one person's wave doesn't
suppress or trigger the other's message.

If all 7 pass, the core system is working correctly.

---

## 7. Troubleshooting Common Errors

| Error message / symptom | What it means | Fix |
|---|---|---|
| `'python' is not recognized as an internal or external command` | Python isn't installed or not on PATH | Reinstall Python, check "Add Python to PATH" during install |
| `No module named 'cv2'` / `No module named 'ultralytics'` etc. | Dependencies not installed, or venv not activated | Make sure `(venv)` shows in your terminal, then re-run `pip install -r requirements.txt` |
| `Could not open webcam source '0'` | Camera is in use by another app, or not detected | Close Zoom/Teams/other apps using the camera; check Windows Camera privacy settings (Settings → Privacy → Camera → allow desktop apps) |
| Black/frozen video window | Camera opened but isn't sending frames | Try unplugging/replugging a USB webcam; try a different USB port; restart the app |
| App crashes immediately with a long red error ending in `ImportError: DLL load failed` (Windows only) | Missing Visual C++ Redistributable | Install "Microsoft Visual C++ Redistributable (x64)" from Microsoft's site — see Section 12 |
| Very slow / laggy video (low FPS shown top-left) | Normal on an older CPU with no GPU | See Section 12 on optional GPU/CUDA setup |
| `pip install` fails with a permissions error | Trying to install outside the virtual environment, or need admin rights | Confirm `(venv)` is active; on Mac/Linux, never use `sudo pip install` for this project |
| Window opens then closes instantly, no error visible | Error happened but terminal window closed too fast to read (rare, usually only if double-clicking a script instead of running from terminal) | Always run via `python main.py` typed into a terminal, not by double-clicking a file |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Wrong mediapipe version got installed | Run `pip install mediapipe==0.10.14` manually to force the correct version |

If you hit an error not listed here: copy the **last 5-10 lines** of
red error text (not the whole wall of text) and share it — that's
almost always enough to diagnose the issue.

---

## 8. Replacing the Laptop Webcam With a USB Webcam

Good news: this requires **zero code changes** in most cases.

1. Plug in the USB webcam.
2. Close the app if it's running.
3. Run `python main.py` again.

Windows/Mac usually auto-selects a newly plugged-in USB camera as the
default, so `CameraConfig.source = 0` (the default in
`config/settings.py`) will often just work.

**If it opens your laptop's built-in camera instead of the USB one:**
open `config/settings.py`, find this line:
```python
source: object = 0
```
Try changing `0` to `1` (or `2`), save the file, and run again. Camera
index numbers depend on plug-in order — there's no way to know the
right number in advance, so this is quick trial and error. A tip: if
you have 2 cameras, `0` is usually the built-in one and `1` is usually
the USB one.

---

## 9. Connecting to a CCTV Camera

This is the part that depends most on what kind of CCTV system the
software company actually has, so read the option that matches your
situation.

### 🅰️ Option A: USB CCTV camera (plugs directly into a computer)
Identical to Section 8 above — try `source = 0`, `1`, `2` until you
find the right index.

### 🅱️ Option B: IP Camera via RTSP stream (most common for real CCTV)
Most modern CCTV/IP cameras stream video over your network using a
protocol called **RTSP**. Instead of a camera index number, you give
the program a URL that looks like this:

```
rtsp://username:password@192.168.1.64:554/Streaming/Channels/101
```

**How to find your camera's exact RTSP URL:**
1. Check the camera or NVR's manual/settings page — most have an
   "RTSP" or "streaming" settings section showing the exact URL
   format for that brand (Hikvision, Dahua, and others all use
   slightly different URL paths).
2. Or check the camera's web interface (type its IP address into a
   browser) — there's usually a network/streaming settings page
   showing it directly.
3. The IP address (e.g. `192.168.1.64`) is the camera's address on
   the company's local network — someone on the IT/facilities side
   will likely already know this, or you can find it via the router's
   connected-devices list.

**Once you have the URL**, open `config/settings.py` and change:
```python
source: object = 0
```
to:
```python
source: object = "rtsp://username:password@192.168.1.64:554/Streaming/Channels/101"
```

**One code change is needed here** (this is the one place I flagged
as needing a small addition later): open `camera/factory.py` and
you'll see a comment showing exactly where to add RTSP detection. I
can write this exact code for you when you're ready to test with a
real RTSP camera — it's about 5 lines.

### 🅲️ Option C: Camera connected through an NVR/DVR box
This is common in offices — individual cameras plug into a central
recorder box (NVR/DVR), and *that box* is what's on the network.

- You almost never connect to the camera directly in this setup.
  Instead, you get an RTSP URL **from the NVR/DVR itself**, which
  proxies one specific camera's feed.
- The process is otherwise identical to Option B — you just get the
  URL from the NVR's settings/manual instead of the camera directly.
  NVR brand manuals usually have a section literally called "RTSP URL
  format" with a fill-in-the-blank example.
- Ask whoever manages the NVR for: the NVR's IP address, the RTSP
  port (often 554), the channel number for the specific camera you
  want, and login credentials.

---

## 10. Testing Everything at Home Before Taking It to the Company

You can validate almost everything except the *actual* company camera
using just your laptop:

1. **Run all 7 tests in Section 6** using your laptop webcam — this
   validates 100% of the detection/tracking/wave/direction logic,
   since that logic doesn't know or care what kind of camera is
   feeding it frames.
2. **Simulate an RTSP camera at home**, if you want to test that code
   path specifically, using your **phone** as a fake IP camera:
   - Install a free app like "IP Webcam" (Android) or similar (iOS
     has several options too).
   - It'll show an RTSP URL like `rtsp://192.168.1.23:8080/h264_ulaw.sdp`
     right on your phone's screen once the app is running.
   - Put that URL into `config/settings.py` as described in Section 9,
     Option B, and confirm the video window shows your phone's camera
     feed instead of your laptop's.
   - This proves the RTSP code path works before you ever touch the
     company's real hardware.
3. **Double check direction orientation** (Section 6, Test 3) and
   decide now whether `entering_is_left_to_right` should be `True` or
   `False` for how you plan to mount the real camera — you can reason
   about this at home even without the real camera present.

If all of this passes at home, the only genuinely new variable at the
company is the specific RTSP URL format for their hardware — which is
a config change, not a code change.

---

## 11. Deploying to the Company's Actual CCTV Camera

When you get there:

1. Bring your laptop with the project already set up and tested (from
   Section 10) — don't set anything up for the first time on-site.
2. Get the RTSP URL, IP address, and any login credentials from
   whoever manages their camera system (Section 9 explains what to
   ask for).
3. Make sure your laptop is on the **same network** as the camera/NVR
   — plug into their network via Ethernet if possible, or join their
   Wi-Fi, since RTSP streams generally don't work across the public
   internet without extra setup (VPN, port forwarding, etc.) that's
   out of scope for this phase.
4. Update `config/settings.py`'s `source` value to their real RTSP
   URL.
5. Run `python main.py` exactly as before — **no other code changes
   needed** if the RTSP branch in `camera/factory.py` was already
   added and tested with your phone at home (Section 10, step 2).
6. Re-run the same 7 tests from Section 6 using their real camera and
   real foot traffic, and adjust `entering_is_left_to_right` if
   directions come out reversed for their camera's mounting
   orientation.

This is exactly why the project was built with `camera/` as a
separate, isolated module in the first place — the swap really is
just a config value once the RTSP code path itself has been tested.

---

## 12. Extra Software: When You Actually Need It

| Software | Mandatory? | When you need it |
|---|---|---|
| **CUDA + NVIDIA GPU drivers** | 🟡 Optional | Only if you have an NVIDIA graphics card AND video feels laggy/slow on CPU. Skip entirely on a laptop with no dedicated NVIDIA GPU, or if performance is already acceptable. Install from https://developer.nvidia.com/cuda-downloads, then set `device: str = "cuda"` in `config/settings.py`'s `DetectionConfig`. |
| **Visual C++ Redistributable (Windows only)** | 🟡 Optional | Only if you get a `DLL load failed` error when running `python main.py` on Windows. Download "Microsoft Visual C++ Redistributable x64" from Microsoft's official site, install, restart your computer, try again. |
| **FFmpeg** | 🟡 Optional | Only needed if OpenCV has trouble decoding a specific RTSP camera's video codec (rare, but happens with some older CCTV brands). You'll know you need it if the RTSP video window opens but shows no image / errors mention "codec." Install via your OS package manager (Mac: `brew install ffmpeg`; Windows: download from ffmpeg.org and add to PATH). |
| **Camera drivers** | 🟡 Optional | Only if Windows doesn't recognize a USB webcam at all (it won't even show up in Windows' own Camera app). Usually Windows auto-installs these; only check the manufacturer's website if the camera is completely invisible to the OS. |
| **Python 3.11** | 🔴 Mandatory | Covered in Section 1. |
| **pip + requirements.txt packages** | 🔴 Mandatory | Covered in Section 4. |

**Bottom line:** for a normal laptop webcam test, you need **only**
Python + the packages in `requirements.txt`. Everything else in this
table is situational and you'll know you need it because something
specific breaks — you won't need to guess upfront.

---

## Quick Reference: Every Command in Order

```bash
# One-time setup
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# Every time you want to run it
venv\Scripts\activate          # Windows (if not already active)
python main.py

# To quit
# (click the video window, press 'q')
```
