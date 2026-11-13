# What This Project Does (Simple Explanation)

*Last updated: after fixing the "checked in multiple times" bug and
adding a simpler calibration slider, fullscreen window, and clearer
hand tracking.*

## In one sentence
This program watches a camera, finds people, lets you drag a vertical
line onto the real doorway, and gives each person exactly ONE "check
in" suggestion and ONE "check out" suggestion — no matter how many
times they walk back and forth or wave.

## The big fix: no more double check-ins

**What was wrong before:** if you waved a couple of times, or crossed
the line more than once, the program could suggest "CHECK IN" for you
more than once. That's clearly not right — you should only be checked
in once per visit.

**What we fixed:** the program now remembers, for each person, exactly
one thing: are they "not checked in yet," "checked in," or "checked
out"? Based on that:

- If you're **not checked in yet**, only a CHECK IN suggestion can
  ever appear for you.
- Once you're **checked in**, waving again does nothing — the program
  ignores it, because you already have your suggestion. Only walking
  back out and waving will produce a CHECK OUT.
- Once you're **checked out**, further waves are ignored too, until
  you actually leave the camera's view and come back (which starts
  you fresh, as a "new" visit).

You can walk in and out, pause, hesitate, wave several times — none of
that matters. Only real, complete entries and exits, each with a
genuine wave, ever produce a suggestion, and only one of each.

## The entrance line is now a simple slider

Before, you had to click two separate points to draw the line, which
could come out crooked. Now:

- Press **C**.
- Just move your mouse left and right — a line follows your mouse,
  automatically stretching from the very top to the very bottom of the
  screen (you never have to adjust its height, only its left-right
  position).
- Click once to lock it in place.
- Press **F** any time to swap which side is IN and which is OUT.

## The window can now go fullscreen

**What was wrong:** the video window was stuck small no matter what
you tried — even the "maximize" button did nothing.

**Why:** it turns out the window was created in a special "locked
size" mode by mistake — a genuine bug, not anything about your camera
or PC. The video itself was always fine; the window just refused to
resize.

**Fixed:** the window now resizes and maximizes normally, and there's
a new **M** key that jumps straight to true fullscreen — good for
showing this off on a big screen at the entrance.

## Hand tracking now shows more detail, and reacts faster

Before, you just saw a plain green box around a hand. Now you see an
actual skeleton — dots at the fingertips, knuckles, and wrist,
connected by lines — so you can see exactly what the program is
tracking, not just a rough box. It also updates every single frame
again (we'd slowed it down in an earlier version to help performance,
but based on your feedback that made it feel sluggish, so we sped it
back up and found other ways to keep the program fast instead).

**A wave is also recognized faster now** — roughly in half the time it
used to take — while we specifically tested it against fake "walking,"
"jogging," and even "running" arm-swinging motions to make sure none
of those falsely trigger a wave.

**One honest limitation:** if someone is genuinely sprinting through
the entrance while also swinging their arm really fast, it can — very
rarely — still be mistaken for a wave. We tested this specifically and
confirmed there's currently no way to tell a "fast wave" and
"sprinting arm swing" apart using hand movement alone, because they
genuinely look almost identical to the camera. A proper fix would also
check how fast the person's whole body is moving, which is a
reasonable next step, not something we've built yet — deliberately, so
we don't add complexity for a very rare situation (most people walk,
not sprint, into an office).

## What this program is NOT doing (on purpose)

- ❌ It does **not** know who anyone is (no face recognition).
- ❌ It does **not** save any record of who came or left.
- ❌ It does **not** use a database.
- ❌ It does **not** log anyone in or out anywhere.

This is intentional — this is only **Phase 1**, the "eyes and brain"
that watches and suggests. A future phase could add face recognition
and an actual attendance log on top of this.

## Is this the final, perfect version now?

Being honest, the same way as every previous update: this fixes real,
important bugs you found, and it's meaningfully more solid. But two
things remain real, open limitations:

- If someone's tracking ID accidentally changes (which can still
  happen when people stand close together or briefly block each
  other), their "checked in" memory can reset, since Phase 1 has no
  way to recognize a person's face or identity yet. This is one of the
  main reasons face recognition is planned for the next phase.
- This still hasn't been tested on your actual workplace CCTV camera —
  only on a laptop webcam and test photos. That real-world test is
  still the next important step.

## Why build it this way first?

Every one of these fixes came directly from you actually using the
program and telling us what felt wrong. That's exactly how this is
supposed to go — build something solid, test it for real, fix what
testing reveals, and be honest about what's still left. Rushing to
claim "done and perfect" before that process finishes would make the
system less trustworthy, not more.

## What changed in this latest update

You reported the wave still wasn't detecting reliably, asked about
connecting a real CCTV camera, and noticed the video seemed "flipped"
left-right. Here's what we actually found and did about each:

**The wave-detection bug — a real one, now fixed.**
The program was deciding "was that a real wave?" by counting how many
*video frames* passed between hand movements — not how many actual
*seconds*. That sounds like a small technical detail, but it's not:
your laptop's actual processing speed (frames per second) can vary a
lot, especially with everything this program is doing at once
(finding people, tracking them, watching hands). If your computer was
running slower than expected, "5 frames" might secretly mean "1.5
seconds" instead of the roughly half-second it was designed for — and
that mismatch could make real waves fail to register at all. We fixed
this by having the program measure actual real time (like a
stopwatch) instead of counting frames. This should make wave detection
noticeably more reliable regardless of how fast or slow your
computer's processing happens to be at any given moment.

**Honest new thing we learned:** if your computer's actual video
processing speed is very slow (roughly below 8 frames per second — you
can check the FPS number shown in the top corner of the video), wave
detection may still struggle, but for a different, more fundamental
reason: the program simply isn't "looking" often enough to catch a
quick hand motion, the same way a slow-motion camera set to too few
pictures-per-second would miss a fast clap. If that turns out to be
the case, the fix is speeding up the overall program (which we've
already done some of), not further adjusting the wave-detection
settings.

**Hand crop size increased.**
The program only looks for hands in a small box around each person, to
keep things fast. That box was slightly too tight — a full, natural
wave motion could swing your hand outside of it, causing the program
to briefly "lose" your hand mid-wave. We made that box noticeably
bigger so a full wave stays inside it.

**The "flipped" video — turns out this isn't a bug.**
This one is worth explaining clearly: the video shows exactly what the
camera sees, unflipped. If you face the camera and move your right
hand, it correctly shows up on the left side of the screen — because
that's genuinely where your hand is from the camera's point of view.
The "mirrored" feeling you're used to only comes from apps like Zoom
or your phone's selfie camera, which flip the image on purpose so it
feels like looking in a mirror. A real security/entrance camera should
**never** do that, because the whole point of the entrance line is to
match the real, physical layout of the room — flipping it would
actually make the IN/OUT sides wrong. That said, we added an optional
button (press **V**) that flips the preview just for your own comfort
while testing on your laptop. If you use it, press **C** afterward to
redraw the entrance line, since flipping changes which side is which.

**How to connect a real CCTV camera — the short version:**
1. Get the camera's web address (called an "RTSP URL") from its
   settings — every camera brand shows this a little differently.
2. Open the settings file and paste that address in, replacing the
   webcam setting.
3. Make sure your laptop and the camera are on the same network.
4. Run the program exactly the same way as before.
5. Press **C** to draw the entrance line against the real doorway.

See `SETUP_GUIDE.md` for the full, detailed version of these steps.

## Big update: the system now knows WHO people are

Everything above was "Phase 1" — the system could see people, watch
for waves, and guess CHECK IN or CHECK OUT, but it never knew *who*
anyone actually was. That's now changed.

### What's new, in plain words

**1. You can register employees' faces.**
Run a new tool (`register_employee.py`), type in someone's name, and
the camera guides you through taking a few pictures of their face —
looking straight ahead, then slightly left, then slightly right.
Taking multiple angles like this makes recognition much more reliable
later, since a real camera won't always catch someone facing it
perfectly straight-on.

**2. The system now recognizes people by name.**
Once someone is registered, the box around them shows their actual
name instead of just "ID 3." If someone hasn't been registered, it
still shows "ID 3" and clearly marks them as unrecognized — the
system never guesses or makes up a name.

**3. Attendance is now actually recorded, not just suggested.**
Before, a "Suggested Action" was just a message that popped up and
disappeared. Now, when a recognized employee waves near the entrance,
the system writes a real, permanent record — like an actual entry in
a logbook — saying "Ahmed Khan, CHECK IN, at this exact time." This is
saved in a small database file on the computer, the same basic idea as
a spreadsheet, just handled by the program automatically.

**4. This fixes a real weak spot from before.**
Earlier, if the camera briefly lost track of someone (like when
another person walked in front of them for a second), the program
could sometimes "forget" that this person had already checked in,
because it was only remembering things in short-term memory tied to a
temporary tracking number. Now that people have a real, recognized
identity, their check-in status is saved properly and doesn't get
lost just because the camera briefly lost track of their outline. We
actually tested this directly: we simulated someone getting a
brand-new tracking number (as if they'd been briefly hidden from
view) and confirmed the system still correctly remembered they'd
already checked in.

**5. Nobody's attendance gets recorded by accident.**
If the system isn't confident about who it's looking at, it simply
won't record anything for that person — it would rather show "I don't
know who this is" than risk crediting the wrong person's attendance.

### An honest note about accuracy

We tested the face-matching carefully using real, different people's
faces, and the system correctly told them apart with a lot of room to
spare — it wasn't a close call. But we're being upfront about one
thing we couldn't fully test yet: how well it recognizes the SAME
person on a different day, in different lighting, from a slightly
different angle than their registration photos. That's the real
day-to-day test, and it's worth trying with your actual coworkers
before fully trusting it — the settings can be adjusted if needed
based on how that goes.

### What this still doesn't do (on purpose, for now)

- ❌ No admin screen yet for browsing attendance history or managing
  employees — that data is all safely saved in the database, just not
  wrapped in a nice screen to browse yet.
- ❌ Doesn't yet handle someone who's never been registered getting
  "lost and re-found" by the camera as gracefully as a registered
  person does — that part specifically needed a real, known identity
  to fix properly.

## One more update: making detection itself faster

You asked whether switching to different underlying technology (like
a different version of the detection software) could make everything
faster and more accurate. Here's what we actually found by testing
it, honestly:

**We tried switching the "engine" that runs person detection** to a
different format (called ONNX) that's often faster on other
computers. When we actually tested it here, it came back the same
speed or slower — not what we expected. Being honest about this
matters: the computer used for testing only has a single processing
core, which isn't a fair stand-in for a real laptop, so the results
weren't trustworthy either way. Rather than guess and possibly make
things worse, we left this as-is, and left clear instructions in the
code for how to try it yourself on your real laptop, where it might
actually help.

**What we DID find and fix: reducing how big of a picture the
detection looks at.** Every video frame was being resized to a fairly
large size before the computer searched it for people, even though a
smaller size works just as well for finding a person. We tested
several smaller sizes and found one (416, if you're curious) that's
about **2.5 times faster** while still reliably finding every person
in a test photo, with plenty of safety margin. A slightly smaller
size after that was even faster but started missing people
occasionally, so we picked the safer, still-fast option instead of
squeezing out every last bit of speed at the cost of reliability.

This should make the whole system noticeably snappier — and since
speed is directly what makes wave detection able to "keep up" with a
real hand movement (explained in the earlier update), this should
help wave detection feel more responsive too, on top of the timing
fix from before.

## The very last check: making sure earlier fixes still work together

Before calling this finished, we did one more careful pass: checking
whether earlier fixes, made in different rounds, might have quietly
interfered with each other. This is a really easy thing to miss — a
fix made last week can pass its own test perfectly, while still
subtly breaking when combined with a DIFFERENT fix made this week.

We found exactly two small examples of this:

**1. Widening the "hand search area" (done to catch full wide waves)
had a side effect on how "big" a wave needed to look.** Since the
system measures wave size as a percentage of the search area, making
that area bigger meant the same real wave now looked like a slightly
smaller percentage. A big, enthusiastic wave was never affected, but
a more modest, casual wave could have been missed. Fixed by adjusting
the sensitivity to match.

**2. A "how quickly must you swing your hand back and forth" setting
was sitting exactly on the edge for a normal-paced wave**, with zero
safety margin for the small, natural timing variations every real
camera has. We widened that margin slightly — just enough to reliably
catch a normal-paced wave, while re-testing carefully to make sure
this didn't let false triggers back in from people just walking
quickly. It didn't — we checked.

Both of these were caught specifically by re-testing the WHOLE system
together at the end, not just each individual piece on its own — which
is exactly the right way to actually finish a project like this,
rather than assuming everything is fine just because each part passed
its own test at some point.
