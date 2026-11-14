# Deploying the Attendance System on AWS with Docker

Simple goal: the office admin opens a link (for example `http://13.234.x.x`
or `https://attendance.yourdomain.com`), logs in, and uses the dashboard.
Nobody runs a terminal.

## The picture in one line

Office camera  -->  (private Tailscale tunnel)  -->  AWS server running Docker
(the app + Caddy)  -->  admin's browser.

- **AWS EC2** = a computer you rent in Amazon's data centre. It stays on 24/7.
- **Docker** = packs the app with Python and every library into one box
  (an "image"), so it runs the same on the server as on your laptop.
- **Docker Compose** = starts the two boxes together with one command:
  the app, and Caddy.
- **Caddy** = the front door. Gives the link, and HTTPS when you use a domain.
- **Tailscale** = a private tunnel so the AWS server can see the office
  camera without opening the camera to the whole internet.

## What is already done in the project (by Claude)

| File | What it does |
|---|---|
| `Dockerfile` | Recipe for the app image (Python 3.11, CPU PyTorch, all requirements, Karachi time zone, Waitress with 16 threads, health check). |
| `docker-compose.yml` | Starts the app + Caddy, restarts them by themselves after a crash or reboot, keeps the database in a volume. |
| `deploy/Caddyfile` | Caddy settings (does not buffer the live video). |
| `.env.example` | The two settings you fill in on the server: camera link and site address. |
| `.dockerignore` | Keeps venv, database, logs and secrets out of the image. |
| `deploy/backup.sh` | Safe copy of the database into `~/backups` (keeps the last 30). |
| `/healthz` route | `http://<server>/healthz` says if the app is up and if camera frames are arriving. No login needed, shows no employee data. |
| Code | `ATTENDANCE_DATA_DIR` env var moves the database and entrance-line file into the Docker volume; `CAMERA_SOURCE` env var gives a starting camera when Settings is empty. On your laptop nothing changes. |

---

## Steps only you can do

### Step 1 - Put the latest code on GitHub
On your laptop, in the project folder:
```
git add .
git commit -m "Add Docker and AWS deployment"
git push
```
Check on github.com that `Dockerfile` and `docker-compose.yml` are there.
(If the repo is private, the server will need a GitHub token to clone it.)

### Step 2 - AWS account and a money alarm
1. Create an account at aws.amazon.com (needs a card), or ask your
   mentor for an IAM user on Hudasoft's account (better).
2. Billing -> Budgets -> create a monthly budget (e.g. $20) with an email alert.
3. Top right, pick the region **Asia Pacific (Mumbai) ap-south-1** (closest to Karachi).

### Step 3 - Launch the server (EC2)
EC2 -> Launch instance:
- Name: `hudasoft-attendance`
- Image: **Ubuntu Server 24.04 LTS**
- Type: **t3.medium** (2 CPU, 4 GB RAM) minimum. `c7i.large` is smoother.
  Do NOT use t2/t3.micro (free tier): 1 GB RAM is too small for the AI models.
- Key pair: create one, download the `.pem` file, keep it safe.
- Network settings -> security group, allow:
  - SSH (22) from **My IP** only
  - HTTP (80) from anywhere
  - HTTPS (443) from anywhere
- Storage: **30 GB** gp3.
- Launch. Then EC2 -> Elastic IPs -> Allocate -> Associate with this
  instance, so the address never changes.

Cost note: t3.medium running all month is roughly $30-35 (check the AWS
pricing page for your region). Stop the instance when you are not testing.

### Step 4 - Open a terminal on the server
Easiest: EC2 -> select instance -> **Connect** -> EC2 Instance Connect ->
Connect. A terminal opens in the browser. (Termius or `ssh -i key.pem
ubuntu@<IP>` also work, but are not required.)

### Step 5 - Install Docker on the server
```
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit
```
Connect again (Step 4), then check: `docker run hello-world`.

### Step 6 - Get the code and fill in settings
```
git clone https://github.com/<your-username>/<repo>.git attendance_system
cd attendance_system
cp .env.example .env
nano .env
```
Leave `CAMERA_SOURCE=` empty for now (Step 8 fills it). Save: Ctrl+O, Enter, Ctrl+X.

### Step 7 - Build and start
```
docker compose up -d --build
docker compose logs -f attendance
```
The first build takes about 10-20 minutes. Ctrl+C stops watching the logs
(the app keeps running). Then open `http://<Elastic IP>` in your browser:
you should see the login page -> create the admin account.
Check `http://<Elastic IP>/healthz` -> `"status": "ok"`.
`camera_ok` is `false` until Step 8.

### Step 8 - Connect a camera (the important part)
The AWS server cannot see your laptop webcam. It needs a camera stream
it can reach. Pick one:

**A. Quick test with a phone (cheapest, good for the demo)**
1. Install **IP Webcam** (Android) on a phone, start server. It shows
   e.g. `http://192.168.1.20:8080`.
2. Install **Tailscale** on the phone and sign in.
3. On the AWS server:
   ```
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
   Open the link it prints and sign in with the SAME Tailscale account.
4. In the Tailscale app, note the phone's address `100.x.y.z`.
5. Dashboard -> Settings -> Camera source: `http://100.x.y.z:8080/video` -> Save.

**B. Real office IP camera / NVR (for the actual office)**
1. Ask the office for the camera's RTSP link, user and password.
   Use the **sub-stream** (lower resolution), e.g. Hikvision
   `rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102`.
   Set the camera to H.264, 10-15 fps, 640x360 or 720p.
2. On one office PC that is always on (same network as the camera),
   install Tailscale and share the office network:
   - Windows: open Command Prompt as admin:
     `tailscale up --advertise-routes=192.168.1.0/24`
     (use the office's real network range)
   - In the Tailscale admin page (login.tailscale.com) -> that machine ->
     Edit route settings -> approve the route.
3. On the AWS server: `sudo tailscale up --accept-routes`
4. Test from the server:
   `docker exec attendance python diagnose_camera.py --url "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"`
   If it reads frames, save the same link in Settings and watch the Live Monitor.

Upload needed from the office: about 0.5-1 Mbps for the sub-stream.

### Step 9 - Set up inside the dashboard
1. Settings: check camera, check-in/out gap, wave sensitivity.
2. Calibrate the entrance line for the real camera view.
3. Register every employee (Register tab) standing in front of the real
   camera, in normal office light.

### Step 10 - Test in the office (one or two days)
- Each person: walk in, wave -> CHECK IN appears. Later wave -> CHECK OUT.
- Note misses, wrong names, delay. Change sensitivity / re-register if needed.
- Check the time on events is Karachi time.

### Step 11 - Backups
```
chmod +x deploy/backup.sh
./deploy/backup.sh
crontab -e
```
Add the line: `0 23 * * * /home/ubuntu/attendance_system/deploy/backup.sh`
(runs every night at 11 pm server time).

### Step 12 (optional) - A proper link with HTTPS
1. Buy or use a domain. In its DNS add an **A record**:
   `attendance` -> your Elastic IP.
2. On the server: `nano .env` -> `SITE_ADDRESS=attendance.yourdomain.com`
3. `docker compose up -d`
Caddy gets a free HTTPS certificate by itself. Share
`https://attendance.yourdomain.com` with the admin.

---

## Everyday commands (on the server)
| Task | Command |
|---|---|
| See if it is running | `docker compose ps` |
| See logs | `docker compose logs -f attendance` |
| Restart | `docker compose restart attendance` |
| Update after a git push | `git pull && docker compose up -d --build` |
| Stop everything | `docker compose down` (data is kept) |

Never run `docker compose down -v`: `-v` deletes the database volume.

## Problems and fixes
| Problem | Fix |
|---|---|
| Page does not open | Security group must allow 80; `docker compose ps` shows both running? |
| Build stops / "Killed" | Not enough RAM: use t3.medium or bigger. |
| `camera_ok: false` | Wrong RTSP link, Tailscale not connected (`tailscale status`), or route not approved. |
| Video lags / freezes | Use the camera sub-stream, 10-15 fps; bigger instance (c7i.large). |
| Wrong times | `.env`/compose sets `TZ=Asia/Karachi`; restart the container. |
| Face model download fails | Server needs internet on first start; check `docker compose logs`. |

## Honest limits
- Every frame travels from the office to AWS and back, so it is a little
  slower than running on a laptop in the office. If the office internet
  drops, check-ins stop until it returns.
- The same Docker files also run on an always-on office PC
  (`docker compose up -d`), which is faster for the camera. AWS is still
  useful so the admin can open the dashboard from anywhere.
- Face data is biometric data: keep the server private (login, HTTPS,
  SSH only from your IP) and get the office's consent.

---

## Free HTTPS link + "Use this browser's webcam"

Browsers only allow the webcam on https:// pages. Free https without buying a domain:

1. On the server: `nano .env` and set
   ```
   SITE_ADDRESS=13-238-25-153.sslip.io
   PUBLIC_IP=13.238.25.153
   ```
   (your Elastic IP, with dashes in the first line). Save: Ctrl+O, Enter, Ctrl+X.
2. `docker compose up -d --build`
3. Open `https://13-238-25-153.sslip.io` (the old `http://13.238.25.153` forwards there).
   Caddy gets a free certificate by itself; the first visit can take ~30 seconds.

Then in the dashboard: Settings -> Camera Source -> **Use this browser's webcam** -> click **Allow**.
That device's webcam is now the camera while its tab stays open (and visible).
Only one browser at a time: choosing another source or another browser stops the previous one.
